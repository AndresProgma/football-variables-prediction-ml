"""ML Microservice — FastAPI wrapper del pipeline de predicción."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from knime_workflow_converter import main as run_pipeline, predecir_partido
from predecir_v2 import predecir_partido_v2

DATASET = os.getenv("DATASET_PATH", "/data/creando_dataset_modificado.xlsx")
RECORD_FILE = Path(os.getenv("DATA_DIR", "/data")) / "record_historico.json"

app = FastAPI(title="ML Service — Predicción UCL")

_resultados: dict[int, dict] = {}
_status: dict[int, str] = {}
_filepaths: dict[int, str] = {}


def _train_bg(eval_id: int, filepath: str) -> None:
    try:
        results = run_pipeline(filepath)
        _resultados[eval_id] = results
        _status[eval_id] = "ready"
    except Exception as exc:
        _status[eval_id] = f"error:{exc}"


def _get_results(eval_id: int) -> dict:
    if eval_id in _resultados:
        return _resultados[eval_id]
    filepath = _filepaths.get(eval_id, DATASET)
    if not Path(filepath).exists():
        raise HTTPException(409, f"Modelo {eval_id} no en memoria y dataset no encontrado: {filepath}")
    results = run_pipeline(filepath)
    _resultados[eval_id] = results
    return results


class TrainRequest(BaseModel):
    filepath: str = ""


@app.post("/train/{eval_id}")
def train(eval_id: int, data: TrainRequest, background_tasks: BackgroundTasks):
    filepath = data.filepath or DATASET
    if not Path(filepath).exists():
        raise HTTPException(422, f"Dataset no encontrado: {filepath}")
    _filepaths[eval_id] = filepath
    _status[eval_id] = "training"
    background_tasks.add_task(_train_bg, eval_id, filepath)
    return {"eval_id": eval_id, "status": "training"}


@app.get("/status/{eval_id}")
def status(eval_id: int):
    return {"eval_id": eval_id, "status": _status.get(eval_id, "not_started")}


class PredictRequest(BaseModel):
    equipo1: str
    equipo2: str
    eval_id: int
    n_runs: int = 20
    fase: str = "Liga"


@app.post("/predict")
def predict(data: PredictRequest):
    results = _get_results(data.eval_id)
    salida = predecir_partido(
        data.equipo1, data.equipo2, results,
        n_runs=data.n_runs, fase=data.fase,
    )
    if salida is None:
        raise HTTPException(500, "predecir_partido() no devolvió datos")
    return salida


@app.get("/elos/{eval_id}")
def elos(eval_id: int):
    results = _get_results(eval_id)
    team_elos = results.get("team_elos", {})
    ranking = sorted(
        ({"equipo": t, "elo": round(float(e), 1)} for t, e in team_elos.items()),
        key=lambda x: -x["elo"],
    )
    return {"ranking": ranking, "total": len(ranking)}


@app.get("/metricas/{eval_id}")
def metricas(eval_id: int):
    results = _get_results(eval_id)
    return {
        "test": results["results"].to_dict(orient="records"),
        "cv": results["cv_results"].to_dict(orient="records"),
    }


@app.get("/feature-importance/{eval_id}")
def feature_importance(eval_id: int, top: int = 15):
    results = _get_results(eval_id)
    fi = results.get("feature_importance", []) or []
    return {"top_features": fi[:top], "total": len(fi)}


@app.get("/predicciones-test/{eval_id}")
def predicciones_test(eval_id: int):
    results = _get_results(eval_id)
    pred_df = results.get("predictions")
    if pred_df is None:
        raise HTTPException(500, "No hay predicciones de test")
    return {"predicciones": pred_df.to_dict(orient="records"), "tipo": "test_split"}


@app.post("/record-historico")
def generar_record_historico(background_tasks: BackgroundTasks, n_partidos: int = 35):
    """Genera record_historico.json usando predecir_v2 (sin leakage)."""
    def _bg():
        df = pd.read_excel(DATASET)
        if "Fecha" in df.columns:
            df["_f"] = pd.to_datetime(df["Fecha"], errors="coerce")
            df = df.sort_values("_f").drop(columns=["_f"]).reset_index(drop=True)

        ultimos = df.tail(n_partidos).reset_index(drop=True)
        resultados = []

        for _, row in ultimos.iterrows():
            e1 = str(row["Equipo1"])
            e2 = str(row["Equipo2"])
            fecha = str(row.get("Fecha", ""))[:10] if pd.notna(row.get("Fecha")) else None
            fase = str(row.get("Fase", "Liga"))
            try:
                pred = predecir_partido_v2(e1, e2, fecha=fecha, fase=fase, n_runs=10)
                g1r = int(row["EQUIPO1_GOLES"]) if pd.notna(row.get("EQUIPO1_GOLES")) else None
                g2r = int(row["EQUIPO2_GOLES"]) if pd.notna(row.get("EQUIPO2_GOLES")) else None
                resultado_real = (
                    "Win" if g1r is not None and g2r is not None and g1r > g2r else
                    "Loss" if g1r is not None and g2r is not None and g1r < g2r else
                    "Draw" if g1r is not None and g2r is not None else None
                )
                resultados.append({
                    "equipo1": e1, "equipo2": e2, "fecha": fecha or "—", "fase": fase,
                    "goles_real": f"{g1r}–{g2r}" if g1r is not None else "—",
                    "resultado_real": resultado_real,
                    "consenso": pred["consenso"],
                    "modelos": pred.get("modelos", []),
                    "acierto": pred.get("acierto_consenso"),
                })
            except Exception as exc:
                resultados.append({"equipo1": e1, "equipo2": e2, "fecha": fecha or "—", "error": str(exc)})

        RECORD_FILE.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

    background_tasks.add_task(_bg)
    return {"status": "generating", "n_partidos": n_partidos}


@app.get("/health")
def health():
    return {"status": "ok", "modelos_en_memoria": list(_resultados.keys())}
