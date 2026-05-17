"""
API REST — Predicción de Fútbol, Champions League.

Recursos:
  /partidos     — GET, GET{id}, DELETE (soft)  — datos del Excel
  /evaluaciones — POST, GET, GET{id}, DELETE (soft)  — corre main()
  /predicciones — POST, GET, GET{id}, DELETE (soft)  — corre predecir_partido()

ORM: SQLModel + SQLite (futbol.db)
Soft delete: campo activo=False en vez de borrar el registro

Arrancar:
    pip install sqlmodel
    uvicorn api:app --reload
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Field, Session, SQLModel, create_engine, select, func

from knime_workflow_converter import main as run_pipeline, predecir_partido

_DEFAULT_DATASET = Path(__file__).parent / "creando_dataset_modificado.xlsx"
DATASET = os.getenv("DATASET_PATH", str(_DEFAULT_DATASET))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./futbol.db")

engine = create_engine(DATABASE_URL)

# Pipeline results guardados en memoria (demasiado pesados para DB)
_resultados_pipeline: dict[int, dict] = {}
# Estado del entrenamiento en background: "training" | "ready" | "error:..."
_training_status: dict[int, str] = {}


# ---------------------------------------------------------------------------
# Modelos ORM — cada clase es una tabla en SQLite
# ---------------------------------------------------------------------------

class Partido(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    equipo1: str
    equipo2: str
    fase: str
    goles_e1: int
    goles_e2: int
    fecha: Optional[str] = None
    activo: bool = True


class Evaluacion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filepath: str
    resultados: str       # JSON con accuracy por modelo
    cv_resultados: str    # JSON con cross-validation
    activo: bool = True


class Prediccion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    equipo1: str
    equipo2: str
    evaluacion_id: int = Field(foreign_key="evaluacion.id")
    n_runs: int
    output: str           # Salida completa de predecir_partido()
    activo: bool = True


class PrediccionTrack(SQLModel, table=True):
    """Predicción guardada para track record público — se compara con el
    resultado real cuando el partido se juegue (auto-resuelta)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    equipo1: str
    equipo2: str
    fecha_partido: Optional[str] = None       # fecha esperada del partido (YYYY-MM-DD)
    fecha_prediccion: str                     # cuándo se hizo la predicción
    evaluacion_id: int = Field(foreign_key="evaluacion.id")
    pred_consenso: str                        # Win / Draw / Loss
    prob_win: float
    prob_draw: float
    prob_loss: float
    elo_e1: float
    elo_e2: float
    fase: str = "Liga"
    # Auto-resuelto cuando llega el resultado
    resultado_real: Optional[str] = None      # Win / Draw / Loss desde la perspectiva E1
    g1_real: Optional[int] = None
    g2_real: Optional[int] = None
    acierto: Optional[bool] = None
    activo: bool = True


# ---------------------------------------------------------------------------
# Schemas de entrada (no son tablas)
# ---------------------------------------------------------------------------

class PartidoUpdate(SQLModel):
    fase: Optional[str] = None
    goles_e1: Optional[int] = None
    goles_e2: Optional[int] = None
    fecha: Optional[str] = None


class EvaluacionCreate(SQLModel):
    filepath: str = DATASET


class EvaluacionUpdate(SQLModel):
    filepath: str = DATASET


class PrediccionCreate(SQLModel):
    equipo1: str
    equipo2: str
    evaluacion_id: int
    n_runs: int = 20


class PrediccionUpdate(SQLModel):
    equipo1: Optional[str] = None
    equipo2: Optional[str] = None
    n_runs: Optional[int] = None


# ---------------------------------------------------------------------------
# Startup: crear tablas y cargar Excel
# ---------------------------------------------------------------------------

def _cargar_excel(session: Session) -> int:
    """Sincroniza la tabla Partido con el Excel.
    Agrega los partidos que faltan en la DB (incremental). Devuelve el número de
    filas insertadas. Seguro de llamar múltiples veces."""
    try:
        df = pd.read_excel(DATASET)
    except FileNotFoundError:
        return 0

    ids_en_db = set(session.exec(select(Partido.id).where(Partido.activo == True)).all())

    nuevos = 0
    for _, row in df.iterrows():
        pid = int(row["Partido_id"]) if pd.notna(row.get("Partido_id")) else None
        if pid is not None and pid in ids_en_db:
            continue  # ya existe
        goles_e1 = row.get("EQUIPO1_GOLES")
        goles_e2 = row.get("EQUIPO2_GOLES")
        partido = Partido(
            id=pid,
            equipo1=str(row.get("Equipo1", "")),
            equipo2=str(row.get("Equipo2", "")),
            fase=str(row.get("Fase", "")),
            goles_e1=int(goles_e1) if pd.notna(goles_e1) else 0,
            goles_e2=int(goles_e2) if pd.notna(goles_e2) else 0,
            fecha=str(row["Fecha"]) if pd.notna(row.get("Fecha")) else None,
        )
        session.add(partido)
        nuevos += 1

    if nuevos:
        session.commit()
    return nuevos


def _run_training_background(evaluacion_id: int, filepath: str) -> None:
    """Entrena el pipeline en un hilo secundario y actualiza DB + memoria."""
    try:
        _training_status[evaluacion_id] = "training"
        results = run_pipeline(filepath)
        # Actualizar DB primero, luego memoria (evita race condition con status endpoint)
        with Session(engine) as sess:
            ev = sess.get(Evaluacion, evaluacion_id)
            if ev:
                ev.resultados = results["results"].to_json(orient="records")
                ev.cv_resultados = results["cv_results"].to_json(orient="records")
                sess.add(ev)
                sess.commit()
        _resultados_pipeline[evaluacion_id] = results
        _training_status[evaluacion_id] = "ready"
    except Exception as exc:
        _training_status[evaluacion_id] = f"error: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        _cargar_excel(session)
        # Si no hay ninguna evaluación entrenada, iniciar entrenamiento automático
        evals = session.exec(select(Evaluacion).where(Evaluacion.activo == True)).all()
        if not evals and Path(DATASET).exists():
            ev = Evaluacion(filepath=DATASET, resultados="[]", cv_resultados="[]")
            session.add(ev)
            session.commit()
            session.refresh(ev)
            t = threading.Thread(target=_run_training_background, args=(ev.id, DATASET), daemon=True)
            t.start()
    yield


app = FastAPI(title="Predicción Fútbol UCL", lifespan=lifespan)

# CORS — permite que el frontend en otro host (Vercel, Netlify, dominio propio)
# llame a este API. Configurable vía env var ALLOWED_ORIGINS (lista separada por comas).
_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir dashboard estático desde /static y / (raíz redirige a index.html)
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard_root():
        return FileResponse(_STATIC_DIR / "index.html")


def get_session():
    with Session(engine) as session:
        yield session


def _get_or_run_pipeline(evaluacion_id: int, session: Session) -> dict:
    """Devuelve los resultados en memoria; si el servidor se reinició, re-entrena en background."""
    if evaluacion_id in _resultados_pipeline:
        return _resultados_pipeline[evaluacion_id]
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(status_code=404, detail=f"Evaluación {evaluacion_id} no encontrada")
    # Si ya está entrenando en background, informar al cliente
    status = _training_status.get(evaluacion_id, "")
    if status == "training":
        raise HTTPException(status_code=503, detail="Modelo entrenando, intenta de nuevo en unos segundos")
    # Re-entrenar en background (puede pasar al despertar de un sleep de Render)
    t = threading.Thread(target=_run_training_background, args=(evaluacion_id, ev.filepath), daemon=True)
    t.start()
    raise HTTPException(status_code=503, detail="Modelo no está en memoria — entrenando en background, reintenta en ~90 s")


def _resolver_predicciones_para_partido(partido: Partido, session: Session) -> int:
    """Cuando un partido tiene resultado, marca las predicciones tracked que
    coincidan (mismos equipos en cualquier orden) como acertadas o falladas."""
    tracks = session.exec(
        select(PrediccionTrack)
        .where(PrediccionTrack.activo == True)
        .where(PrediccionTrack.resultado_real == None)
    ).all()

    e1_lo, e2_lo = partido.equipo1.lower(), partido.equipo2.lower()
    resueltas = 0
    for t in tracks:
        t1_lo, t2_lo = t.equipo1.lower(), t.equipo2.lower()
        # Coinciden si los dos equipos son los mismos (en cualquier orden)
        mismo_orden    = (t1_lo == e1_lo and t2_lo == e2_lo)
        orden_invertido = (t1_lo == e2_lo and t2_lo == e1_lo)
        if not (mismo_orden or orden_invertido):
            continue
        if mismo_orden:
            g1, g2 = partido.goles_e1, partido.goles_e2
        else:
            g1, g2 = partido.goles_e2, partido.goles_e1
        real = 'Win' if g1 > g2 else ('Loss' if g1 < g2 else 'Draw')
        t.resultado_real = real
        t.g1_real = g1
        t.g2_real = g2
        t.acierto = (t.pred_consenso == real)
        session.add(t)
        resueltas += 1
    if resueltas:
        session.commit()
    return resueltas


# ---------------------------------------------------------------------------
# /partidos — solo lectura + soft delete
# ---------------------------------------------------------------------------

@app.get("/partidos", response_model=list[Partido])
def listar_partidos(
    equipo1: Optional[str] = None,
    equipo2: Optional[str] = None,
    session: Session = Depends(get_session),
):
    partidos = session.exec(select(Partido).where(Partido.activo == True)).all()

    if equipo1 and equipo2:
        e1, e2 = equipo1.lower(), equipo2.lower()
        partidos = [
            p for p in partidos
            if (e1 in p.equipo1.lower() and e2 in p.equipo2.lower())
            or (e2 in p.equipo1.lower() and e1 in p.equipo2.lower())
        ]
    elif equipo1:
        e1 = equipo1.lower()
        partidos = [p for p in partidos if e1 in p.equipo1.lower() or e1 in p.equipo2.lower()]
    elif equipo2:
        e2 = equipo2.lower()
        partidos = [p for p in partidos if e2 in p.equipo1.lower() or e2 in p.equipo2.lower()]

    return partidos


@app.get("/partidos/{partido_id}", response_model=Partido)
def obtener_partido(partido_id: int, session: Session = Depends(get_session)):
    partido = session.get(Partido, partido_id)
    if not partido or not partido.activo:
        raise HTTPException(status_code=404, detail="Partido no encontrado")
    return partido


@app.put("/partidos/{partido_id}", response_model=Partido)
def actualizar_partido(partido_id: int, data: PartidoUpdate, session: Session = Depends(get_session)):
    partido = session.get(Partido, partido_id)
    if not partido or not partido.activo:
        raise HTTPException(status_code=404, detail="Partido no encontrado")
    if data.fase is not None:
        partido.fase = data.fase
    if data.goles_e1 is not None:
        partido.goles_e1 = data.goles_e1
    if data.goles_e2 is not None:
        partido.goles_e2 = data.goles_e2
    if data.fecha is not None:
        partido.fecha = data.fecha
    session.add(partido)
    session.commit()
    session.refresh(partido)
    # Si llegó el resultado real, resolver predicciones tracked
    if data.goles_e1 is not None or data.goles_e2 is not None:
        _resolver_predicciones_para_partido(partido, session)
    return partido


@app.delete("/partidos/{partido_id}", status_code=204)
def desactivar_partido(partido_id: int, session: Session = Depends(get_session)):
    """Soft delete: marca activo=False para mantener historial."""
    partido = session.get(Partido, partido_id)
    if not partido or not partido.activo:
        raise HTTPException(status_code=404, detail="Partido no encontrado")
    partido.activo = False
    session.add(partido)
    session.commit()


# ---------------------------------------------------------------------------
# /evaluaciones — CRUD + soft delete
# ---------------------------------------------------------------------------

@app.post("/evaluaciones", response_model=Evaluacion, status_code=201)
def crear_evaluacion(data: EvaluacionCreate, background_tasks: BackgroundTasks,
                     session: Session = Depends(get_session)):
    """Crea la evaluación y entrena el pipeline en segundo plano (no bloquea)."""
    if not Path(data.filepath).exists():
        raise HTTPException(status_code=422, detail=f"Archivo no encontrado: {data.filepath}")
    evaluacion = Evaluacion(filepath=data.filepath, resultados="[]", cv_resultados="[]")
    session.add(evaluacion)
    session.commit()
    session.refresh(evaluacion)
    background_tasks.add_task(_run_training_background, evaluacion.id, data.filepath)
    return evaluacion


@app.get("/evaluaciones", response_model=list[Evaluacion])
def listar_evaluaciones(session: Session = Depends(get_session)):
    return session.exec(select(Evaluacion).where(Evaluacion.activo == True)).all()


@app.get("/evaluaciones/{evaluacion_id}", response_model=Evaluacion)
def obtener_evaluacion(evaluacion_id: int, session: Session = Depends(get_session)):
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(status_code=404, detail="Evaluación no encontrada")
    return ev


@app.put("/evaluaciones/{evaluacion_id}", response_model=Evaluacion)
def actualizar_evaluacion(evaluacion_id: int, data: EvaluacionUpdate,
                          background_tasks: BackgroundTasks,
                          session: Session = Depends(get_session)):
    """Re-entrena en background con nuevo filepath."""
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(status_code=404, detail="Evaluación no encontrada")
    if not Path(data.filepath).exists():
        raise HTTPException(status_code=422, detail=f"Archivo no encontrado: {data.filepath}")
    ev.filepath = data.filepath
    session.add(ev)
    session.commit()
    session.refresh(ev)
    _resultados_pipeline.pop(evaluacion_id, None)
    background_tasks.add_task(_run_training_background, evaluacion_id, data.filepath)
    return ev


@app.delete("/evaluaciones/{evaluacion_id}", status_code=204)
def desactivar_evaluacion(evaluacion_id: int, session: Session = Depends(get_session)):
    """Soft delete: el historial de accuracy queda en la DB con activo=False."""
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(status_code=404, detail="Evaluación no encontrada")
    ev.activo = False
    session.add(ev)
    session.commit()
    _resultados_pipeline.pop(evaluacion_id, None)


# ---------------------------------------------------------------------------
# /predicciones — CRUD + soft delete
# ---------------------------------------------------------------------------

@app.post("/predicciones", response_model=Prediccion, status_code=201)
def crear_prediccion(data: PrediccionCreate, session: Session = Depends(get_session)):
    """Predice un partido usando el modelo de la evaluacion indicada."""
    ev = session.get(Evaluacion, data.evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(status_code=404, detail=f"Evaluación {data.evaluacion_id} no encontrada")

    results = _get_or_run_pipeline(data.evaluacion_id, session)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        predecir_partido(data.equipo1, data.equipo2, results, n_runs=data.n_runs)

    prediccion = Prediccion(
        equipo1=data.equipo1,
        equipo2=data.equipo2,
        evaluacion_id=data.evaluacion_id,
        n_runs=data.n_runs,
        output=buf.getvalue(),
    )
    session.add(prediccion)
    session.commit()
    session.refresh(prediccion)
    return prediccion


@app.get("/predicciones", response_model=list[Prediccion])
def listar_predicciones(session: Session = Depends(get_session)):
    return session.exec(select(Prediccion).where(Prediccion.activo == True)).all()


@app.get("/predicciones/{prediccion_id}", response_model=Prediccion)
def obtener_prediccion(prediccion_id: int, session: Session = Depends(get_session)):
    pred = session.get(Prediccion, prediccion_id)
    if not pred or not pred.activo:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    return pred


@app.put("/predicciones/{prediccion_id}", response_model=Prediccion)
def actualizar_prediccion(prediccion_id: int, data: PrediccionUpdate, session: Session = Depends(get_session)):
    """Re-ejecuta predecir_partido con los nuevos parámetros y actualiza el resultado."""
    pred = session.get(Prediccion, prediccion_id)
    if not pred or not pred.activo:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    if pred.evaluacion_id not in _resultados_pipeline:
        raise HTTPException(
            status_code=409,
            detail="El pipeline no está en memoria — vuelve a ejecutar POST /evaluaciones",
        )
    if data.equipo1 is not None:
        pred.equipo1 = data.equipo1
    if data.equipo2 is not None:
        pred.equipo2 = data.equipo2
    if data.n_runs is not None:
        pred.n_runs = data.n_runs
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        predecir_partido(pred.equipo1, pred.equipo2, _resultados_pipeline[pred.evaluacion_id], n_runs=pred.n_runs)
    pred.output = buf.getvalue()
    session.add(pred)
    session.commit()
    session.refresh(pred)
    return pred


@app.delete("/predicciones/{prediccion_id}", status_code=204)
def desactivar_prediccion(prediccion_id: int, session: Session = Depends(get_session)):
    """Soft delete: la predicción queda en la DB con activo=False."""
    pred = session.get(Prediccion, prediccion_id)
    if not pred or not pred.activo:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    pred.activo = False
    session.add(pred)
    session.commit()


# ---------------------------------------------------------------------------
# Endpoints "dashboard" — devuelven JSON estructurado para el frontend
# ---------------------------------------------------------------------------

@app.get("/api/equipos")
def listar_equipos(session: Session = Depends(get_session)):
    """Lista única de equipos del dataset (para los dropdowns del predictor)."""
    partidos = session.exec(select(Partido).where(Partido.activo == True)).all()
    equipos = sorted({p.equipo1 for p in partidos} | {p.equipo2 for p in partidos})
    return {"equipos": equipos, "total": len(equipos)}


@app.get("/api/evaluaciones/{evaluacion_id}/elos")
def ranking_elo(evaluacion_id: int, session: Session = Depends(get_session)):
    """Ranking ELO de todos los equipos según la evaluación indicada."""
    results = _get_or_run_pipeline(evaluacion_id, session)
    team_elos = results.get("team_elos", {})
    ranking = sorted(
        ({"equipo": t, "elo": round(float(e), 1)} for t, e in team_elos.items()),
        key=lambda x: -x["elo"],
    )
    return {"ranking": ranking, "total": len(ranking)}


@app.get("/api/evaluaciones/{evaluacion_id}/metricas")
def metricas_evaluacion(evaluacion_id: int, session: Session = Depends(get_session)):
    """Resultados de modelos: CV (walk-forward) + test cronológico."""
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(status_code=404, detail="Evaluación no encontrada")
    return {
        "evaluacion_id": evaluacion_id,
        "test": json.loads(ev.resultados),       # accuracy + F1 macro en test único
        "cv": json.loads(ev.cv_resultados),       # CV walk-forward
    }


@app.get("/api/evaluaciones/{evaluacion_id}/feature-importance")
def feature_importance(evaluacion_id: int, top: int = 15, session: Session = Depends(get_session)):
    """Top features del Random Forest (las que el modelo más usa)."""
    results = _get_or_run_pipeline(evaluacion_id, session)
    fi = results.get("feature_importance", []) or []
    return {"top_features": fi[:top], "total": len(fi)}


@app.get("/api/evaluaciones/{evaluacion_id}/predicciones-test")
def predicciones_test(evaluacion_id: int, session: Session = Depends(get_session)):
    """Predicciones del split de test (20% más reciente) vs resultado real."""
    results = _get_or_run_pipeline(evaluacion_id, session)
    pred_df = results.get("predictions")
    if pred_df is None:
        raise HTTPException(status_code=500, detail="No hay predicciones de test")
    return {"predicciones": json.loads(pred_df.to_json(orient="records"))}


class PrediccionRapida(SQLModel):
    equipo1: str
    equipo2: str
    evaluacion_id: int
    n_runs: int = 20
    fase: str = "Liga"


@app.post("/api/predecir")
def predecir_rapido(data: PrediccionRapida, session: Session = Depends(get_session)):
    """
    Predicción estructurada en JSON (no guarda en DB).
    Devuelve probabilidades por modelo, consenso, marcadores y ELO.
    """
    results = _get_or_run_pipeline(data.evaluacion_id, session)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        salida = predecir_partido(
            data.equipo1, data.equipo2, results,
            n_runs=data.n_runs, fase=data.fase,
        )
    if salida is None:
        raise HTTPException(status_code=500, detail="predecir_partido() no devolvió datos")
    salida["log"] = buf.getvalue()
    return salida


@app.get("/api/evaluaciones/{evaluacion_id}/status")
def estado_entrenamiento(evaluacion_id: int):
    """Polling: devuelve 'training' | 'ready' | 'error:...' | 'unknown'."""
    if evaluacion_id in _resultados_pipeline:
        return {"status": "ready"}
    s = _training_status.get(evaluacion_id, "unknown")
    return {"status": s}


@app.get("/api/health")
def health():
    return {"status": "ok", "evaluaciones_en_memoria": list(_resultados_pipeline.keys())}


@app.post("/api/sync")
def sync_dataset(session: Session = Depends(get_session)):
    """Sincroniza la tabla Partido con el Excel: agrega los partidos nuevos.
    Llamar después de agregar partidos con agregar_partido.py para reflejar
    los cambios en el dashboard sin reiniciar el servidor."""
    nuevos = _cargar_excel(session)
    total = session.exec(select(func.count(Partido.id)).where(Partido.activo == True)).one()
    return {"nuevos_agregados": nuevos, "total_partidos": total}


# ---------------------------------------------------------------------------
# Track record — predicciones públicas vs resultados reales (monetización)
# ---------------------------------------------------------------------------

class PrediccionTrackCreate(SQLModel):
    equipo1: str
    equipo2: str
    fecha_partido: Optional[str] = None
    evaluacion_id: int
    n_runs: int = 20
    fase: str = "Liga"


@app.post("/api/track", response_model=PrediccionTrack, status_code=201)
def crear_prediccion_track(data: PrediccionTrackCreate, session: Session = Depends(get_session)):
    """Predice un partido futuro y lo guarda para tracking. Si el partido ya
    está en la DB con resultado, se autoresuelve inmediatamente."""
    from datetime import date
    results = _get_or_run_pipeline(data.evaluacion_id, session)
    salida = predecir_partido(
        data.equipo1, data.equipo2, results,
        n_runs=data.n_runs, fase=data.fase,
    )
    if salida is None:
        raise HTTPException(status_code=500, detail="predecir_partido() no devolvió datos")

    track = PrediccionTrack(
        equipo1=data.equipo1,
        equipo2=data.equipo2,
        fecha_partido=data.fecha_partido,
        fecha_prediccion=str(date.today()),
        evaluacion_id=data.evaluacion_id,
        pred_consenso=salida["consenso"]["pred"],
        prob_win=salida["consenso"]["win"],
        prob_draw=salida["consenso"]["draw"],
        prob_loss=salida["consenso"]["loss"],
        elo_e1=salida["elo_e1"],
        elo_e2=salida["elo_e2"],
        fase=data.fase,
    )
    session.add(track)
    session.commit()
    session.refresh(track)

    # Si el partido ya existe en /partidos con resultado, autorresolverlo
    partidos = session.exec(select(Partido).where(Partido.activo == True)).all()
    e1_lo, e2_lo = data.equipo1.lower(), data.equipo2.lower()
    for p in partidos:
        p1_lo, p2_lo = p.equipo1.lower(), p.equipo2.lower()
        if {p1_lo, p2_lo} == {e1_lo, e2_lo} and (p.goles_e1 or p.goles_e2):
            _resolver_predicciones_para_partido(p, session)
            session.refresh(track)
            break

    return track


@app.get("/api/track", response_model=list[PrediccionTrack])
def listar_track(session: Session = Depends(get_session)):
    return session.exec(
        select(PrediccionTrack).where(PrediccionTrack.activo == True)
    ).all()


@app.delete("/api/track/{track_id}", status_code=204)
def desactivar_track(track_id: int, session: Session = Depends(get_session)):
    t = session.get(PrediccionTrack, track_id)
    if not t or not t.activo:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    t.activo = False
    session.add(t)
    session.commit()


@app.get("/api/track/stats")
def stats_track_record(session: Session = Depends(get_session)):
    """Estadísticas agregadas del track record público (lo que verá la audiencia)."""
    tracks = session.exec(
        select(PrediccionTrack).where(PrediccionTrack.activo == True)
    ).all()
    resueltas = [t for t in tracks if t.resultado_real is not None]
    pendientes = [t for t in tracks if t.resultado_real is None]
    aciertos = [t for t in resueltas if t.acierto]
    total = len(resueltas)
    accuracy = (len(aciertos) / total) if total else 0.0

    # Accuracy por clase predicha
    por_clase = {}
    for cls in ('Win', 'Draw', 'Loss'):
        sub = [t for t in resueltas if t.pred_consenso == cls]
        ok = [t for t in sub if t.acierto]
        por_clase[cls] = {
            'total': len(sub),
            'aciertos': len(ok),
            'accuracy': round((len(ok) / len(sub)) if sub else 0.0, 4),
        }

    return {
        'total_predicciones': len(tracks),
        'resueltas': total,
        'pendientes': len(pendientes),
        'aciertos': len(aciertos),
        'accuracy_global': round(accuracy, 4),
        'por_clase_predicha': por_clase,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
