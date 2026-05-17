"""
API REST — Predicción de Fútbol, Champions League.
Microservicio API: CRUD + orquestación.
Llama a ml-service via HTTP en vez de importar el pipeline directamente.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Field, Session, SQLModel, create_engine, func, select

DATASET = os.getenv("DATASET_PATH", "/data/creando_dataset_modificado.xlsx")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////data/futbol.db")
ML_URL = os.getenv("ML_SERVICE_URL", "http://ml:8001")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
FEATURED_PICK_FILE = DATA_DIR / "featured_pick.json"
RECORD_HISTORICO_FILE = DATA_DIR / "record_historico.json"

engine = create_engine(DATABASE_URL)


# ---------------------------------------------------------------------------
# ORM Models
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
    resultados: str
    cv_resultados: str
    activo: bool = True


class Prediccion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    equipo1: str
    equipo2: str
    evaluacion_id: int = Field(foreign_key="evaluacion.id")
    n_runs: int
    output: str
    activo: bool = True


class PrediccionTrack(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    equipo1: str
    equipo2: str
    fecha_partido: Optional[str] = None
    fecha_prediccion: str
    evaluacion_id: int = Field(foreign_key="evaluacion.id")
    pred_consenso: str
    prob_win: float
    prob_draw: float
    prob_loss: float
    elo_e1: float
    elo_e2: float
    fase: str = "Liga"
    resultado_real: Optional[str] = None
    g1_real: Optional[int] = None
    g2_real: Optional[int] = None
    acierto: Optional[bool] = None
    activo: bool = True


# ---------------------------------------------------------------------------
# Input schemas
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


class PrediccionRapida(SQLModel):
    equipo1: str
    equipo2: str
    evaluacion_id: int
    n_runs: int = 20
    fase: str = "Liga"


class PrediccionTrackCreate(SQLModel):
    equipo1: str
    equipo2: str
    fecha_partido: Optional[str] = None
    evaluacion_id: int
    n_runs: int = 20
    fase: str = "Liga"


class ResolverTrackBody(SQLModel):
    resultado_real: str
    g1_real: int = 0
    g2_real: int = 0


class ValorBet(SQLModel):
    nombre: str
    descripcion: str = ""
    porcentaje: int


class FeaturedPickBody(SQLModel):
    equipo1: str
    equipo2: str
    hora: str = "21:00"
    fase: str = "UCL"
    prob_win: float
    prob_draw: float
    prob_loss: float
    valores: list[ValorBet] = []
    mercados: Optional[dict] = None
    modelos: Optional[list] = None
    goles: Optional[list] = None
    ultimos_e1: Optional[list] = None
    ultimos_e2: Optional[list] = None


# ---------------------------------------------------------------------------
# ML service helper
# ---------------------------------------------------------------------------

def _ml(method: str, path: str, timeout: float = 600.0, **kwargs) -> dict:
    with httpx.Client(timeout=timeout) as client:
        resp = getattr(client, method)(f"{ML_URL}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _cargar_excel(session: Session) -> int:
    try:
        df = pd.read_excel(DATASET)
    except FileNotFoundError:
        return 0

    ids_en_db = set(session.exec(select(Partido.id).where(Partido.activo == True)).all())
    nuevos = 0
    for _, row in df.iterrows():
        pid = int(row["Partido_id"]) if pd.notna(row.get("Partido_id")) else None
        if pid is not None and pid in ids_en_db:
            continue
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        _cargar_excel(session)
    yield


app = FastAPI(title="API — Predicción Fútbol UCL", lifespan=lifespan)

_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard_root():
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/admin", include_in_schema=False)
    def admin_page():
        return FileResponse(_STATIC_DIR / "admin.html")


def get_session():
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Background: orquesta entrenamiento en ml-service y actualiza DB
# ---------------------------------------------------------------------------

def _run_pipeline_background(evaluacion_id: int, filepath: str) -> None:
    # 1. Trigger training (non-blocking en ml-service)
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(f"{ML_URL}/train/{evaluacion_id}", json={"filepath": filepath})
    except Exception:
        return

    # 2. Poll hasta que ml-service termine (max 10 min)
    for _ in range(120):
        time.sleep(5)
        try:
            with httpx.Client(timeout=10.0) as client:
                s = client.get(f"{ML_URL}/status/{evaluacion_id}").json()["status"]
        except Exception:
            continue
        if s == "ready":
            break
        if s.startswith("error"):
            return

    # 3. Guardar métricas en DB
    try:
        metricas = _ml("get", f"/metricas/{evaluacion_id}")
        with Session(engine) as sess:
            ev = sess.get(Evaluacion, evaluacion_id)
            if ev:
                ev.resultados = json.dumps(metricas["test"])
                ev.cv_resultados = json.dumps(metricas["cv"])
                sess.add(ev)
                sess.commit()
    except Exception:
        pass

    # 4. Generar record histórico en ml-service
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(f"{ML_URL}/record-historico")
    except Exception:
        pass


def _resolver_predicciones_para_partido(partido: Partido, session: Session) -> int:
    tracks = session.exec(
        select(PrediccionTrack)
        .where(PrediccionTrack.activo == True)
        .where(PrediccionTrack.resultado_real == None)
    ).all()

    e1_lo, e2_lo = partido.equipo1.lower(), partido.equipo2.lower()
    resueltas = 0
    for t in tracks:
        t1_lo, t2_lo = t.equipo1.lower(), t.equipo2.lower()
        mismo_orden = (t1_lo == e1_lo and t2_lo == e2_lo)
        orden_invertido = (t1_lo == e2_lo and t2_lo == e1_lo)
        if not (mismo_orden or orden_invertido):
            continue
        g1, g2 = (partido.goles_e1, partido.goles_e2) if mismo_orden else (partido.goles_e2, partido.goles_e1)
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
# /partidos
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
        raise HTTPException(404, "Partido no encontrado")
    return partido


@app.put("/partidos/{partido_id}", response_model=Partido)
def actualizar_partido(partido_id: int, data: PartidoUpdate, session: Session = Depends(get_session)):
    partido = session.get(Partido, partido_id)
    if not partido or not partido.activo:
        raise HTTPException(404, "Partido no encontrado")
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
    if data.goles_e1 is not None or data.goles_e2 is not None:
        _resolver_predicciones_para_partido(partido, session)
    return partido


@app.delete("/partidos/{partido_id}", status_code=204)
def desactivar_partido(partido_id: int, session: Session = Depends(get_session)):
    partido = session.get(Partido, partido_id)
    if not partido or not partido.activo:
        raise HTTPException(404, "Partido no encontrado")
    partido.activo = False
    session.add(partido)
    session.commit()


# ---------------------------------------------------------------------------
# /evaluaciones
# ---------------------------------------------------------------------------

@app.post("/evaluaciones", response_model=Evaluacion, status_code=201)
def crear_evaluacion(data: EvaluacionCreate, background_tasks: BackgroundTasks,
                     session: Session = Depends(get_session)):
    if not Path(data.filepath).exists():
        raise HTTPException(422, f"Archivo no encontrado: {data.filepath}")
    evaluacion = Evaluacion(filepath=data.filepath, resultados="[]", cv_resultados="[]")
    session.add(evaluacion)
    session.commit()
    session.refresh(evaluacion)
    background_tasks.add_task(_run_pipeline_background, evaluacion.id, data.filepath)
    return evaluacion


@app.get("/evaluaciones", response_model=list[Evaluacion])
def listar_evaluaciones(session: Session = Depends(get_session)):
    return session.exec(select(Evaluacion).where(Evaluacion.activo == True)).all()


@app.get("/evaluaciones/{evaluacion_id}", response_model=Evaluacion)
def obtener_evaluacion(evaluacion_id: int, session: Session = Depends(get_session)):
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(404, "Evaluación no encontrada")
    return ev


@app.put("/evaluaciones/{evaluacion_id}", response_model=Evaluacion)
def actualizar_evaluacion(evaluacion_id: int, data: EvaluacionUpdate,
                          session: Session = Depends(get_session)):
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(404, "Evaluación no encontrada")
    try:
        metricas = _ml("post", f"/train/{evaluacion_id}", json={"filepath": data.filepath})
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)
    ev.filepath = data.filepath
    session.add(ev)
    session.commit()
    session.refresh(ev)
    return ev


@app.delete("/evaluaciones/{evaluacion_id}", status_code=204)
def desactivar_evaluacion(evaluacion_id: int, session: Session = Depends(get_session)):
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(404, "Evaluación no encontrada")
    ev.activo = False
    session.add(ev)
    session.commit()


# ---------------------------------------------------------------------------
# /predicciones
# ---------------------------------------------------------------------------

@app.post("/predicciones", response_model=Prediccion, status_code=201)
def crear_prediccion(data: PrediccionCreate, session: Session = Depends(get_session)):
    ev = session.get(Evaluacion, data.evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(404, f"Evaluación {data.evaluacion_id} no encontrada")
    try:
        salida = _ml("post", "/predict", json={
            "equipo1": data.equipo1,
            "equipo2": data.equipo2,
            "eval_id": data.evaluacion_id,
            "n_runs": data.n_runs,
            "fase": "Liga",
        })
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)

    prediccion = Prediccion(
        equipo1=data.equipo1,
        equipo2=data.equipo2,
        evaluacion_id=data.evaluacion_id,
        n_runs=data.n_runs,
        output=json.dumps(salida),
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
        raise HTTPException(404, "Predicción no encontrada")
    return pred


@app.delete("/predicciones/{prediccion_id}", status_code=204)
def desactivar_prediccion(prediccion_id: int, session: Session = Depends(get_session)):
    pred = session.get(Prediccion, prediccion_id)
    if not pred or not pred.activo:
        raise HTTPException(404, "Predicción no encontrada")
    pred.activo = False
    session.add(pred)
    session.commit()


# ---------------------------------------------------------------------------
# Dashboard endpoints — proxian a ml-service
# ---------------------------------------------------------------------------

@app.get("/api/evaluaciones/{evaluacion_id}/status")
def estado_entrenamiento(evaluacion_id: int):
    try:
        return _ml("get", f"/status/{evaluacion_id}", timeout=10.0)
    except Exception:
        return {"id": evaluacion_id, "status": "unknown"}


@app.get("/api/equipos")
def listar_equipos(session: Session = Depends(get_session)):
    partidos = session.exec(select(Partido).where(Partido.activo == True)).all()
    equipos = sorted({p.equipo1 for p in partidos} | {p.equipo2 for p in partidos})
    return {"equipos": equipos, "total": len(equipos)}


@app.get("/api/evaluaciones/{evaluacion_id}/elos")
def ranking_elo(evaluacion_id: int, session: Session = Depends(get_session)):
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(404, "Evaluación no encontrada")
    try:
        return _ml("get", f"/elos/{evaluacion_id}")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)


@app.get("/api/evaluaciones/{evaluacion_id}/metricas")
def metricas_evaluacion(evaluacion_id: int, session: Session = Depends(get_session)):
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(404, "Evaluación no encontrada")
    try:
        data = _ml("get", f"/metricas/{evaluacion_id}")
        return {"evaluacion_id": evaluacion_id, "test": data["test"], "cv": data["cv"]}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)


@app.get("/api/evaluaciones/{evaluacion_id}/feature-importance")
def feature_importance(evaluacion_id: int, top: int = 15, session: Session = Depends(get_session)):
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(404, "Evaluación no encontrada")
    try:
        return _ml("get", f"/feature-importance/{evaluacion_id}", params={"top": top})
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)


@app.get("/api/evaluaciones/{evaluacion_id}/predicciones-test")
def predicciones_test(evaluacion_id: int, session: Session = Depends(get_session)):
    if RECORD_HISTORICO_FILE.exists():
        data = json.loads(RECORD_HISTORICO_FILE.read_text(encoding="utf-8"))
        return {"predicciones": data, "tipo": "v2_honesto"}
    try:
        return _ml("get", f"/predicciones-test/{evaluacion_id}")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)


@app.post("/api/predecir")
def predecir_rapido(data: PrediccionRapida, session: Session = Depends(get_session)):
    ev = session.get(Evaluacion, data.evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(404, "Evaluación no encontrada")
    try:
        return _ml("post", "/predict", json={
            "equipo1": data.equipo1,
            "equipo2": data.equipo2,
            "eval_id": data.evaluacion_id,
            "n_runs": data.n_runs,
            "fase": data.fase,
        })
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)


@app.get("/api/health")
def health(session: Session = Depends(get_session)):
    evaluaciones = session.exec(select(Evaluacion).where(Evaluacion.activo == True)).all()
    return {"status": "ok", "evaluaciones": len(evaluaciones)}


@app.post("/api/sync")
def sync_dataset(session: Session = Depends(get_session)):
    nuevos = _cargar_excel(session)
    total = session.exec(select(func.count(Partido.id)).where(Partido.activo == True)).one()
    return {"nuevos_agregados": nuevos, "total_partidos": total}


# ---------------------------------------------------------------------------
# Track record
# ---------------------------------------------------------------------------

@app.post("/api/track", response_model=PrediccionTrack, status_code=201)
def crear_prediccion_track(data: PrediccionTrackCreate, session: Session = Depends(get_session)):
    ev = session.get(Evaluacion, data.evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(404, "Evaluación no encontrada")
    try:
        salida = _ml("post", "/predict", json={
            "equipo1": data.equipo1,
            "equipo2": data.equipo2,
            "eval_id": data.evaluacion_id,
            "n_runs": data.n_runs,
            "fase": data.fase,
        })
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)

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

    # Auto-resolver si el partido ya tiene resultado
    e1_lo, e2_lo = data.equipo1.lower(), data.equipo2.lower()
    for p in session.exec(select(Partido).where(Partido.activo == True)).all():
        if {p.equipo1.lower(), p.equipo2.lower()} == {e1_lo, e2_lo} and (p.goles_e1 or p.goles_e2):
            _resolver_predicciones_para_partido(p, session)
            session.refresh(track)
            break

    return track


@app.get("/api/track", response_model=list[PrediccionTrack])
def listar_track(session: Session = Depends(get_session)):
    return session.exec(select(PrediccionTrack).where(PrediccionTrack.activo == True)).all()


@app.delete("/api/track/{track_id}", status_code=204)
def desactivar_track(track_id: int, session: Session = Depends(get_session)):
    t = session.get(PrediccionTrack, track_id)
    if not t or not t.activo:
        raise HTTPException(404, "Predicción no encontrada")
    t.activo = False
    session.add(t)
    session.commit()


@app.post("/api/track/{track_id}/resolver", response_model=PrediccionTrack)
def resolver_manual_track(track_id: int, data: ResolverTrackBody,
                           session: Session = Depends(get_session)):
    t = session.get(PrediccionTrack, track_id)
    if not t or not t.activo:
        raise HTTPException(404, "Predicción no encontrada")
    t.resultado_real = data.resultado_real
    t.g1_real = data.g1_real
    t.g2_real = data.g2_real
    t.acierto = (data.resultado_real == t.pred_consenso)
    session.add(t)
    session.commit()
    session.refresh(t)
    return t


@app.get("/api/track/stats")
def stats_track_record(session: Session = Depends(get_session)):
    tracks = session.exec(select(PrediccionTrack).where(PrediccionTrack.activo == True)).all()
    resueltas = [t for t in tracks if t.resultado_real is not None]
    pendientes = [t for t in tracks if t.resultado_real is None]
    aciertos = [t for t in resueltas if t.acierto]
    total = len(resueltas)
    accuracy = (len(aciertos) / total) if total else 0.0

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
# Featured Pick del Día
# ---------------------------------------------------------------------------

@app.get("/api/featured-pick")
def get_featured_pick():
    if FEATURED_PICK_FILE.exists():
        return json.loads(FEATURED_PICK_FILE.read_text(encoding="utf-8"))
    return {}


@app.post("/api/admin/featured-pick")
def set_featured_pick(data: FeaturedPickBody):
    FEATURED_PICK_FILE.write_text(
        json.dumps(data.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data


@app.delete("/api/admin/featured-pick", status_code=204)
def delete_featured_pick():
    if FEATURED_PICK_FILE.exists():
        FEATURED_PICK_FILE.unlink()


# ---------------------------------------------------------------------------
# Record histórico
# ---------------------------------------------------------------------------

@app.get("/api/record")
def record_publico(session: Session = Depends(get_session)):
    if RECORD_HISTORICO_FILE.exists():
        data = json.loads(RECORD_HISTORICO_FILE.read_text(encoding="utf-8"))
        return {"tipo": "v2_honesto", "predicciones": data}
    evals = session.exec(select(Evaluacion).where(Evaluacion.activo == True)).all()
    if not evals:
        return {"tipo": "sin_datos", "predicciones": []}
    ev = max(evals, key=lambda e: e.id)
    try:
        return _ml("get", f"/predicciones-test/{ev.id}")
    except Exception:
        return {"tipo": "sin_datos", "predicciones": []}


@app.get("/api/admin/record-status")
def record_status():
    return {"status": "ok" if RECORD_HISTORICO_FILE.exists() else "pending"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
