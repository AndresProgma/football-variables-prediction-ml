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
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from sqlmodel import Field, Session, SQLModel, create_engine, select

from knime_workflow_converter import main as run_pipeline, predecir_partido

DATASET = r"C:\Users\fehgb\OneDrive\Desktop\prediccion futybol\creando_dataset_modificado.xlsx"
DATABASE_URL = "sqlite:///./futbol.db"

engine = create_engine(DATABASE_URL)

# Pipeline results guardados en memoria (demasiado pesados para DB)
_resultados_pipeline: dict[int, dict] = {}


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


# ---------------------------------------------------------------------------
# Schemas de entrada (no son tablas)
# ---------------------------------------------------------------------------

class EvaluacionCreate(SQLModel):
    filepath: str = DATASET


class PrediccionCreate(SQLModel):
    equipo1: str
    equipo2: str
    evaluacion_id: int
    n_runs: int = 20


# ---------------------------------------------------------------------------
# Startup: crear tablas y cargar Excel
# ---------------------------------------------------------------------------

def _cargar_excel(session: Session) -> None:
    if session.exec(select(Partido)).first():
        return  # Ya cargado en sesiones anteriores
    try:
        df = pd.read_excel(DATASET)
    except FileNotFoundError:
        return
    for _, row in df.iterrows():
        goles_e1 = row.get("EQUIPO1_GOLES")
        goles_e2 = row.get("EQUIPO2_GOLES")
        partido = Partido(
            id=int(row["Partido_id"]) if pd.notna(row.get("Partido_id")) else None,
            equipo1=str(row.get("Equipo1", "")),
            equipo2=str(row.get("Equipo2", "")),
            fase=str(row.get("Fase", "")),
            goles_e1=int(goles_e1) if pd.notna(goles_e1) else 0,
            goles_e2=int(goles_e2) if pd.notna(goles_e2) else 0,
            fecha=str(row["Fecha"]) if pd.notna(row.get("Fecha")) else None,
        )
        session.add(partido)
    session.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        _cargar_excel(session)
    yield


app = FastAPI(title="Predicción Fútbol UCL", lifespan=lifespan)


def get_session():
    with Session(engine) as session:
        yield session


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
def crear_evaluacion(data: EvaluacionCreate, session: Session = Depends(get_session)):
    """Ejecuta el pipeline completo y guarda los resultados de accuracy."""
    try:
        results = run_pipeline(data.filepath)
    except FileNotFoundError:
        raise HTTPException(status_code=422, detail=f"Archivo no encontrado: {data.filepath}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    evaluacion = Evaluacion(
        filepath=data.filepath,
        resultados=results["results"].to_json(orient="records"),
        cv_resultados=results["cv_results"].to_json(orient="records"),
    )
    session.add(evaluacion)
    session.commit()
    session.refresh(evaluacion)

    _resultados_pipeline[evaluacion.id] = results
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

    if data.evaluacion_id not in _resultados_pipeline:
        raise HTTPException(
            status_code=409,
            detail="El pipeline no está en memoria — vuelve a ejecutar POST /evaluaciones",
        )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        predecir_partido(data.equipo1, data.equipo2, _resultados_pipeline[data.evaluacion_id], n_runs=data.n_runs)

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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
