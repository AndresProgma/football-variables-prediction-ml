# UCL Predictor

Pipeline de predicción para partidos de Champions League con dashboard web y API REST.

Scrapea stats de uefa.com, calcula features pre-partido (ELO incremental, forma últimos 5, head-to-head, días de descanso), entrena 6 clasificadores + 2 regresores con validación walk-forward, y expone todo en un dashboard responsive + API.

## Highlights

- **6 clasificadores** (Random Forest, Gradient Boosting, Logistic Regression, SVM, XGBoost, KNN) con consenso por ensemble
- **2 regresores** (RF, XGBoost) para marcador estimado
- **Features avanzadas pre-partido**: ELO Elo-style, forma últimos 5 partidos, H2H, descanso, todo sin leakage
- **Validación temporal** TimeSeriesSplit walk-forward, F1 macro como métrica principal
- **Track record público auto-resuelto**: guardas una predicción y se compara automáticamente con el resultado real cuando llega
- **Dashboard web** con ranking ELO, métricas de modelo, predictor interactivo y track record histórico
- **Listo para hostear** con Docker / Render Blueprint

## Arquitectura

```
api.py                          FastAPI (API + dashboard servidor)
knime_workflow_converter.py     Pipeline ML: features, modelos, predicción
scraper_uefa.py                 Scraper Playwright de uefa.com
agregar_partido.py              CLI para agregar partidos al dataset

static/                         Dashboard web (HTML + Tailwind + Chart.js CDN)
  ├── index.html
  ├── js/api.js                 Cliente HTTP centralizado
  ├── js/app.js                 Lógica del dashboard
  └── css/styles.css

creando_dataset_modificado.xlsx Dataset principal
futbol.db                       SQLite (tracks de predicciones)

Dockerfile · render.yaml · .env.example   Deploy en producción
```

## Setup local

```bash
pip install -r requirements.txt
python -m playwright install chromium     # solo si vas a scrapear
uvicorn api:app --reload
```

Abre [http://localhost:8000](http://localhost:8000) para el dashboard.
Documentación interactiva del API en [http://localhost:8000/docs](http://localhost:8000/docs).

## Scrapear partidos nuevos

```bash
# Toda una jornada por fecha:
python agregar_partido.py --fecha 2026-03-10 --fase Octavos --si

# Por URL específica:
python agregar_partido.py --url https://es.uefa.com/uefachampionsleague/match/.../

# Interactivo (manual):
python agregar_partido.py
```

## Endpoints principales

| Método | Endpoint | Descripción |
|---|---|---|
| GET  | `/` | Dashboard web |
| GET  | `/api/health` | Health check |
| GET  | `/api/equipos` | Lista de equipos del dataset |
| POST | `/evaluaciones` | Reentrenar el pipeline completo |
| GET  | `/api/evaluaciones/{id}/elos` | Ranking ELO |
| GET  | `/api/evaluaciones/{id}/metricas` | Resultados CV + test |
| POST | `/api/predecir` | Predicción rápida (JSON, no guarda) |
| POST | `/api/track` | Predecir + guardar al track record público |
| GET  | `/api/track` · `/api/track/stats` | Listar / stats agregadas del track |
| GET  | `/docs` | Documentación interactiva (Swagger) |

## Deploy a producción

### Render.com (recomendado, gratis)

1. Sube el repo a GitHub.
2. En Render: **New → Blueprint** → conecta el repo.
3. Render detecta `render.yaml` y crea el servicio automáticamente.
4. Para que la SQLite persista entre deploys, descomenta el `disk:` en `render.yaml`.

### Railway / Fly.io / cualquier Docker host

```bash
docker build -t ucl-predictor .
docker run -p 8000:8000 -e ALLOWED_ORIGINS="*" ucl-predictor
```

### Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `PORT` | `8000` | Puerto del servidor |
| `ALLOWED_ORIGINS` | `*` | CORS — lista separada por comas |
| `DATABASE_URL` | `sqlite:///./futbol.db` | Conexión a DB (soporta Postgres) |
| `DATASET_PATH` | `./creando_dataset_modificado.xlsx` | Ruta al Excel |

## Features pre-partido del modelo

- **ELO incremental** (K=30, ventaja local=60 pts, bonus por margen de goles)
- **Forma últimos 5**: W/D/L, goles a favor/contra, puntos acumulados
- **Días de descanso** desde el último partido
- **Head-to-head**: últimos 3 enfrentamientos directos
- **Diferenciales**: Δ ELO, Δ forma, Δ goal-diff
- **+100 stats** agregadas por equipo desde el dataset histórico
- `SelectKBest(f_classif, k=25)` dentro de cada pipeline → recorta el ruido sin leakage

Todo se calcula cronológicamente con `TimeSeriesSplit` walk-forward.

## Stack

- **Backend**: Python 3.12 · FastAPI · SQLModel · SQLite (default) / Postgres
- **ML**: scikit-learn · XGBoost · pandas
- **Scraping**: Playwright
- **Frontend**: HTML + Tailwind CDN + Chart.js CDN (sin build step)
- **Deploy**: Docker / Render Blueprint
