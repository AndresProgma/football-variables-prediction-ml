# UCL Predictor

Predicción de partidos de Champions League con dashboard web y API REST. Vivo en <https://futbolazoia.live>.

Scrapea estadísticas de uefa.com con Playwright, calcula features pre-partido (ELO incremental, forma últimos 5 partidos, head-to-head, días de descanso), entrena 6 clasificadores + 2 regresores con validación temporal walk-forward, y expone todo en un dashboard responsive + API.

## Highlights

- **6 clasificadores** (Random Forest, Gradient Boosting, Logistic Regression, SVM, XGBoost, KNN) con consenso por ensemble de 20 corridas con seeds distintos
- **2 regresores** (RF, XGBoost) para marcador estimado con desvío estándar
- **Features pre-partido sin leakage**: ELO Elo-style, forma últimos 5 partidos, H2H, días de descanso
- **Validación temporal honesta**: `TimeSeriesSplit` walk-forward + slice cronológico (no random shuffle)
- **Track record público auto-resuelto**: guardás una predicción y se compara automáticamente con el resultado real cuando el partido se juega
- **Dashboard web** con ranking ELO, métricas por modelo, predictor interactivo y track record histórico
- **API REST** con SQLModel + SQLite, soft delete, filtros por equipo
- **Deploy listo**: Docker + Render Blueprint, dominio custom funcionando

## Estructura del código

```
prediccion-futybol/
├─ api.py                            FastAPI: API REST + sirve el dashboard
├─ knime_workflow_converter.py       Pipeline ML completo (legacy name)
├─ scraper_uefa.py                   Scraper Playwright de uefa.com
├─ agregar_partido.py                CLI para agregar partidos (manual/--url/--fecha)
│
├─ static/                           Dashboard web (sin build step)
│  ├─ index.html                     Markup + Tailwind CDN
│  ├─ js/api.js                      Cliente HTTP centralizado
│  ├─ js/app.js                      Lógica del dashboard, render de tablas y gráficos
│  └─ css/styles.css
│
├─ creando_dataset_modificado.xlsx   Dataset principal (81 partidos × 140 cols)
├─ futbol.db                         SQLite local (se crea al arrancar)
│
├─ Dockerfile · render.yaml          Deploy
├─ requirements.txt · .env.example
└─ ideas.txt                         Ideas pendientes / roadmap
```

## Cómo funciona cada archivo

### `knime_workflow_converter.py` (pipeline ML, 1156 líneas)

El cerebro del proyecto. Originalmente fue una conversión de un workflow KNIME (de ahí el nombre del archivo, ya legacy), pero hoy es un pipeline ML completo escrito en Python puro. 20 funciones, las clave:

- **`compute_elo_features(df)`** — recorre partidos cronológicamente, calcula ELO antes de cada uno (K=30, ventaja local=60 pts, bonus por margen de goles). Devuelve `team_elos` final.
- **`compute_form_features(df)`** — forma últimos 5: W/D/L, goles a favor/contra, puntos. Usa solo partidos anteriores → sin leakage.
- **`compute_h2h_features(df)`** — head-to-head: últimos 3 enfrentamientos directos entre los dos equipos.
- **`select_columns(df)`** — descarta stats post-partido que no se pueden usar (ej. posesión real, disparos al arco del partido).
- **`train_models(...)`** — entrena los 6 clasificadores con `SelectKBest(k=25)` + `class_weight='balanced'` dentro de un `Pipeline` para evitar leakage en CV.
- **`cross_validate_models(...)`** — CV walk-forward con `TimeSeriesSplit`.
- **`main(filepath)`** — corre todo: carga, ordena cronológicamente, calcula features, entrena, evalúa, guarda CSVs. Devuelve un dict con todos los results.
- **`predecir_partido(equipo1, equipo2, results, n_runs=20, fase='Liga')`** — predice un partido futuro entrenando cada modelo 20 veces con seeds distintos y promediando probabilidades. Devuelve dict estructurado con consenso, probas por modelo, marcador estimado, ELOs.

Outputs CSV: `model_results.csv` (accuracy + F1 macro por modelo en test cronológico), `predictions.csv` (test set vs predicción de cada modelo), `processed_data.csv` (dataset con features derivadas).

### `api.py` (FastAPI + SQLModel, 671 líneas)

API REST que expone el pipeline ML como servicio web y sirve el dashboard estático.

**4 tablas SQLite** (todas con campo `activo: bool` para soft delete):
- `Partido` — autocarga desde el Excel al startup si la tabla está vacía
- `Evaluacion` — cada vez que se corre `main()` se guarda accuracy + CV results
- `Prediccion` — historial de predicciones one-off, captura stdout de `predecir_partido()`
- `PrediccionTrack` — predicciones públicas con autoresolución cuando llega el resultado real

**Cache en memoria**: `_resultados_pipeline: dict[int, dict]` guarda los models entrenados por evaluación. Si el server se reinicia (ej. cold start de Render), `_get_or_run_pipeline()` los reentrena automáticamente.

**Auto-resolución del track record** (`api.py:211-242`): cuando un `PUT /partidos/{id}` recibe goles nuevos, busca tracks pendientes con esos dos equipos (en cualquier orden), invierte los goles si el orden está al revés, y marca `acierto = (pred_consenso == resultado_real)`.

### `scraper_uefa.py` (Playwright, 436 líneas)

Scrapea stats de partidos UCL desde es.uefa.com. Dos modos:
- `obtener_info_partido(url)` — un partido específico
- `listar_partidos_por_fecha(fecha)` — todos los partidos de una jornada

Extrae stats de los `pk-list-stat-item` del DOM y fecha/marcador del JSON embebido. Mapea 70+ alias de nombres de equipos al esquema del dataset (ej. "Atlético de Madrid" → "Atleti").

### `agregar_partido.py` (CLI, 417 líneas)

Agrega partidos al Excel. 3 modos:

```bash
python agregar_partido.py                                      # interactivo (pega stats manualmente)
python agregar_partido.py --url <URL>                          # scrapea una URL
python agregar_partido.py --fecha 2026-03-10 --fase Octavos    # toda una jornada
```

El modo `--fecha` lanza un subprocess por partido para evitar que Chromium acumule recursos y se cuelgue. Dedup por `(Equipo1, Equipo2, Fecha)` normalizada.

### `static/` (dashboard, ~800 líneas total)

Frontend vanilla — sin build step, sin framework, sin npm. Tailwind y Chart.js cargados por CDN.

- **`index.html`** — markup. Secciones: KPIs, ranking ELO, métricas CV, predicciones de test, predictor interactivo, track record público, historial de partidos.
- **`js/api.js`** — wrapper de `fetch` con manejo de errores y URL base configurable (`window.API_BASE_URL`).
- **`js/app.js`** — lógica del dashboard. Carga datos al arrancar, renderiza tablas, dibuja gráficos con Chart.js, maneja el form del predictor.

## Setup local

```bash
pip install -r requirements.txt
python -m playwright install chromium     # solo si vas a scrapear
uvicorn api:app --reload
```

Abrí <http://localhost:8000> para el dashboard. Documentación interactiva del API en <http://localhost:8000/docs>.

Si solo querés correr el pipeline una vez sin el server:

```bash
python knime_workflow_converter.py
```

## Endpoints principales

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Dashboard web |
| GET | `/api/health` | Health check + evaluaciones en memoria |
| GET | `/api/equipos` | Lista de equipos del dataset |
| GET | `/partidos`, `/partidos/{id}` | Listar / por ID (filtro `?equipo1=&equipo2=`) |
| PUT | `/partidos/{id}` | Actualizar goles/fecha → dispara autoresolución de tracks |
| POST | `/evaluaciones` | Reentrena el pipeline completo |
| GET | `/api/evaluaciones/{id}/elos` | Ranking ELO |
| GET | `/api/evaluaciones/{id}/metricas` | Accuracy + F1 (test y CV) |
| GET | `/api/evaluaciones/{id}/feature-importance` | Top features del RF |
| GET | `/api/evaluaciones/{id}/predicciones-test` | Test set vs predicciones por modelo |
| POST | `/api/predecir` | Predicción rápida (JSON, no guarda) |
| POST | `/api/track` | Predecir + guardar al track record público |
| GET | `/api/track`, `/api/track/stats` | Listar / stats agregadas del track |
| GET | `/docs` | Swagger UI |

## Deploy

### Render.com (lo que está usado)

1. Push del repo a GitHub
2. En Render: **New → Blueprint** → conectar el repo
3. Render detecta `render.yaml` y crea el servicio automáticamente
4. Para que SQLite persista entre deploys, descomentar el bloque `disk:` en `render.yaml`

Variables de entorno (`PORT`, `ALLOWED_ORIGINS`, `DATABASE_URL`, `DATASET_PATH`) se configuran en el dashboard de Render. Default OK para empezar.

### Cualquier Docker host (Railway, Fly.io, VPS, etc.)

```bash
docker build -t ucl-predictor .
docker run -p 8000:8000 -e ALLOWED_ORIGINS="*" ucl-predictor
```

## Métricas actuales (81 partidos, todos en fase Liga UCL 2025-26)

Validación walk-forward CV (3 folds):

| Modelo | CV F1 mean | F1 último fold | Acc test cronológico |
|---|---|---|---|
| Logistic Regression | 0.63 | 0.50 | 0.71 |
| Random Forest | 0.58 | 0.44 | 0.59 |
| KNN | 0.47 | 0.44 | 0.41 |
| Gradient Boosting | 0.50 | 0.28 | 0.59 |
| SVM | 0.51 | 0.34 | 0.47 |
| XGBoost | 0.45 | 0.24 | 0.47 |

F1 último fold ≈ rendimiento esperado en partidos reales próximos. **0.50 está al nivel de las casas de apuestas profesionales** — bueno para 81 partidos. Mejora esperada con dataset >200 partidos y al incorporar eliminatorias.

## Stack

- **Backend**: Python 3.12 · FastAPI · SQLModel · SQLite (default) / Postgres-compatible
- **ML**: scikit-learn · XGBoost · pandas · numpy
- **Scraping**: Playwright · Chromium headless
- **Frontend**: HTML + Tailwind CDN + Chart.js CDN (sin build step)
- **Deploy**: Docker · Render Blueprint
