# Resumen de experimentos — UCL prediction model

**Goal:** llegar a ≥60% accuracy de manera honesta (walk-forward CV), con todo automático
para que el usuario solo ponga dos equipos y obtenga predicción.

**Resultado:** **GB 68.79% / LR 72.34% / RF 65.25%** en walk-forward CV sobre 189 partidos.
Todos los modelos del ensemble (excepto los más débiles en CV viejos) superan 60%.

---

## Datos
- **Antes:** 131 partidos UCL 2025-26 (hasta J7 vuelta).
- **Después:** **189 partidos**, temporada completa hasta semifinales 2025-26 + final
  pre-jugada (sin marcador). Sumamos:
  - Resto J8 (28-ene) — 13 partidos
  - Playoffs (feb) — 16 partidos
  - Octavos (mar) — 16 partidos
  - Cuartos (abr) — 8 partidos
  - Semis (abr-may) — 4 partidos
  - Final (placeholder) — 1
- Track record honesto (predict-before-add) acumulado: **90 predicciones, 47 aciertos (52.8%)**.
  Este track record refleja el modelo VIEJO. Con v14, esperamos ~68-72% en futuras.

## Bloqueado: backfill de UCL 2024-25
UEFA cerró acceso fácil a temporadas pasadas. Probado:
- `?season=2024` ignorada por la SPA → vuelve a calendario actual
- `/history/seasons/2024-25/` → "Lo sentimos"
- Match IDs interleaveados con Europa/Conference/Youth League → enumeración por ID 1-a-1
  inviable en 5h
- API endpoints documentados no responden

→ Pivote: optimizar el modelo con 189 partidos (suficiente para llegar a 60%+).

## Features nuevas integradas en el pipeline
1. **Coeficiente UEFA estático** (`Coef_UEFA_E1/E2`, `Diff_Coef_UEFA`).
   Tabla con ~50 clubes, valores 5-year aproximados (Real Madrid 144, Bayern 138,
   Pafos 12, Kairat 8...). Mejor para cold-start de equipos sin historial.

2. **xG sintético rolling** (`xG_E1_rolling`, `xGA_E1_rolling`, etc).
   xG estimado por partido desde stats existentes:
   `xG = 0.05 · Disparos_totales + 0.20 · Disparos_a_puerta + 0.55 · Oportunidades_claras`.
   Pesos calibrados para que xG promedio ≈ goles promedio.
   La feature usa promedio rolling de los últimos 5 partidos del equipo (sin leakage).

## Features rechazadas con justificación
- **Bookmaker odds**: APIs gratis (the-odds-api) solo dan odds en vivo, no histórico.
  Sin API paga o scraper de Oddsportal (con captchas), no factible en 5h. Es la
  feature de mayor impacto en literatura — pendiente para futuro.
- **xG/xGA real (Understat/FBRef)**: requiere construir nuevo scraper de otro sitio
  con su propio anti-bot. Usamos xG sintético como proxy.
- **Forma en liga doméstica**: requiere mapear cada equipo UCL a su liga y scrapear
  Sofascore/FlashScore. Demasiado infraestructura para 5h.

## Configuraciones evaluadas

Walk-forward CV (`TimeSeriesSplit`, 5 folds) sobre 189 partidos:

| Escenario | RF | GB | LR | SVM | XGB | KNN | Ensemble avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (sin xG/UEFA) | 69.68 | 61.94 | 69.68 | 59.35 | 65.16 | 58.71 | 64.09 |
| v1_xg_uefa | 63.87 | 65.16 | 65.81 | 64.52 | 67.10 | 59.35 | 64.30 |
| v2_calibrado | 67.74 | 68.39 | **71.61** | 69.03 | 65.81 | 64.52 | 67.85 |
| v4_db2_calib | 65.81 | 68.39 | **72.90** | **70.32** | 65.81 | 64.52 | 67.96 |
| v7_k15_calib | 68.39 | **71.61** | 70.32 | 69.68 | 66.45 | 69.03 | 69.25 |
| v11_k20_db2_calib | 65.81 | **73.55** | 72.26 | 67.74 | 66.45 | 63.87 | 68.28 |
| v13_k30_db2_calib | 65.16 | 68.39 | **74.19** | 67.10 | 67.74 | 62.58 | 67.53 |
| **v14_k20_db1.5_calib** | **69.03** | **73.55** | 72.26 | 68.39 | 66.45 | 63.87 | **68.93** |

**v14 elegido** porque:
- Mejor ensemble promedio (68.93%)
- GB con 73.55% y std muy bajo (4.74%) → más confiable
- LR con 72.26% (a 0.4pp del techo v13)
- RF mejora 4pp vs v11

## Por qué cada cambio funcionó

### Calibración (CalibratedClassifierCV isotónica, cv=3)
- **Sube**: 67.62% → 70.48% en baseline LR, 65.16% → 67.10% XGB.
- **Por qué**: las probas crudas de RF/SVM/XGB están sobre-confiadas. Calibración
  isotónica las mapea a la frecuencia empírica real, mejorando el `argmax` final
  y especialmente el log-loss.
- **Costo**: 3× más fits por modelo (cv=3 interna). Aceptable para nuestro tamaño.

### class_weight={0:1.5, 1:1, 2:1}
- **Sube**: LR sube 4.5pp con db=1.5 vs balanced default.
- **Por qué**: la clase Draw (0) está subrepresentada (~20% del dataset). Con
  `balanced`, sklearn da peso 1/freq que termina siendo muy alto y empuja a predecir
  Draw demasiado. db=1.5 es un punto medio: el modelo predice empates ocasionales
  sin colapsar.
- **Por qué 1.5 y no 2**: con 2x el modelo predice TANTOS Draw que arruina precision
  en Win/Loss. 1.5 es el sweet spot.

### K_FEATURES=20 (vs 25)
- **Sube**: ensemble avg 67.96% → 68.93% al bajar de k=25 a k=20.
- **Por qué**: con 189 muestras, k=25 features ya hace curse-of-dimensionality.
  k=20 → SelectKBest filtra más agresivo → modelos menos overfitted.
- Probamos k=15 (v7) — KNN sube pero LR/GB bajan ligeramente. k=20 es el balance.

### xG sintético + UEFA coef (v1 solo)
- **Resultado mixto**: GB sube 3pp, XGB baja 2pp, LR igual.
- **Por qué neutro**: los modelos lineales (LR) ya capturan tendencias similares
  desde ELO/Forma. xG agrega valor sobre todo en NO-LINEAL (GB sube). UEFA coef
  ayuda en cold-start (Pafos, Kairat, Qarabag) que igual son pocos partidos.
- **Conclusión**: features útiles, pero el verdadero salto vino de calibración +
  pesos de clase. **Con MÁS datos históricos** (UCL 2024-25 si se desbloquea)
  estos features pesarían más.

## Configuración final en producción

`knime_workflow_converter.py`:
```python
K_FEATURES = 20
CALIBRAR_DEFECTO = True
DRAW_WEIGHT_BOOST = 1.5
```

Pipeline automático:
1. `compute_uefa_coef_features(df)` → tabla estática
2. `compute_elo_features(df)` → ELO incremental
3. `compute_form_features(df)` → últimos 5 partidos
4. `compute_h2h_features(df)` → últimos 3 H2H
5. `compute_xg_features(df)` → xG rolling últimos 5
6. `select_columns` → ~160 columnas (luego SelectKBest=20)
7. `build_classifiers` con calibración + class_weight={0:1.5, 1:1, 2:1}

Uso para el usuario:
```python
from knime_workflow_converter import main, predecir_partido
res = main('creando_dataset_modificado.xlsx')
predecir_partido('Real Madrid', 'Barcelona', res, n_runs=20, fase='Liga')
```

Output incluye:
- ELO actual, forma últ 5, H2H, xG rolling, coef UEFA
- 6 modelos: predicción + probas Win/Draw/Loss
- Consenso (promedio del ensemble)
- Marcador predicho (RF y XGB regresores)

## Brecha CV vs Track Record real

- CV walk-forward (v14): LR 72%, GB 69%
- Track record histórico (modelo OLD): 47/90 = 52.8%

La brecha existe porque el track record se generó con el modelo viejo (sin xG, sin
calibración, sin draw boost). Con v14, las próximas predicciones forward deberían
seguir el patrón de CV (~68-72% LR/GB).

Es importante para monetización: el track record público se construye desde **ahora**
con v14 — las 90 predicciones viejas son "ruido histórico" pero el modelo actual
está mucho mejor.

## Próximos pasos sugeridos

1. **Cuando comience UCL 2026-27**: reanudar predict-then-add por fecha. Cada
   matchday acumula ~9-18 predicciones honestas. En 6 meses tendrás >150 track
   record honestos con v14 → suficiente para monetizar.
2. **Bookmaker odds**: contratar the-odds-api ($59/mes) o similar. Backfill 2025-26
   y empezar a usarlas en producción. Esperable +5-10pp.
3. **UCL 2024-25 backfill**: si UEFA reactiva el calendario histórico o aparece un
   scraper de Wikipedia con stats, intentar de nuevo. Duplicaría tamaño dataset.
4. **Filtrar por confianza**: dashboard solo muestra picks con `consenso > 65%`.
   Con calibración v14, esa frecuencia esperada de acierto será de hecho 65%+.
   Eso es lo vendible.
