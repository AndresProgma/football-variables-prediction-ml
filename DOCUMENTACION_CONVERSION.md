# 📋 Conversión de Workflow KNIME a Python
## Análisis de Datos Deportivos - Predicción de Goles

---

## 📌 RESUMEN EJECUTIVO

El workflow KNIME original (`proyecto.knwf`) es un **pipeline complejo de análisis deportivo** que:

1. **Carga datos** de partidos de fútbol (Excel)
2. **Limpia y transforma** múltiples variables estadísticas
3. **Crea nuevas características** (variables derivadas)
4. **Filtra y agrega** datos por equipo
5. **Entrena 3 modelos ML** (Random Forest, Gradient Boosting, Linear Regression)
6. **Evalúa modelos** usando múltiples métricas
7. **Visualiza resultados** (scatter plots, line plots)

**Complejidad:** 135+ nodos KNIME

---

## 🔄 MAPEO DE NODOS KNIME → PYTHON

### Grupos de Nodos y sus Equivalentes

| Nodo KNIME | Tipo | Función | Equivalente Python |
|-----------|------|---------|-------------------|
| Excel Reader (#4) | Input | Cargar datos | `pd.read_excel()` |
| Column Filter (#5, #43, #44, #75, #76, #99, #102, #129, #130) | Transform | Seleccionar columnas | `df[columns]`, `df.drop()` |
| Statistics (#6) | Analytic | Estadísticas básicas | `df.describe()` |
| Linear Correlation (#7) | Analytic | Correlación | `df.corr()` |
| Scatter Plot Matrix (#8) | Visualization | Matriz de scatter | `matplotlib`, `seaborn` |
| Parallel Coordinates Plot (#9) | Visualization | Gráfico paralelo | `pandas.plotting` |
| Random Forest Learner (#13) | ML | Entrenar RF | `RandomForestRegressor()` |
| Rule Engine (#18) | Transform | Crear variables | Funciones custom + `apply()` |
| GroupBy (#20, #26, #32, etc.) | Aggregation | Agrupar datos | `df.groupby()` |
| Row Filter (#21, #23, #34, etc.) | Transform | Filtrar filas | `df[condition]`, `df.dropna()` |
| Random Forest Predictor (#45) | ML | Predicciones RF | `model.predict()` |
| Numeric Scorer (#46, #65, #101, #119) | Evaluation | Evaluar modelo | `sklearn.metrics` |
| Column Renamer (#30, #38, #50, #56, #58, #59, #85, #87, #88, #110, #112, #113) | Transform | Renombrar columnas | `df.rename()` |
| Concatenate (#31, #39, #57, #60, #86, #90, #111, #114) | Transform | Unir tablas | `pd.concat()`, `pd.append()` |
| Missing Value (#33, #41, #62, #71, #91, #94, #116, #125) | Transform | Imputar NaNs | `fillna()`, `interpolate()` |
| Joiner (#42, #73, #95, #127) | Transform | Join tablas | `pd.merge()` |
| Column Name Extractor (#47, #48) | Transform | Extraer nombres | Manipulación de strings |
| Reference Row Filter (#49) | Transform | Filtro referencial | Lógica personalizada |
| Gradient Boosted Trees Learner (#79) | ML | Entrenar GB | `GradientBoostingRegressor()` |
| Gradient Boosted Trees Predictor (#82) | ML | Predicciones GB | `model.predict()` |
| Simple Regression Tree Learner (#77) | ML | Entrenar árbol | `DecisionTreeRegressor()` |
| Simple Regression Tree Predictor (#80) | ML | Predicciones árbol | `model.predict()` |
| Linear Regression Learner (#131) | ML | Entrenar LR | `LinearRegression()` |
| Regression Predictor (#132) | ML | Predicciones LR | `model.predict()` |
| Data Generator (#134) | Input | Generar datos | `numpy`, `pandas` |
| Scatter Plot (#133) | Visualization | Scatter plot | `matplotlib.pyplot.scatter()` |
| Line Plot (#136) | Visualization | Gráfico línea | `matplotlib.pyplot.plot()` |

---

## 📊 FLUJO DE DATOS

```
1. ENTRADA
   └─ Excel Reader (#4) → Raw Data

2. LIMPIEZA Y SELECCIÓN
   ├─ Column Filter (#5) → Seleccionar features
   └─ Statistics (#6) → Análisis descriptivo

3. TRANSFORMACIÓN
   ├─ Rule Engine (#18) → Variables derivadas
   ├─ Missing Value → Imputación
   ├─ Row Filter → Filtrado de filas
   └─ Column Renamer → Renombrado

4. AGREGACIÓN
   └─ GroupBy (#20, #26, #32...) → Estadísticas por grupo

5. FUSIÓN
   └─ Joiner (#42, #73...) → Combinar tablas

6. MODELADO
   ├─ Random Forest Learner (#13) → Entrenamiento
   ├─ Gradient Boosting Learner (#79) → Entrenamiento
   ├─ Linear Regression Learner (#131) → Entrenamiento
   └─ Predictors (#45, #82, #132) → Predicciones

7. EVALUACIÓN
   └─ Numeric Scorer (#46, #65...) → Métricas RMSE, R², MAE

8. VISUALIZACIÓN
   ├─ Scatter Plot (#133)
   └─ Line Plot (#136)

9. SALIDA
   └─ Resultados de modelos y predicciones
```

---

## 🚀 GUÍA DE USO

### Requisitos

```bash
pip install pandas numpy scikit-learn openpyxl matplotlib seaborn
```

### Ejecución Básica

```python
from knime_workflow_converter import main

# Ejecutar pipeline completo
results = main("/ruta/al/archivo.xlsx")

# Acceder a resultados
df_processed = results['df']
models = results['models']
results_df = results['results']
predictions_df = results['predictions']
```

### Uso Avanzado

```python
# Entrenar solo un modelo
from knime_workflow_converter import *

# 1. Cargar y preparar datos
df = load_data("datos.xlsx")
df = handle_missing_values(df)
df = create_derived_variables(df)
df = filter_rows(df)

# 2. Preparar para modelado
df_model, le_dict = prepare_for_modeling(df)

# 3. Entrenar modelo específico
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, max_depth=15)
model.fit(X_train, y_train)

# 4. Hacer predicciones
predictions = model.predict(X_test)
```

---

## 📝 ESTRUCTURA DEL CÓDIGO PYTHON

### Funciones Principales

1. **load_data(filepath)**
   - Lee archivo Excel
   - Retorna DataFrame con datos crudos

2. **select_columns(df)**
   - Filtra columnas relevantes
   - Similar a Column Filter de KNIME

3. **handle_missing_values(df, strategy='mean')**
   - Imputa valores faltantes
   - Estrategias: mean, median, forward_fill, drop

4. **create_derived_variables(df)**
   - Crea características nuevas
   - Eficiencia de tiros, precisión de pases, etc.

5. **filter_rows(df)**
   - Elimina filas según criterios
   - Maneja duplicados

6. **aggregate_by_team(df)**
   - Agrupa por equipo
   - Calcula estadísticas (media, suma, desv. est.)

7. **join_team_stats(df, team_stats)**
   - Une estadísticas al dataset original
   - Enriquece features

8. **prepare_for_modeling(df)**
   - Codifica variables categóricas
   - Rellenan valores faltantes
   - Retorna datos listos para ML

9. **train_models(X_train, y_train, X_test, y_test)**
   - Entrena 3 modelos
   - RF, GB, LR
   - Retorna predicciones y scores

10. **evaluate_models(models, predictions, y_test)**
    - Calcula RMSE, R², MAE
    - Imprime resumen comparativo

---

## 📈 MÉTRICAS DE EVALUACIÓN

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **RMSE** | √(Σ(ŷ-y)²/n) | Raíz del error cuadrático medio. Más bajo = mejor |
| **R²** | 1 - (SS_res/SS_tot) | Proporción de varianza explicada. 0-1, más alto = mejor |
| **MAE** | Σ\|ŷ-y\|/n | Error absoluto medio. Más bajo = mejor |

---

## 🔧 CÓMO ADAPTARLO A TUS DATOS

### Paso 1: Identifica tus columnas
```python
df = pd.read_excel("tu_archivo.xlsx")
print(df.columns)
print(df.head())
```

### Paso 2: Actualiza `select_columns()`
```python
# En la función select_columns(), cambia columns_to_keep:
columns_to_keep = [
    'TuColumna1', 'TuColumna2', 'TuColumna3', ...
]
```

### Paso 3: Personaliza variables derivadas
```python
# Agrega reglas en create_derived_variables()
df['MiNuevaVariable'] = df['Col1'] / df['Col2']
```

### Paso 4: Especifica target
```python
# En main(), cambia la línea de target:
y = df_model['TuVariableObjetivo']
```

### Paso 5: Ejecuta
```python
results = main("tu_archivo.xlsx", test_size=0.2)
```

---

## 📊 ARCHIVOS DE SALIDA

| Archivo | Contenido | Uso |
|---------|-----------|-----|
| `model_results.csv` | RMSE, R², MAE de cada modelo | Comparación de modelos |
| `predictions.csv` | Valores reales vs predicciones | Análisis de errores |
| `processed_data.csv` | Dataset completo procesado | Análisis posterior |

---

## ⚙️ CONFIGURACIÓN AVANZADA

### Parámetros de Modelos

```python
# Random Forest
RandomForestRegressor(
    n_estimators=100,    # Número de árboles
    max_depth=20,        # Profundidad máxima
    min_samples_split=5, # Muestras mínimas para split
    random_state=42
)

# Gradient Boosting
GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Linear Regression
LinearRegression()
```

### Validación Cruzada

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y, 
    cv=5,  # 5-fold cross-validation
    scoring='r2'
)
print(f"R² Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 🐛 TROUBLESHOOTING

| Problema | Solución |
|----------|----------|
| `FileNotFoundError` | Verifica la ruta del archivo Excel |
| `KeyError: columna` | Revisa nombres exactos de columnas con `df.columns` |
| `ValueError: X has NaN values` | Llama `handle_missing_values()` antes de modelado |
| `MemoryError` | Reduce `n_estimators` en modelos o usa `sample_frac()` |
| Bajo R² | Agrega más features, ajusta hiperparámetros, revisa target |

---

## 📚 REFERENCIAS

- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Scikit-learn**: https://scikit-learn.org/stable/
- **NumPy**: https://numpy.org/doc/
- **KNIME Analytics**: https://www.knime.com/

---

**Creado el:** Abril 2026  
**Versión:** 1.0  
**Estado:** ✅ Completo y Funcional

