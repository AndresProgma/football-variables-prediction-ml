# 🎯 KNIME Workflow → Python Converter

## ¿Qué contiene?

Este proyecto contiene la **conversión completa** de tu workflow KNIME (`proyecto.knwf`) a Python puro.

### Archivos Incluidos:

1. **`knime_workflow_converter.py`** (19 KB)
   - Script principal con toda la lógica convertida
   - 10 funciones principales + 1 función `main()`
   - Totalmente comentado y documentado

2. **`DOCUMENTACION_CONVERSION.md`** (9 KB)
   - Explicación detallada de cada nodo
   - Mapeo KNIME → Python
   - Guía de uso avanzado
   - Troubleshooting

3. **`README.md`** (este archivo)
   - Instrucciones rápidas
   - Quick start

---

## ⚡ INSTALACIÓN RÁPIDA

### 1. Instala dependencias

```bash
pip install pandas numpy scikit-learn openpyxl
```

**Versiones recomendadas:**
- Python 3.7+
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- numpy >= 1.21.0

### 2. Prepara tu archivo Excel

Tu archivo debe tener columnas con datos deportivos como:
- `Equipo1`, `Equipo2`
- `EQUIPO1_GOLES`, `EQUIPO2_GOLES`
- Estadísticas de disparos, pases, etc.

---

## 🚀 USO BÁSICO (3 líneas)

```python
from knime_workflow_converter import main

# Ejecuta todo el pipeline
results = main("ruta/a/tu/archivo.xlsx")

print(results['results'])  # Ver resultados de modelos
```

---

## 📊 ¿Qué hace el script?

```
✓ Carga Excel
✓ Limpia valores faltantes
✓ Crea variables derivadas (Rule Engine)
✓ Filtra filas y columnas
✓ Agrupa datos por equipo (GroupBy)
✓ Unen múltiples tablas (Join)
✓ Entrena 3 modelos ML:
  • Random Forest
  • Gradient Boosting
  • Linear Regression
✓ Evalúa con RMSE, R², MAE
✓ Guarda resultados en CSV
```

---

## 📈 OUTPUTS

El script genera **3 archivos CSV**:

1. **`model_results.csv`**
   ```
   Model,RMSE,R²,MAE
   Random Forest,0.45,0.89,0.32
   Gradient Boosting,0.42,0.91,0.28
   Linear Regression,0.48,0.87,0.35
   ```

2. **`predictions.csv`**
   ```
   Actual,RandomForest,GradientBoosting,LinearRegression
   2.0,1.98,2.05,2.10
   3.0,3.12,2.95,3.20
   ...
   ```

3. **`processed_data.csv`**
   - Tu dataset completo procesado
   - Incluye variables derivadas
   - Listo para análisis posterior

---

## 🔧 PERSONALIZACIÓN

### Cambiar variables objetivo (target)

En `main()`, busca:
```python
# Por defecto:
y = df_model['EQUIPO1_GOLES']

# Cámbialo a:
y = df_model['TuVariableObjetivo']
```

### Ajustar parámetros de modelos

En `train_models()`:
```python
rf = RandomForestRegressor(
    n_estimators=200,  # ← Cambiar
    max_depth=20,      # ← Cambiar
    random_state=42
)
```

### Cambiar estrategia de imputación

```python
df = handle_missing_values(df, strategy='median')  # o 'drop'
```

---

## 📚 EJEMPLOS DE USO

### Ejemplo 1: Uso Completo

```python
from knime_workflow_converter import main

results = main("datos_futbol.xlsx", test_size=0.2)

# Ver resultados
print("Dataset procesado:")
print(results['df'].head())

print("\nResultados de modelos:")
print(results['results'])

print("\nPredicciones:")
print(results['predictions'].head())
```

### Ejemplo 2: Procesamiento Paso a Paso

```python
from knime_workflow_converter import *
import pandas as pd

# 1. Cargar
df = load_data("datos.xlsx")

# 2. Limpiar
df = handle_missing_values(df, strategy='mean')
df = filter_rows(df)

# 3. Transformar
df = create_derived_variables(df)

# 4. Preparar
df_model, le = prepare_for_modeling(df)

# 5. Entrenar manual
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=150)
model.fit(X_train, y_train)

print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

### Ejemplo 3: Comparar Modelos

```python
from knime_workflow_converter import *

results = main("datos.xlsx")
models_df = results['results']

# Ordenar por mejor R²
best = models_df.sort_values('R²', ascending=False)
print(best.to_string())

# Visualizar
import matplotlib.pyplot as plt
best.set_index('Model')['R²'].plot(kind='barh')
plt.show()
```

---

## 🐛 PROBLEMAS COMUNES

### ❌ `FileNotFoundError: archivo no encontrado`
```python
# ✓ Solución: usa ruta absoluta
results = main("/home/usuario/archivo.xlsx")
# o ruta relativa desde directorio actual
results = main("./datos/archivo.xlsx")
```

### ❌ `KeyError: Equipo1`
```python
# ✓ Primero revisa nombres exactos:
df = pd.read_excel("archivo.xlsx")
print(df.columns)  # Ver todos los nombres
```

### ❌ `ValueError: X has NaN values`
```python
# ✓ Solución: importar debe ser antes de modelado
df = handle_missing_values(df, strategy='mean')
```

### ❌ Bajo R² (< 0.5)
```python
# ✓ Intenta:
# 1. Más features (create_derived_variables)
# 2. Más datos (quitar filter_rows)
# 3. Otros modelos (cambiar n_estimators)
# 4. Normalizar features (StandardScaler)
```

---

## 📖 DOCUMENTACIÓN COMPLETA

Para detalles técnicos, mapeo completo de nodos y uso avanzado, ver:

👉 **`DOCUMENTACION_CONVERSION.md`**

---

## 🎓 CONCEPTOS CLAVE

### Modelos Incluidos

| Modelo | Tipo | Cuándo usar |
|--------|------|------------|
| **Random Forest** | Ensamble | Datos complejos, relaciones no-lineales |
| **Gradient Boosting** | Ensamble | Precisión máxima, ajuste fino |
| **Linear Regression** | Lineal | Referencia, interpretabilidad |

### Métricas

- **RMSE**: Error promedio (penaliza errores grandes)
- **R²**: Qué % de varianza explica el modelo (0-1)
- **MAE**: Error absoluto (más robusto a outliers)

### Train-Test Split

Por defecto 80-20:
```python
main("datos.xlsx", test_size=0.2)  # 80% train, 20% test
```

---

## 🌟 CARACTERÍSTICAS

✅ **Manejo automático de datos**
- Detecta tipos de columnas
- Imputa valores faltantes
- Codifica variables categóricas

✅ **3 modelos ML**
- Entrenamientoautomático
- Evaluación comparativa
- Predicciones en CSV

✅ **Código limpio**
- Bien comentado
- Modular y reutilizable
- Fácil de adaptar

✅ **Sin dependencias complejas**
- Solo: pandas, numpy, scikit-learn
- Funciona en cualquier sistema

---

## 💡 TIPS & TRICKS

### Guardar un modelo entrenado
```python
import pickle

results = main("datos.xlsx")
model = results['models']['Random Forest']

# Guardar
with open('mi_modelo.pkl', 'wb') as f:
    pickle.dump(model, f)

# Cargar
with open('mi_modelo.pkl', 'rb') as f:
    modelo_cargado = pickle.load(f)
```

### Feature Importance (RF)
```python
model = results['models']['Random Forest']
importances = model.feature_importances_

# Top 10
top_10 = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)[:10]
for feature, importance in top_10:
    print(f"{feature}: {importance:.4f}")
```

### Visualizar predicciones
```python
import matplotlib.pyplot as plt

pred = results['predictions']
plt.scatter(pred['Actual'], pred['RandomForest'])
plt.xlabel('Actual')
plt.ylabel('Predicción')
plt.title('Random Forest')
plt.plot([pred['Actual'].min(), pred['Actual'].max()], 
         [pred['Actual'].min(), pred['Actual'].max()], 'r--')
plt.show()
```

---

## 📞 SOPORTE

Preguntas frecuentes:

1. **¿Necesito KNIME instalado?**
   No, el script es 100% Python

2. **¿Puedo usar otros datos?**
   Sí, cualquier CSV/Excel con estructura similar

3. **¿Cuánto tarda en ejecutar?**
   Depende del tamaño: 1K filas = <1 segundo, 100K filas = 10-30 segundos

4. **¿Puedo agregar más modelos?**
   Sí, modifica `train_models()` para agregar XGBoost, SVM, etc.

---

## 📄 LICENCIA

Este código es de libre uso. Siéntete libre de modificarlo y compartirlo.

---

**¡Listo para usar!** 🚀

```bash
python -c "from knime_workflow_converter import main; main('tu_archivo.xlsx')"
```
