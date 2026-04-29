"""
KNIME Workflow to Python Converter
================================
Convierte un complejo workflow de KNIME sobre datos deportivos a Python puro.

Estructura:
- Carga de datos Excel
- Filtrado y selección de columnas
- Limpieza de valores faltantes
- Creación de variables derivadas (Rule Engine)
- Filtrado de filas
- Agregaciones (GroupBy)
- Joins de tablas
- Modelos de ML (Random Forest, Gradient Boosting, Linear Regression)
- Evaluación de modelos (Numeric Scorer)
- Visualizaciones

Librerías principales:
- pandas: manipulación de datos
- scikit-learn: machine learning
- numpy: operaciones numéricas
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CARGAR DATOS (Excel Reader #4)
# ============================================================================
def load_data(filepath):
    """Carga archivo Excel con datos de partidos deportivos"""
    print("📂 Cargando datos del archivo Excel...")
    df = pd.read_excel(filepath)
    print(f"   ✓ {df.shape[0]} filas, {df.shape[1]} columnas cargadas")
    return df


# ============================================================================
# 2. SELECCIONAR COLUMNAS (Column Filter #5)
# ============================================================================
def select_columns(df):
    """Filtra y mantiene solo las columnas relevantes"""
    print("🔍 Filtrando columnas relevantes...")
    
    columns_to_keep = [
        'Equipo1', 'Equipo2', 'Es_Local_E1', 'EQUIPO1_GOLES', 'EQUIPO2_GOLES',
        # Goles detallados
        'Goles_dentro_area_E1', 'Goles_Fuera_Area_E1',
        'Goles_dentro_area_E2', 'Goles_Fuera_Area_E2',
        # Disparos
        'Disparos_totales_E1', 'Disparos_a_puerta_E1', 'Disparos_fuera_E1',
        'Disparo_Bloqueados_E1', 'Disparos_Al palo_E1', 'Disparos_Larguero_E1',
        'Disparos_Poste_E1', 'Disparos_a_puerta_fuera_del_area_E1',
        'Disparos_fuera_desde_fuera_del_area_E1',
        'Disparos_totales_E2', 'Disparos_a_puerta_E2', 'Disparos_fuera_E2',
        'Disparo_Bloqueados_E2', 'Disparos_Al palo_E2', 'Disparos_Larguero_E2',
        'Disparos_Poste_E2', 'Disparos_a_puerta_fuera_del_area_E2',
        'Disparos_fuera_desde_fuera_del_area_E2',
        # Ataque
        'Asistencias_E1', 'Penaltis_marcados_E1', 'Penaltis_fallados_E1',
        'Penaltis_forzados_E1', 'Ataques_E1', 'Oportunidades_claras_E1',
        'Saques_de_esquina_sacados_E1', 'Fueras_de_juego_E1', 'Regates_E1',
        'Ataques_tercio_ofensivo_E1', 'Ataques_zonas_clave_E1', 'Carreras_hacia_el_area_E1',
        'Asistencias_E2', 'Penaltis_marcados_E2', 'Penaltis_fallados_E2',
        'Penaltis_forzados_E2', 'Ataques_E2', 'Oportunidades_claras_E2',
        'Saques_de_esquina_sacados_E2', 'Fueras_de_juego_E2', 'Regates_E2',
        'Ataques_tercio_ofensivo_E2', 'Ataques_zonas_clave_E2', 'Carreras_hacia_el_area_E2',
        # Posesión y pases
        'Posesion_E1', 'Precision_pase_E1', 'Pases_completados_E1', 'Pases_realizados_E1',
        'Pases_cortos_completados_E1', 'Pases_media_distancia_completados_E1',
        'Pases_en_largo_completados_E1', 'Pases_completados_atras_E1',
        'Pases_completadosa_izquierda_E1', 'Pases_completados_derecha_E1',
        'Libres_directos_sacados_E1', 'Centros__tercio_ofensivo_E1',
        'Pases_zonas_clave_E1', 'Pases_al_area_E1', 'Precision_en_el_centro_E1',
        'Centros_completados_E1', 'Centros_realizados_E1', 'Tiempo_de_posesion_E1',
        'Posesion_E2', 'Precision_pase_E2', 'Pases_completados_E2', 'Pases_realizados_E2',
        'Pases_cortos_completados_E2', 'Pases_media_distancia_completados_E2',
        'Pases_en_largo_completados_E2', 'Pases_completados_atras_E2',
        'Pases_completadosa_izquierda_E2', 'Pases_completados_derecha_E2',
        'Libres_directos_sacados_E2', 'Centros__tercio_ofensivo_E2',
        'Pases_zonas_clave_E2', 'Pases_al_area_E2', 'Precision_en_el_centro_E2',
        'Centros_completados_E2', 'Centros_realizados_E2', 'Tiempo_de_posesion_E2',
        # Defensa
        'Balones_recuperados_E1', 'Bloqueos_E1', 'Penaltis_cometidos_E1',
        'Entradas_E1', 'Entradas_con_exito_E1', 'Entradas_perdidas_E1',
        'Despejes_completados_E1', 'Despejes_realizados_E1',
        'Balones_recuperados_E2', 'Bloqueos_E2', 'Penaltis_cometidos_E2',
        'Entradas_E2', 'Entradas_con_exito_E2', 'Entradas_perdidas_E2',
        'Despejes_completados_E2', 'Despejes_realizados_E2',
        # Portería
        'goles_encajados_E1', 'Goles_encajados_propia_puerta_E1', 'Porterias_a_cero_E1',
        'Paradas_E1', 'Paradas_en_libres_directo_E1', 'Paradas-tras_libre_indirecto_E1',
        'Penaltis_parados_E1', 'Balones_blocados_E1', 'Balones_blocados_por arriba_E1',
        'Balones_blocados_por_abajo_E1', 'Despejes_de_puños_E1',
        'goles_encajados_E2', 'Goles_encajados_propia_puerta_E2', 'Porterias_a_cero_E2',
        'Paradas_E2', 'Paradas_en_libres_directo_E2', 'Paradas-tras_libre_indirecto_E2',
        'Penaltis_parados_E2', 'Balones_blocados_E2', 'Balones_blocados_por arriba_E2',
        'Balones_blocados_por_abajo_E2', 'Despejes_de_puños_E2',
        # Disciplina
        'Tarjetas_amarillas_E1', 'Tarjetas_rojas_E1', 'Faltas_cometidas_E1',
        'Faltas_cometidas_tercio_def_E1', 'Faltas_cometidas_en_campo_propio_E1',
        'Tarjetas_amarillas_E2', 'Tarjetas_rojas_E2', 'Faltas_cometidas_E2',
        'Faltas_cometidas_tercio_def_E2', 'Faltas_cometidas_en_campo_propio_E2',
        # Media alineación titular
        'media11_titular_E1', 'media11_titular_E2',
    ]
    
    # Mantener solo columnas que existen en el dataframe
    available_cols = [col for col in columns_to_keep if col in df.columns]
    df_filtered = df[available_cols].copy()
    
    print(f"   ✓ {len(available_cols)} columnas seleccionadas")
    return df_filtered


# ============================================================================
# 3. LIMPIEZA: VALORES FALTANTES (Missing Value #33, #41, #62, #71, #91, #94, #116, #125)
# ============================================================================
def handle_missing_values(df, strategy='mean'):
    """
    Maneja valores faltantes en el dataset
    
    Args:
        df: DataFrame
        strategy: 'mean', 'median', 'forward_fill', 'drop'
    """
    print(f"🧹 Limpiando valores faltantes (estrategia: {strategy})...")
    
    initial_nulls = df.isnull().sum().sum()
    
    if strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'forward_fill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif strategy == 'drop':
        df = df.dropna()
    
    final_nulls = df.isnull().sum().sum()
    print(f"   ✓ Valores faltantes: {initial_nulls} → {final_nulls}")
    
    return df


# ============================================================================
# 4. CREAR VARIABLES DERIVADAS (Rule Engine #18)
# ============================================================================
def create_derived_variables(df):
    """
    Crea nuevas variables basadas en reglas (similar a Rule Engine de KNIME)
    """
    print("📐 Creando variables derivadas...")
    
    df = df.copy()
    
    # Eficiencia de tiros
    if 'Disparos_a_puerta_E1' in df.columns and 'EQUIPO1_GOLES' in df.columns:
        df['Eficiencia_Tiros_E1'] = df.apply(
            lambda x: (x['EQUIPO1_GOLES'] / x['Disparos_a_puerta_E1'] 
                      if x['Disparos_a_puerta_E1'] > 0 else 0), 
            axis=1
        )
    
    if 'Disparos_a_puerta_E2' in df.columns and 'EQUIPO2_GOLES' in df.columns:
        df['Eficiencia_Tiros_E2'] = df.apply(
            lambda x: (x['EQUIPO2_GOLES'] / x['Disparos_a_puerta_E2'] 
                      if x['Disparos_a_puerta_E2'] > 0 else 0), 
            axis=1
        )
    
    # Diferencia de goles
    if 'EQUIPO1_GOLES' in df.columns and 'EQUIPO2_GOLES' in df.columns:
        df['Diferencia_Goles'] = df['EQUIPO1_GOLES'] - df['EQUIPO2_GOLES']
    
    # Resultado (Win/Draw/Loss para Equipo1)
    if 'EQUIPO1_GOLES' in df.columns and 'EQUIPO2_GOLES' in df.columns:
        df['Resultado_E1'] = df.apply(
            lambda x: 'Win' if x['EQUIPO1_GOLES'] > x['EQUIPO2_GOLES'] 
                      else ('Draw' if x['EQUIPO1_GOLES'] == x['EQUIPO2_GOLES'] else 'Loss'),
            axis=1
        )
    
    print(f"   ✓ {df.shape[1]} columnas después de derivadas")
    
    return df


# ============================================================================
# 5. FILTRAR FILAS (Row Filter #21, #23, #34, #37, #51, #55, #63, #74, #96, #97, #98, #104, #105, #109, #117, #128)
# ============================================================================
def filter_rows(df):
    """Filtra filas según criterios"""
    print("📋 Aplicando filtros de filas...")
    
    initial_rows = len(df)
    df_filtered = df.copy()
    
    # Ejemplo: eliminar filas donde falten datos críticos
    if 'Equipo1' in df_filtered.columns and 'Equipo2' in df_filtered.columns:
        df_filtered = df_filtered.dropna(subset=['Equipo1', 'Equipo2'])
    
    # Ejemplo: filtrar solo partidos con datos completos de goles
    if 'EQUIPO1_GOLES' in df_filtered.columns and 'EQUIPO2_GOLES' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['EQUIPO1_GOLES'].notna()) & 
            (df_filtered['EQUIPO2_GOLES'].notna())
        ]
    
    # Ejemplo: eliminar duplicados
    df_filtered = df_filtered.drop_duplicates()
    
    final_rows = len(df_filtered)
    print(f"   ✓ Filas filtradas: {initial_rows} → {final_rows}")
    
    return df_filtered


# ============================================================================
# 6. AGREGACIONES (GroupBy #20, #26, #32, #35, #36, #40, #53, #54, #61, #67, #70, #72, #84, #89, #92, #93, #100, #103, #107, #108, #115, #121, #124, #126)
# ============================================================================
def aggregate_by_team(df):
    """Agrega estadísticas por equipo"""
    print("📊 Agregando estadísticas por equipo...")
    
    team_stats = {}
    
    # Estadísticas cuando el equipo juega como Equipo1
    if 'Equipo1' in df.columns:
        agg_e1 = {'EQUIPO1_GOLES': ['mean', 'sum', 'std'], 'EQUIPO2_GOLES': 'mean'}
        if 'Disparos_a_puerta_E1' in df.columns:
            agg_e1['Disparos_a_puerta_E1'] = 'mean'
        if 'Pases_realizados_E1' in df.columns:
            agg_e1['Pases_realizados_E1'] = 'mean'
        stats_e1 = df.groupby('Equipo1').agg(agg_e1).round(2)
        team_stats['as_home'] = stats_e1

    # Estadísticas cuando el equipo juega como Equipo2
    if 'Equipo2' in df.columns:
        agg_e2 = {'EQUIPO2_GOLES': ['mean', 'sum', 'std'], 'EQUIPO1_GOLES': 'mean'}
        if 'Disparos_a_puerta_E2' in df.columns:
            agg_e2['Disparos_a_puerta_E2'] = 'mean'
        if 'Pases_realizados_E2' in df.columns:
            agg_e2['Pases_realizados_E2'] = 'mean'
        stats_e2 = df.groupby('Equipo2').agg(agg_e2).round(2)
        team_stats['as_away'] = stats_e2
    
    print(f"   ✓ Estadísticas de {len(team_stats)} grupos calculadas")
    
    return team_stats


# ============================================================================
# 7. JOINS (Joiner #42, #73, #95, #127)
# ============================================================================
def join_team_stats(df, team_stats):
    """Une estadísticas de equipos al dataset principal"""
    print("🔗 Realizando joins de estadísticas de equipos...")
    
    df_enriched = df.copy()
    
    # Agregar estadísticas por equipo (simplificado)
    if 'Equipo1' in df.columns:
        team_means = df.groupby('Equipo1')['EQUIPO1_GOLES'].transform('mean')
        df_enriched['Equipo1_PromedioGoles'] = team_means
    
    if 'Equipo2' in df.columns:
        team_means = df.groupby('Equipo2')['EQUIPO2_GOLES'].transform('mean')
        df_enriched['Equipo2_PromedioGoles'] = team_means
    
    print(f"   ✓ Dataset enriquecido con estadísticas de equipos")
    
    return df_enriched


# ============================================================================
# 8. PREPARAR DATOS PARA MODELADO
# ============================================================================
def prepare_for_modeling(df):
    """Prepara datos para modelos ML"""
    print("🤖 Preparando datos para modelado...")
    
    df_model = df.copy()
    
    # Codificar variables categóricas
    le_dict = {}
    categorical_cols = df_model.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col not in ['Equipo1', 'Equipo2']:  # Excluir variables de identificación
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le
    
    # También codificar Equipo1 y Equipo2
    if 'Equipo1' in df_model.columns:
        le_e1 = LabelEncoder()
        df_model['Equipo1_Encoded'] = le_e1.fit_transform(df_model['Equipo1'].astype(str))
        le_dict['Equipo1'] = le_e1
    
    if 'Equipo2' in df_model.columns:
        le_e2 = LabelEncoder()
        df_model['Equipo2_Encoded'] = le_e2.fit_transform(df_model['Equipo2'].astype(str))
        le_dict['Equipo2'] = le_e2
    
    # Rellenar NaNs restantes con 0
    df_model = df_model.fillna(0)
    
    print(f"   ✓ Datos preparados: {df_model.shape[1]} features")
    
    return df_model, le_dict


# ============================================================================
# 9. ENTRENAR MODELOS — CLASIFICACIÓN (Win / Draw / Loss)
# ============================================================================
def train_models(X_train, y_train, X_test, y_test):
    """Entrena 3 clasificadores para predecir resultado del partido"""
    print("🎓 Entrenando modelos de clasificación...")

    models = {}
    predictions = {}

    print("   - Entrenando Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    predictions['Random Forest'] = rf.predict(X_test)

    print("   - Entrenando Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    predictions['Gradient Boosting'] = gb.predict(X_test)

    print("   - Entrenando Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    predictions['Logistic Regression'] = lr.predict(X_test)

    # SVM necesita features escaladas
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("   - Entrenando SVM...")
    svm = SVC(kernel='rbf', C=1, probability=True, random_state=42)
    svm.fit(X_train_sc, y_train)
    models['SVM'] = (svm, scaler)
    predictions['SVM'] = svm.predict(X_test_sc)

    print("   - Entrenando XGBoost...")
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                         random_state=42, eval_metric='mlogloss', verbosity=0)
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    predictions['XGBoost'] = xgb.predict(X_test)

    print("   - Entrenando KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_sc, y_train)
    models['KNN'] = (knn, scaler)
    predictions['KNN'] = knn.predict(X_test_sc)

    print(f"   ✓ {len(models)} modelos entrenados")
    return models, predictions


# ============================================================================
# 10. EVALUAR MODELOS
# ============================================================================
def evaluate_models(models, predictions, y_test, class_labels=None):
    """Evalúa y compara clasificadores"""
    print("\n" + "="*60)
    print("RESULTADOS DE MODELOS  (Win / Draw / Loss)")
    print("="*60)

    results_rows = []

    for model_name, y_pred in predictions.items():
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{model_name}  —  Accuracy: {acc:.2%}")
        print(classification_report(y_test, y_pred, target_names=class_labels, zero_division=0))
        results_rows.append({'Model': model_name, 'Accuracy': round(acc, 4)})

    # Ranking final
    print("\nRANKING:")
    for r in sorted(results_rows, key=lambda x: x['Accuracy'], reverse=True):
        print(f"  {r['Model']:<22} {r['Accuracy']:.2%}")

    print("="*60)
    return pd.DataFrame(results_rows)


# ============================================================================
# 10B. CROSS-VALIDATION — accuracy estable con pocos datos
# ============================================================================
def cross_validate_models(X, y, n_splits=5, random_state=42):
    """Evalúa clasificadores con StratifiedKFold para obtener accuracy estable."""
    print("\n" + "="*60)
    print(f"CROSS-VALIDATION ({n_splits} folds)")
    print("="*60)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scaler = StandardScaler()

    classifiers = {
        'Random Forest':      RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        'Gradient Boosting':  GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        'Logistic Regression': Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=random_state))]),
        'SVM':                Pipeline([('sc', StandardScaler()), ('svm', SVC(kernel='rbf', C=1, random_state=random_state))]),
        'XGBoost':            XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=random_state, eval_metric='mlogloss', verbosity=0),
        'KNN':                Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))]),
    }

    rows = []
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        rows.append({'Model': name, 'CV Mean': scores.mean(), 'CV Std': scores.std()})
        print(f"  {name:<22}  {scores.mean():.2%} ± {scores.std():.2%}  (folds: {' '.join(f'{s:.0%}' for s in scores)})")

    print("\nRANKING (por CV Mean):")
    for r in sorted(rows, key=lambda x: x['CV Mean'], reverse=True):
        print(f"  {r['Model']:<22}  {r['CV Mean']:.2%} ± {r['CV Std']:.2%}")
    print("="*60)

    return pd.DataFrame(rows)


# ============================================================================
# 10C. ENTRENAR REGRESORES — predice goles de cada equipo
# ============================================================================
def train_regressors(X_train, y1_train, y2_train, X_test, y1_test, y2_test):
    """Entrena modelos para predecir goles de Equipo1 y Equipo2 por separado"""
    print("\n⚽ Entrenando regresores de goles...")

    regs = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost':       XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                      random_state=42, verbosity=0),
    }

    regressors = {}
    for name, reg in regs.items():
        reg.fit(X_train, y1_train)
        reg2 = type(reg)(**reg.get_params())
        reg2.fit(X_train, y2_train)

        mae1 = mean_absolute_error(y1_test, reg.predict(X_test))
        mae2 = mean_absolute_error(y2_test, reg2.predict(X_test))
        print(f"   {name:<16} MAE goles E1: {mae1:.2f}  |  MAE goles E2: {mae2:.2f}")

        regressors[name] = {'E1': reg, 'E2': reg2}

    return regressors


# ============================================================================
# FLUJO PRINCIPAL
# ============================================================================
def main(filepath, test_size=0.2, random_state=42):
    """
    Pipeline completo: carga, limpia, transforma y predice el resultado
    del partido (Win / Draw / Loss para Equipo1).

    Args:
        filepath:     Ruta al archivo Excel con datos de partidos
        test_size:    Proporción de datos de prueba (default 0.2 = 20 %)
        random_state: Semilla para reproducibilidad
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "="*60)
    print("⚽ PREDICCIÓN DE RESULTADO — CHAMPIONS LEAGUE")
    print("="*60 + "\n")

    # 1-7. Carga, limpieza, transformación y enriquecimiento
    df = load_data(filepath)
    df = select_columns(df)
    df = handle_missing_values(df, strategy='mean')
    df = create_derived_variables(df)
    df = filter_rows(df)
    team_stats = aggregate_by_team(df)
    df = join_team_stats(df, team_stats)

    # 8. Preparar para modelado
    df_model, le_dict = prepare_for_modeling(df)

    # 9. Separar features y target
    print("\n📌 Separando features y target...")

    if 'Resultado_E1' not in df_model.columns:
        print("⚠️  ERROR: columna 'Resultado_E1' no encontrada")
        return

    y = df_model['Resultado_E1']  # 0=Draw, 1=Loss, 2=Win  (orden LabelEncoder)
    class_labels = le_dict['Resultado_E1'].classes_  # ['Draw', 'Loss', 'Win']

    # Excluir identificadores, el target y columnas que revelan el resultado
    # (son estadísticas POST-partido directamente derivadas de los goles)
    exclude_cols = [
        'Partido_id', 'Fase', 'Equipo1', 'Equipo2',
        'EQUIPO1_GOLES', 'EQUIPO2_GOLES', 'Resultado_E1', 'Diferencia_Goles',
        'Goles_dentro_area_E1', 'Goles_Fuera_Area_E1',
        'Goles_dentro_area_E2', 'Goles_Fuera_Area_E2',
        'goles_encajados_E1', 'goles_encajados_E2',
        'Goles_encajados_propia_puerta_E1', 'Goles_encajados_propia_puerta_E2',
        'Porterias_a_cero_E1', 'Porterias_a_cero_E2',
        'Eficiencia_Tiros_E1', 'Eficiencia_Tiros_E2',
    ]
    X = df_model.drop(columns=[c for c in exclude_cols if c in df_model.columns])

    print(f"   ✓ Features: {X.shape[1]} | Partidos: {y.shape[0]}")
    print(f"   Distribución clases: {dict(zip(class_labels, [int((y==i).sum()) for i in range(len(class_labels))]))}")

    # 10. Train-test split (estratificado para respetar proporción de clases)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n🔀 Split: {len(X_train)} train — {len(X_test)} test")

    # 11. Cross-validation antes de entrenar el modelo final
    cv_results = cross_validate_models(X, y)

    # 12. Entrenar clasificadores (resultado) y regresores (goles)
    models, predictions = train_models(X_train, y_train, X_test, y_test)

    y1_train = df_model.loc[X_train.index, 'EQUIPO1_GOLES']
    y2_train = df_model.loc[X_train.index, 'EQUIPO2_GOLES']
    y1_test  = df_model.loc[X_test.index,  'EQUIPO1_GOLES']
    y2_test  = df_model.loc[X_test.index,  'EQUIPO2_GOLES']
    regressors = train_regressors(X_train, y1_train, y2_train, X_test, y1_test, y2_test)

    # 13. Evaluar modelos (split único)
    results_df = evaluate_models(models, predictions, y_test, class_labels)

    # 13. Guardar resultados junto a los partidos reales
    idx_test = X_test.index
    predictions_df = df.loc[idx_test, ['Equipo1', 'Equipo2']].copy()
    predictions_df['Resultado_Real'] = df.loc[idx_test, 'Resultado_E1'].values
    for model_name, y_pred in predictions.items():
        predictions_df[model_name] = le_dict['Resultado_E1'].inverse_transform(y_pred)

    output_dir = r"C:\Users\fehgb\OneDrive\Desktop\prediccion futybol"
    results_df.to_csv(rf"{output_dir}\model_results.csv", index=False)
    predictions_df.to_csv(rf"{output_dir}\predictions.csv", index=False)
    df.to_csv(rf"{output_dir}\processed_data.csv", index=False)

    print(f"\n✅ Archivos guardados en: {output_dir}")
    print("\n📋 Predicciones vs resultado real:")
    print(predictions_df.to_string(index=False))

    print("\n" + "="*60)
    print("✨ PIPELINE COMPLETADO")
    print("="*60 + "\n")

    return {
        'df': df,
        'df_model': df_model,
        'models': models,
        'regressors': regressors,
        'le_dict': le_dict,
        'feature_cols': list(X.columns),
        'results': results_df,
        'predictions': predictions_df,
        'cv_results': cv_results,
    }


# ============================================================================
# PREDICCIÓN DE PARTIDO FUTURO  (ensemble de n_runs corridas)
# ============================================================================
def predecir_partido(equipo1, equipo2, results, n_runs=20):
    """
    Predice el resultado entrenando cada modelo n_runs veces con distintas
    semillas sobre el dataset completo y promediando las probabilidades.
    """
    df        = results['df']
    df_model  = results['df_model']
    le_dict   = results['le_dict']
    feat_cols = results['feature_cols']

    # --- Aviso si el partido ya existe en el historial ---
    partidos_previos = df[(df['Equipo1'] == equipo1) & (df['Equipo2'] == equipo2)]
    if not partidos_previos.empty:
        for _, p in partidos_previos.iterrows():
            g1 = int(p['EQUIPO1_GOLES'])
            g2 = int(p['EQUIPO2_GOLES'])
            fecha = p.get('Fecha', '')
            fecha_str = f" ({fecha})" if pd.notna(fecha) and fecha else ""
            print(f"⚠️  Este partido ya está en el dataset{fecha_str}: {equipo1} {g1}–{g2} {equipo2}  →  la predicción puede estar sesgada")

    # --- Perfiles: promedio de cada stat numérica por equipo ---
    e1_stat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c.endswith('_E1')]
    e2_stat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c.endswith('_E2')]
    perfil_e1 = df.groupby('Equipo1')[e1_stat_cols].mean()
    perfil_e2 = df.groupby('Equipo2')[e2_stat_cols].mean()

    global_e1 = df[e1_stat_cols].mean()
    global_e2 = df[e2_stat_cols].mean()

    if equipo1 in perfil_e1.index:
        stats_e1 = perfil_e1.loc[equipo1]
    elif equipo1 in perfil_e2.index:
        stats_e1 = perfil_e2.loc[equipo1].rename(lambda c: c.replace('_E2', '_E1'))
    else:
        stats_e1 = global_e1

    if equipo2 in perfil_e2.index:
        stats_e2 = perfil_e2.loc[equipo2]
    elif equipo2 in perfil_e1.index:
        stats_e2 = perfil_e1.loc[equipo2].rename(lambda c: c.replace('_E1', '_E2'))
    else:
        stats_e2 = global_e2

    # --- Construir fila de features ---
    row = {'Es_Local_E1': 1}
    for col in feat_cols:
        if col in stats_e1.index:
            row[col] = stats_e1[col]
        elif col in stats_e2.index:
            row[col] = stats_e2[col]
        elif col == 'Equipo1_Encoded':
            classes = le_dict['Equipo1'].classes_
            row[col] = int(le_dict['Equipo1'].transform([equipo1])[0]) if equipo1 in classes else 0
        elif col == 'Equipo2_Encoded':
            classes = le_dict['Equipo2'].classes_
            row[col] = int(le_dict['Equipo2'].transform([equipo2])[0]) if equipo2 in classes else 0
        elif col == 'Equipo1_PromedioGoles':
            mask = df['Equipo1'] == equipo1
            row[col] = df.loc[mask, 'EQUIPO1_GOLES'].mean() if mask.any() else df['EQUIPO1_GOLES'].mean()
        elif col == 'Equipo2_PromedioGoles':
            mask = df['Equipo2'] == equipo2
            row[col] = df.loc[mask, 'EQUIPO2_GOLES'].mean() if mask.any() else df['EQUIPO2_GOLES'].mean()
        else:
            row[col] = df_model[col].mean()

    X_pred = pd.DataFrame([row])[feat_cols]

    # --- Dataset completo para entrenar (sin split) ---
    X_full  = df_model[feat_cols]
    y_full  = df_model['Resultado_E1']
    y1_full = df_model['EQUIPO1_GOLES']
    y2_full = df_model['EQUIPO2_GOLES']
    le_res  = le_dict['Resultado_E1']
    classes = list(le_res.classes_)   # ['Draw', 'Loss', 'Win']

    def make_clfs(seed):
        return {
            'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
            'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=seed),
            'Logistic Regression': Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=seed))]),
            'SVM':                 Pipeline([('sc', StandardScaler()), ('svm', SVC(kernel='rbf', C=1, probability=True, random_state=seed))]),
            'XGBoost':             XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                                 random_state=seed, eval_metric='mlogloss', verbosity=0),
            'KNN':                 Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))]),
        }

    # Ordenar modelos por CV Mean (mejor primero)
    model_names = list(make_clfs(0).keys())
    cv_df = results.get('cv_results')
    if cv_df is not None:
        cv_order = cv_df.set_index('Model')['CV Mean'].to_dict()
        model_names = sorted(model_names, key=lambda n: cv_order.get(n, 0), reverse=True)

    probas_acum = {name: [] for name in model_names}

    print(f"\n{'='*58}")
    print(f"🔮  {equipo1}  vs  {equipo2}  ({n_runs} corridas por modelo)")
    print(f"{'='*58}")
    print("   Entrenando ensemble...", end='', flush=True)

    for seed in range(n_runs):
        for name, clf in make_clfs(seed).items():
            clf.fit(X_full, y_full)
            probas_acum[name].append(clf.predict_proba(X_pred)[0])

    print(" listo.")

    # --- Resultados clasificadores ---
    win_idx  = classes.index('Win')  if 'Win'  in classes else None
    draw_idx = classes.index('Draw') if 'Draw' in classes else None
    loss_idx = classes.index('Loss') if 'Loss' in classes else None

    print(f"\n  {'Modelo':<22}  Pred     Win    Draw    Loss")
    print(f"  {'-'*55}")
    for name in model_names:
        avg  = np.mean(probas_acum[name], axis=0)
        pred = classes[np.argmax(avg)]
        w = avg[win_idx]  if win_idx  is not None else 0
        d = avg[draw_idx] if draw_idx is not None else 0
        l = avg[loss_idx] if loss_idx is not None else 0
        print(f"  {name:<22} → {pred:<6}  {w:>5.1%}  {d:>5.1%}  {l:>5.1%}")

    # --- Promedio global entre todos los modelos ---
    todas = np.mean([np.mean(probas_acum[n], axis=0) for n in model_names], axis=0)
    pred_global = classes[np.argmax(todas)]
    w = todas[win_idx]  if win_idx  is not None else 0
    d = todas[draw_idx] if draw_idx is not None else 0
    l = todas[loss_idx] if loss_idx is not None else 0
    print(f"  {'-'*55}")
    print(f"  {'CONSENSO':<22} → {pred_global:<6}  {w:>5.1%}  {d:>5.1%}  {l:>5.1%}")

    # --- Regresores de goles (ensemble) ---
    def make_regs(seed):
        return {
            'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
                              RandomForestRegressor(n_estimators=100, random_state=seed+1000, n_jobs=-1)),
            'XGBoost':       (XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                           random_state=seed, verbosity=0),
                              XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                           random_state=seed+1000, verbosity=0)),
        }

    print(f"\n  {'Modelo':<16}  Marcador predicho")
    print(f"  {'-'*42}")
    for reg_name in make_regs(0).keys():
        g1_preds, g2_preds = [], []
        for seed in range(n_runs):
            r1, r2 = make_regs(seed)[reg_name]
            r1.fit(X_full, y1_full)
            r2.fit(X_full, y2_full)
            g1_preds.append(r1.predict(X_pred)[0])
            g2_preds.append(r2.predict(X_pred)[0])
        g1 = max(0, round(np.mean(g1_preds)))
        g2 = max(0, round(np.mean(g2_preds)))
        print(f"  {reg_name:<16}  {equipo1} {g1} – {g2} {equipo2}  (±{np.std(g1_preds):.1f} / ±{np.std(g2_preds):.1f})")

    print()
    equipos_conocidos = sorted(set(df['Equipo1'].tolist() + df['Equipo2'].tolist()))
    for e in [equipo1, equipo2]:
        if e not in equipos_conocidos:
            print(f"  ⚠️  '{e}' no está en el historial — se usaron promedios globales")


# ============================================================================
# EJECUTAR
# ============================================================================
if __name__ == "__main__":
    FILEPATH = r"C:\Users\fehgb\OneDrive\Desktop\prediccion futybol\creando_dataset_modificado.xlsx"

    try:
        results = main(FILEPATH)

        # ── Equipos disponibles en el historial ──────────────────────────
        equipos = sorted(set(
            results['df']['Equipo1'].tolist() + results['df']['Equipo2'].tolist()
        ))
        print("⚽ Equipos en el historial:")
        print("   " + " | ".join(equipos))

        # ── Predicciones de partidos futuros ─────────────────────────────
        predecir_partido("Paris",  "Bayern Munchen",       results)
        predecir_partido("Atleti",    "Union SG", results)
        predecir_partido("Juventus",    "Sporting CP",          results)

    except FileNotFoundError:
        print(f"❌ No se encontró el archivo: {FILEPATH}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
