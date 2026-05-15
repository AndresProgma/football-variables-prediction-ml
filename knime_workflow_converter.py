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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, f1_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, XGBRegressor

K_FEATURES = 20  # top features según ANOVA-F dentro de cada fold (evita curse of dimensionality)

# Configuración v14 (mejor ensemble en walk-forward CV con 189 partidos):
#   - features extra (xG sintético + UEFA coef) ya integrados en compute_*
#   - clase Draw recibe peso 1.5× → modelos pueden predecir empates sin colapsar
#   - todos los clasificadores envueltos en CalibratedClassifierCV (isotónica, cv=3)
#     → probabilidades calibradas: P(Win)=70% ≈ frecuencia real de aciertos 70%
#   - K=20 features (de ~160) → óptimo en CV walk-forward, evita curse-of-dim
CALIBRAR_DEFECTO = True
DRAW_WEIGHT_BOOST = 1.5   # peso de la clase Draw


def _class_weight():
    """Pesos de clase para LabelEncoder (0=Draw, 1=Loss, 2=Win — orden alfabético)."""
    return {0: DRAW_WEIGHT_BOOST, 1: 1.0, 2: 1.0}


def _wrap_calibrado(pipe, calibrar=None):
    if calibrar is None:
        calibrar = CALIBRAR_DEFECTO
    if calibrar:
        return CalibratedClassifierCV(pipe, method='isotonic', cv=3)
    return pipe


def build_classifiers(seed=42, n_features=None, calibrar=None):
    """Pipelines con SelectKBest + clasificador + (opcional) calibración isotónica.

    Cambios v4 vs v0:
      - class_weight={0:2, 1:1, 2:1}  (Draw boost para no colapsar empates)
      - CalibratedClassifierCV(isotónica, cv=3) envuelve cada clasificador
    """
    k = min(K_FEATURES, n_features) if n_features else K_FEATURES
    sk = lambda: SelectKBest(f_classif, k=k)
    cw = _class_weight()
    return {
        'Random Forest':       _wrap_calibrado(Pipeline([('sk', sk()),
                                         ('rf', RandomForestClassifier(
                                             n_estimators=200, max_depth=5, min_samples_leaf=3,
                                             class_weight=cw,
                                             random_state=seed, n_jobs=-1))]), calibrar),
        'Gradient Boosting':   _wrap_calibrado(Pipeline([('sk', sk()),
                                         ('gb', GradientBoostingClassifier(
                                             n_estimators=100, max_depth=3, min_samples_leaf=3,
                                             random_state=seed))]), calibrar),
        'Logistic Regression': _wrap_calibrado(Pipeline([('sc', StandardScaler()), ('sk', sk()),
                                         ('lr', LogisticRegression(
                                             max_iter=2000, C=0.3, class_weight=cw,
                                             random_state=seed))]), calibrar),
        'SVM':                 _wrap_calibrado(Pipeline([('sc', StandardScaler()), ('sk', sk()),
                                         ('svm', SVC(
                                             kernel='rbf', C=0.5, class_weight=cw,
                                             probability=True, random_state=seed))]), calibrar),
        'XGBoost':             _wrap_calibrado(Pipeline([('sk', sk()),
                                         ('xgb', XGBClassifier(
                                             n_estimators=100, max_depth=3, learning_rate=0.1,
                                             reg_lambda=1.0, random_state=seed,
                                             eval_metric='mlogloss', verbosity=0))]), calibrar),
        'KNN':                 _wrap_calibrado(Pipeline([('sc', StandardScaler()), ('sk', sk()),
                                         ('knn', KNeighborsClassifier(
                                             n_neighbors=5, weights='distance'))]), calibrar),
    }


ELO_BASE = 1500
ELO_K = 30           # qué tan rápido reacciona (estándar football: 20-32)
ELO_HOME_ADV = 60    # ventaja de local en puntos ELO (ClubElo usa ~60)


def compute_elo_features(df):
    """
    Calcula ELO incremental recorriendo partidos en orden cronológico.
    Para cada fila agrega ELO_E1, ELO_E2 y Diff_ELO con el rating ANTES
    del partido (sin leakage). Luego actualiza los ratings con el resultado.

    Bonus por margen de goles: 1x si ≤1 gol, 1.5x si 2, 1.75x+0.1*extra si ≥3.
    Devuelve también el dict de ELO finales para usar en predicciones futuras.
    """
    print("📈 Calculando ELO incremental...")
    df = df.copy()
    elos = {}
    elo_e1_list, elo_e2_list = [], []

    for _, row in df.iterrows():
        e1, e2 = row['Equipo1'], row['Equipo2']
        elo_e1 = elos.get(e1, ELO_BASE)
        elo_e2 = elos.get(e2, ELO_BASE)
        elo_e1_list.append(elo_e1)
        elo_e2_list.append(elo_e2)

        g1, g2 = row.get('EQUIPO1_GOLES'), row.get('EQUIPO2_GOLES')
        if pd.isna(g1) or pd.isna(g2):
            continue  # partido sin jugar todavía → no actualiza

        es_local = int(row.get('Es_Local_E1', 1)) if pd.notna(row.get('Es_Local_E1')) else 1
        adv = ELO_HOME_ADV * es_local

        # Probabilidad esperada de victoria de E1
        expected_e1 = 1 / (1 + 10 ** ((elo_e2 - elo_e1 - adv) / 400))

        # Resultado real (1 / 0.5 / 0)
        s1 = 1.0 if g1 > g2 else (0.0 if g1 < g2 else 0.5)

        # Multiplicador por margen de goles
        gd = abs(g1 - g2)
        margin = 1.0 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) * 0.1)

        delta = ELO_K * margin * (s1 - expected_e1)
        elos[e1] = elo_e1 + delta
        elos[e2] = elo_e2 - delta

    df['ELO_E1'] = elo_e1_list
    df['ELO_E2'] = elo_e2_list
    df['Diff_ELO'] = df['ELO_E1'] - df['ELO_E2']

    top5 = sorted(elos.items(), key=lambda x: -x[1])[:5]
    bot5 = sorted(elos.items(), key=lambda x: x[1])[:5]
    print(f"   ✓ ELO calculado para {len(elos)} equipos")
    print(f"   Top 5: " + ' · '.join(f'{t} {int(e)}' for t, e in top5))
    print(f"   Bot 5: " + ' · '.join(f'{t} {int(e)}' for t, e in bot5))

    return df, elos


# ============================================================================
# Coeficiente UEFA aproximado por club (5-year, ~temporada 2024-25)
# ============================================================================

UEFA_COEF = {
    'Real Madrid': 144.0, 'Man City': 145.0, 'Bayern Munchen': 138.0,
    'Liverpool': 132.0, 'Paris': 122.5, 'Inter': 119.5,
    'BDortmund': 109.0, 'Leverkursen': 107.0, 'Atleti': 107.0,
    'Juventus': 100.5, 'Napoli': 100.0, 'Barcelona': 99.5,
    'Arsenal': 98.0, 'Benfica': 94.0, 'Atlanta': 90.0,
    'Chelsea': 88.0, 'Milan': 88.0, 'Sporting CP': 84.0,
    'Tottenham': 81.0, 'PSV': 71.0, 'Villareal': 70.0,
    'Marseille': 65.0, 'Ajax': 64.0, 'Athletic Club': 53.0,
    'Frankfurt': 52.0, 'Slavia Praha': 51.5, 'Club Brugge': 51.5,
    'Galatasaray': 51.0, 'Newcastle': 50.0, 'Olympiacos': 47.5,
    'Bodo': 49.0, 'Copengagen': 49.0, 'Qarabag': 41.0,
    'Union SG': 41.0, 'Monaco': 30.0, 'Pafos': 12.0,
    'Kairat Almaty': 8.0,
    # 2024-25 únicos (no en 2025-26)
    'Leipzig': 87.0, 'Stuttgart': 50.0, 'Sturm Graz': 35.0,
    'Salzburg': 55.0, 'Brest': 25.0, 'Bologna': 35.0,
    'Aston Villa': 50.0, 'Lille': 60.0, 'Crvena Zvezda': 40.0,
    'YB': 35.0, 'Celtic': 50.0, 'Feyenoord': 75.0,
    'Dinamo Zagreb': 50.0, 'Sparta': 40.0, 'Girona': 30.0,
    'Slovan Bratislava': 20.0, 'Shakhtar': 70.0,
    # 2023-24 grupos
    'Sevilla': 92.0, 'Lazio': 70.0, 'Lens': 25.0,
    'Antwerp': 25.0, 'Galatasaray SK': 51.0, 'Manchester United': 95.0,
    'Man United': 95.0, 'RBL': 87.0, 'Young Boys': 35.0,
    'Braga': 65.0, 'Real Sociedad': 65.0, 'Union Berlin': 45.0,
    'Berlin': 45.0, 'Royal Antwerp': 25.0, 'Real Madrid CF': 144.0,
    'Real Betis': 50.0,
}

UEFA_COEF_DEFAULT = 25.0   # equipo desconocido → underdog plausible


def compute_uefa_coef_features(df):
    """Anexa Coef_UEFA_E1, Coef_UEFA_E2 y Diff_Coef_UEFA.

    Es una señal estática del 'nivel histórico europeo' del club. Útil
    sobre todo para cold-start (Pafos, Kairat, Qarabag) donde el ELO está
    en su default de 1500 al no tener historia en el dataset."""
    print("🏆 Calculando coeficiente UEFA por club...")
    df = df.copy()
    df['Coef_UEFA_E1']  = df['Equipo1'].map(UEFA_COEF).fillna(UEFA_COEF_DEFAULT)
    df['Coef_UEFA_E2']  = df['Equipo2'].map(UEFA_COEF).fillna(UEFA_COEF_DEFAULT)
    df['Diff_Coef_UEFA'] = df['Coef_UEFA_E1'] - df['Coef_UEFA_E2']

    desconocidos = set(df.loc[~df['Equipo1'].isin(UEFA_COEF), 'Equipo1']) | \
                   set(df.loc[~df['Equipo2'].isin(UEFA_COEF), 'Equipo2'])
    if desconocidos:
        print(f"   ⚠️  {len(desconocidos)} equipos sin coeficiente UEFA, usando default {UEFA_COEF_DEFAULT}: {sorted(desconocidos)[:10]}")
    return df


# ============================================================================
# xG sintético pre-partido (promedio histórico del equipo en partidos previos)
# ============================================================================

# Pesos calibrados aproximadamente para que xG promedio ≈ goles promedio
XG_W_SHOTS = 0.05   # disparos totales
XG_W_SOT   = 0.20   # disparos a puerta
XG_W_CC    = 0.55   # oportunidades claras
XG_WINDOW  = 5      # promedio de últimos N partidos (más receptivo a forma actual)


def _xg_partido(shots, sot, clear):
    """xG estimado desde stats del propio partido."""
    return (XG_W_SHOTS * (shots or 0)
            + XG_W_SOT * (sot or 0)
            + XG_W_CC  * (clear or 0))


def compute_xg_features(df):
    """xG / xGA sintético pre-partido: promedio rolling de los últimos
    XG_WINDOW partidos del equipo (como E1 o E2). Sin leakage.

    Anexa: xG_E1_rolling, xGA_E1_rolling, xG_E2_rolling, xGA_E2_rolling,
            Diff_xG_rolling. Si el equipo no tiene historial → NaN
            (handle_missing_values lo imputará luego con la media)."""
    print("🎯 Calculando xG sintético pre-partido...")
    df = df.copy()
    historial = {}   # equipo → list of {xg, xga}

    xg1, xga1, xg2, xga2 = [], [], [], []

    for _, row in df.iterrows():
        e1, e2 = row['Equipo1'], row['Equipo2']

        # Rolling xG/xGA del equipo desde su historial previo
        h1 = (historial.get(e1) or [])[-XG_WINDOW:]
        h2 = (historial.get(e2) or [])[-XG_WINDOW:]
        xg1.append(np.mean([h['xg']  for h in h1]) if h1 else np.nan)
        xga1.append(np.mean([h['xga'] for h in h1]) if h1 else np.nan)
        xg2.append(np.mean([h['xg']  for h in h2]) if h2 else np.nan)
        xga2.append(np.mean([h['xga'] for h in h2]) if h2 else np.nan)

        # Calcular xG de ESTE partido y agregar al historial (sin leakage:
        # las features rolling ya fueron asignadas antes, esto es para los
        # partidos futuros).
        m_xg1 = _xg_partido(row.get('Disparos_totales_E1'),
                            row.get('Disparos_a_puerta_E1'),
                            row.get('Oportunidades_claras_E1'))
        m_xg2 = _xg_partido(row.get('Disparos_totales_E2'),
                            row.get('Disparos_a_puerta_E2'),
                            row.get('Oportunidades_claras_E2'))
        if not pd.isna(m_xg1):
            historial.setdefault(e1, []).append({'xg': m_xg1, 'xga': m_xg2})
        if not pd.isna(m_xg2):
            historial.setdefault(e2, []).append({'xg': m_xg2, 'xga': m_xg1})

    df['xG_E1_rolling']  = xg1
    df['xGA_E1_rolling'] = xga1
    df['xG_E2_rolling']  = xg2
    df['xGA_E2_rolling'] = xga2
    df['Diff_xG_rolling'] = df['xG_E1_rolling'].fillna(0) - df['xG_E2_rolling'].fillna(0)

    print(f"   ✓ xG rolling calculado (ventana={XG_WINDOW}) para {len(historial)} equipos")
    return df, historial


# ============================================================================
# Features pre-partido avanzadas: forma reciente, descanso, head-to-head
# ============================================================================

FORMA_WINDOW = 5      # últimos N partidos
H2H_WINDOW = 3        # últimos N enfrentamientos directos


def compute_form_features(df):
    """
    Para cada partido calcula la forma reciente de ambos equipos basada en
    los últimos FORMA_WINDOW partidos (jugados como local O visitante).
    Usa SOLO partidos anteriores (sin leakage).

    Agrega columnas: Forma_W_E1, Forma_D_E1, Forma_L_E1, Forma_GF_E1,
    Forma_GC_E1, Forma_Pts_E1 (+ equivalentes _E2),
    Diff_Forma_Pts, Diff_Forma_GD, Dias_Descanso_E1/_E2.

    Devuelve también un dict por equipo con su historial completo, para que
    predecir_partido() pueda reconstruir las features de partidos futuros.
    """
    print("📊 Calculando forma reciente y descanso...")
    df = df.copy()
    if 'Fecha' in df.columns:
        df['_fecha_dt'] = pd.to_datetime(df['Fecha'], errors='coerce')
    else:
        df['_fecha_dt'] = pd.NaT

    # Historial por equipo: lista de dicts con {fecha, gf, gc, resultado}
    historial = {}

    cols = ['Forma_W_E1', 'Forma_D_E1', 'Forma_L_E1', 'Forma_GF_E1', 'Forma_GC_E1', 'Forma_Pts_E1',
            'Forma_W_E2', 'Forma_D_E2', 'Forma_L_E2', 'Forma_GF_E2', 'Forma_GC_E2', 'Forma_Pts_E2',
            'Dias_Descanso_E1', 'Dias_Descanso_E2',
            'Diff_Forma_Pts', 'Diff_Forma_GD']
    rows = {c: [] for c in cols}

    def resumen(hist):
        last = hist[-FORMA_WINDOW:] if hist else []
        w = sum(1 for h in last if h['res'] == 'W')
        d = sum(1 for h in last if h['res'] == 'D')
        l = sum(1 for h in last if h['res'] == 'L')
        gf = sum(h['gf'] for h in last) if last else 0
        gc = sum(h['gc'] for h in last) if last else 0
        pts = w * 3 + d
        return w, d, l, gf, gc, pts

    for _, row in df.iterrows():
        e1, e2 = row['Equipo1'], row['Equipo2']
        fecha = row['_fecha_dt']

        # Forma actual (antes del partido)
        w1, d1, l1, gf1, gc1, pts1 = resumen(historial.get(e1, []))
        w2, d2, l2, gf2, gc2, pts2 = resumen(historial.get(e2, []))
        rows['Forma_W_E1'].append(w1); rows['Forma_D_E1'].append(d1); rows['Forma_L_E1'].append(l1)
        rows['Forma_GF_E1'].append(gf1); rows['Forma_GC_E1'].append(gc1); rows['Forma_Pts_E1'].append(pts1)
        rows['Forma_W_E2'].append(w2); rows['Forma_D_E2'].append(d2); rows['Forma_L_E2'].append(l2)
        rows['Forma_GF_E2'].append(gf2); rows['Forma_GC_E2'].append(gc2); rows['Forma_Pts_E2'].append(pts2)
        rows['Diff_Forma_Pts'].append(pts1 - pts2)
        rows['Diff_Forma_GD'].append((gf1 - gc1) - (gf2 - gc2))

        # Días de descanso (desde último partido)
        for equipo, key in [(e1, 'Dias_Descanso_E1'), (e2, 'Dias_Descanso_E2')]:
            h = historial.get(equipo, [])
            if h and pd.notna(fecha) and h[-1]['fecha'] is not pd.NaT and pd.notna(h[-1]['fecha']):
                dias = (fecha - h[-1]['fecha']).days
                rows[key].append(max(0, int(dias)))
            else:
                rows[key].append(7)  # default razonable (semana)

        # Actualizar historial con el resultado real
        g1, g2 = row.get('EQUIPO1_GOLES'), row.get('EQUIPO2_GOLES')
        if pd.notna(g1) and pd.notna(g2):
            res1 = 'W' if g1 > g2 else ('L' if g1 < g2 else 'D')
            res2 = 'W' if g2 > g1 else ('L' if g2 < g1 else 'D')
            historial.setdefault(e1, []).append({'fecha': fecha, 'gf': int(g1), 'gc': int(g2), 'res': res1})
            historial.setdefault(e2, []).append({'fecha': fecha, 'gf': int(g2), 'gc': int(g1), 'res': res2})

    for c, vals in rows.items():
        df[c] = vals
    df = df.drop(columns=['_fecha_dt'])

    print(f"   ✓ Forma calculada para {len(historial)} equipos (ventana={FORMA_WINDOW})")
    return df, historial


def compute_h2h_features(df):
    """
    Para cada partido cuenta los últimos H2H_WINDOW enfrentamientos directos
    entre los dos equipos (en cualquier orden de local/visitante). Sin leakage.

    Agrega: H2H_W_E1, H2H_D, H2H_L_E1, H2H_GF_E1, H2H_GC_E1 (todos sobre los
    últimos H2H_WINDOW partidos entre ambos).

    Devuelve dict {(e1,e2): [partidos]} ordenado para reconstruir en predicción.
    """
    print("🤝 Calculando head-to-head...")
    df = df.copy()
    h2h_log = {}   # key: frozenset({e1,e2}) → lista de partidos en orden

    cols = ['H2H_W_E1', 'H2H_D', 'H2H_L_E1', 'H2H_GF_E1', 'H2H_GC_E1', 'H2H_N']
    rows = {c: [] for c in cols}

    for _, row in df.iterrows():
        e1, e2 = row['Equipo1'], row['Equipo2']
        key = frozenset({e1, e2})
        previos = h2h_log.get(key, [])[-H2H_WINDOW:]
        w, d, l, gf, gc = 0, 0, 0, 0, 0
        for p in previos:
            # Resultado desde la perspectiva de E1
            if p['equipo1'] == e1:
                g_propios, g_rival = p['g1'], p['g2']
            else:
                g_propios, g_rival = p['g2'], p['g1']
            gf += g_propios
            gc += g_rival
            if g_propios > g_rival: w += 1
            elif g_propios < g_rival: l += 1
            else: d += 1
        rows['H2H_W_E1'].append(w); rows['H2H_D'].append(d); rows['H2H_L_E1'].append(l)
        rows['H2H_GF_E1'].append(gf); rows['H2H_GC_E1'].append(gc); rows['H2H_N'].append(len(previos))

        # Registrar este partido para futuros H2H
        g1, g2 = row.get('EQUIPO1_GOLES'), row.get('EQUIPO2_GOLES')
        if pd.notna(g1) and pd.notna(g2):
            h2h_log.setdefault(key, []).append({
                'equipo1': e1, 'equipo2': e2,
                'g1': int(g1), 'g2': int(g2),
            })

    for c, vals in rows.items():
        df[c] = vals
    print(f"   ✓ H2H calculado para {len(h2h_log)} pares de equipos (ventana={H2H_WINDOW})")
    return df, h2h_log


def resumen_forma(hist_equipo, window=FORMA_WINDOW):
    """Helper para predecir_partido: resume el historial de un equipo."""
    last = (hist_equipo or [])[-window:]
    w = sum(1 for h in last if h['res'] == 'W')
    d = sum(1 for h in last if h['res'] == 'D')
    l = sum(1 for h in last if h['res'] == 'L')
    gf = sum(h['gf'] for h in last) if last else 0
    gc = sum(h['gc'] for h in last) if last else 0
    return {'w': w, 'd': d, 'l': l, 'gf': gf, 'gc': gc, 'pts': w * 3 + d}


def resumen_h2h(h2h_log, e1, e2, window=H2H_WINDOW):
    """Helper para predecir_partido: resume el head-to-head entre dos equipos."""
    key = frozenset({e1, e2})
    previos = (h2h_log.get(key) or [])[-window:]
    w, d, l, gf, gc = 0, 0, 0, 0, 0
    for p in previos:
        if p['equipo1'] == e1:
            g_propios, g_rival = p['g1'], p['g2']
        else:
            g_propios, g_rival = p['g2'], p['g1']
        gf += g_propios; gc += g_rival
        if g_propios > g_rival: w += 1
        elif g_propios < g_rival: l += 1
        else: d += 1
    return {'w': w, 'd': d, 'l': l, 'gf': gf, 'gc': gc, 'n': len(previos)}


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
        # Coeficiente UEFA estático (compute_uefa_coef_features)
        'Coef_UEFA_E1', 'Coef_UEFA_E2', 'Diff_Coef_UEFA',
        # ELO pre-partido (calculado en compute_elo_features)
        'ELO_E1', 'ELO_E2', 'Diff_ELO',
        # Forma últimos 5 partidos (calculado en compute_form_features)
        'Forma_W_E1', 'Forma_D_E1', 'Forma_L_E1', 'Forma_GF_E1', 'Forma_GC_E1', 'Forma_Pts_E1',
        'Forma_W_E2', 'Forma_D_E2', 'Forma_L_E2', 'Forma_GF_E2', 'Forma_GC_E2', 'Forma_Pts_E2',
        'Diff_Forma_Pts', 'Diff_Forma_GD',
        'Dias_Descanso_E1', 'Dias_Descanso_E2',
        # Head-to-head últimos 3 enfrentamientos (compute_h2h_features)
        'H2H_W_E1', 'H2H_D', 'H2H_L_E1', 'H2H_GF_E1', 'H2H_GC_E1', 'H2H_N',
        # xG sintético rolling (compute_xg_features)
        'xG_E1_rolling', 'xGA_E1_rolling', 'xG_E2_rolling', 'xGA_E2_rolling', 'Diff_xG_rolling',
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
    """Entrena los clasificadores balanceados sobre el split cronológico."""
    print("🎓 Entrenando modelos de clasificación...")

    models = build_classifiers(seed=42, n_features=X_train.shape[1])
    predictions = {}
    for name, clf in models.items():
        print(f"   - Entrenando {name}...")
        clf.fit(X_train, y_train)
        predictions[name] = clf.predict(X_test)

    print(f"   ✓ {len(models)} modelos entrenados")
    return models, predictions


# ============================================================================
# 10. EVALUAR MODELOS
# ============================================================================
def evaluate_models(models, predictions, y_test, class_labels=None):
    """Evalúa y compara clasificadores (accuracy + macro-F1)."""
    print("\n" + "="*60)
    print("RESULTADOS DE MODELOS  (Win / Draw / Loss)")
    print("="*60)

    results_rows = []

    for model_name, y_pred in predictions.items():
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
        print(f"\n{model_name}  —  Accuracy: {acc:.2%}  |  Macro-F1: {f1m:.2%}")
        print(classification_report(y_test, y_pred, target_names=class_labels, zero_division=0))
        results_rows.append({'Model': model_name, 'Accuracy': round(acc, 4), 'Macro_F1': round(f1m, 4)})

    print("\nRANKING (por Macro-F1, mejor para clases desbalanceadas):")
    for r in sorted(results_rows, key=lambda x: x['Macro_F1'], reverse=True):
        print(f"  {r['Model']:<22}  acc {r['Accuracy']:.2%}   F1 {r['Macro_F1']:.2%}")

    print("="*60)
    return pd.DataFrame(results_rows)


# ============================================================================
# 10B. CROSS-VALIDATION — accuracy estable con pocos datos
# ============================================================================
def cross_validate_models(X, y, n_splits=3, random_state=42):
    """
    Walk-forward validation: respeta el orden cronológico del dataset.
    Cada fold entrena solo con partidos ANTERIORES a los del test.
    n_splits=3 → test folds de ~20 partidos (con 81 totales) — menos ruido de muestreo
    que n_splits=5 (que daba folds de ~13 y varianza inflada artificialmente).
    Requisito: X e y deben venir ordenados por Fecha ascendente.
    """
    n = len(X)
    test_size = n // (n_splits + 1)
    print("\n" + "="*60)
    print(f"VALIDACIÓN TEMPORAL WALK-FORWARD ({n_splits} folds, ~{test_size} partidos por test)")
    print("="*60)

    cv = TimeSeriesSplit(n_splits=n_splits)
    classifiers = build_classifiers(seed=random_state, n_features=X.shape[1])

    rows = []
    for name, clf in classifiers.items():
        out = cross_validate(
            clf, X, y, cv=cv,
            scoring={'acc': 'accuracy', 'f1m': 'f1_macro'},
            n_jobs=1,
        )
        acc_m, acc_s = out['test_acc'].mean(), out['test_acc'].std()
        f1_m, f1_s   = out['test_f1m'].mean(), out['test_f1m'].std()
        acc_last = out['test_acc'][-1]   # fold con más historial = mejor proxy del rendimiento futuro
        f1_last  = out['test_f1m'][-1]
        rows.append({'Model': name, 'CV Mean': acc_m, 'CV Std': acc_s,
                     'F1 Mean': f1_m, 'F1 Std': f1_s,
                     'Acc Last': acc_last, 'F1 Last': f1_last})
        folds_str = ' '.join(f'{s:.0%}' for s in out['test_acc'])
        print(f"  {name:<22}  acc {acc_m:.2%}±{acc_s:.2%}  F1 {f1_m:.2%}±{f1_s:.2%}  "
              f"último fold: {acc_last:.0%}  (folds: {folds_str})")

    print("\nRANKING (por Macro-F1 del último fold — entrenó con más historial):")
    for r in sorted(rows, key=lambda x: x['F1 Last'], reverse=True):
        print(f"  {r['Model']:<22}  acc {r['Acc Last']:.2%}   F1 {r['F1 Last']:.2%}   "
              f"(CV medio: {r['F1 Mean']:.2%}±{r['F1 Std']:.2%})")
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

    # Orden cronológico — crítico para que la validación temporal funcione.
    # Hay que hacerlo ANTES de select_columns, que descarta Fecha y Partido_id.
    if 'Fecha' in df.columns:
        df['_fecha_orden'] = pd.to_datetime(df['Fecha'], errors='coerce')
        sort_cols = ['_fecha_orden']
        if 'Partido_id' in df.columns:
            sort_cols.append('Partido_id')
        df = df.sort_values(by=sort_cols, na_position='last')
        df = df.drop(columns=['_fecha_orden']).reset_index(drop=True)
    elif 'Partido_id' in df.columns:
        df = df.sort_values('Partido_id').reset_index(drop=True)
    print(f"   ✓ Dataset ordenado cronológicamente (más antiguo → más reciente)")

    # Sede neutral: las finales se juegan en cancha neutral → Es_Local_E1 = 0
    if 'Fase' in df.columns and 'Es_Local_E1' in df.columns:
        mask_final = df['Fase'].astype(str).str.strip().str.lower() == 'final'
        if mask_final.any():
            df.loc[mask_final, 'Es_Local_E1'] = 0
            print(f"   ✓ {int(mask_final.sum())} partido(s) de Final marcados como sede neutral")

    # Features pre-partido (orden crítico: deben ir después del sort cronológico)
    df = compute_uefa_coef_features(df)
    df, team_elos = compute_elo_features(df)
    df, team_historial = compute_form_features(df)
    df, h2h_log = compute_h2h_features(df)
    df, xg_historial = compute_xg_features(df)

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

    # 10. Split CRONOLÓGICO: primeros 80 % entrenan, últimos 20 % validan.
    # El dataset ya viene ordenado por fecha desde main(), así que slice directo.
    n = len(X)
    n_train = int(n * (1 - test_size))
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    print(f"\n🔀 Split cronológico: {len(X_train)} train (antiguos) — {len(X_test)} test (recientes)")

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

    # Feature importance del mejor modelo basado en RF (más interpretable que XGB)
    feature_importance = []
    try:
        rf_pipeline = models.get('Random Forest')
        if rf_pipeline is not None:
            sk_step = rf_pipeline.named_steps.get('sk')
            rf_step = rf_pipeline.named_steps.get('rf')
            if sk_step is not None and rf_step is not None:
                mask = sk_step.get_support()
                selected_features = [c for c, keep in zip(X.columns, mask) if keep]
                importances = rf_step.feature_importances_
                feature_importance = sorted(
                    [{'feature': f, 'importance': float(imp)}
                     for f, imp in zip(selected_features, importances)],
                    key=lambda x: -x['importance'],
                )
    except Exception as e:
        print(f"   ⚠️  No se pudo calcular feature importance: {e}")

    # Entrenar modelos sobre el dataset COMPLETO (no solo train split) para
    # que predecir_partido los use directamente sin re-entrenar en cada llamada.
    # Esto hace que las predicciones sean casi instantáneas.
    print("⚡ Entrenando modelos de predicción sobre dataset completo...")
    full_models = build_classifiers(seed=42, n_features=len(X.columns))
    for clf in full_models.values():
        clf.fit(X, y)

    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    full_reg_r1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    full_reg_r2 = RandomForestRegressor(n_estimators=100, random_state=43, n_jobs=-1)
    full_reg_x1 = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                               random_state=42, verbosity=0)
    full_reg_x2 = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                               random_state=43, verbosity=0)
    y1_full = df_model['EQUIPO1_GOLES']
    y2_full = df_model['EQUIPO2_GOLES']
    full_reg_r1.fit(X, y1_full); full_reg_r2.fit(X, y2_full)
    full_reg_x1.fit(X, y1_full); full_reg_x2.fit(X, y2_full)
    print("   ✓ Modelos de predicción listos (predicción instantánea)")

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
        'team_elos': team_elos,
        'team_historial': team_historial,
        'h2h_log': h2h_log,
        'xg_historial': xg_historial,
        'feature_importance': feature_importance,
        'full_models': full_models,
        'full_regressors': {
            'Random Forest': (full_reg_r1, full_reg_r2),
            'XGBoost':       (full_reg_x1, full_reg_x2),
        },
    }


# ============================================================================
# PREDICCIÓN DE PARTIDO FUTURO  (ensemble de n_runs corridas)
# ============================================================================
def predecir_partido(equipo1, equipo2, results, n_runs=20, fase='Liga'):
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
    # Final = sede neutral → Es_Local_E1 = 0; resto de fases = local real (Equipo1)
    es_final = str(fase).strip().lower() == 'final'
    if es_final:
        print(f"⚖️  Fase: Final → sede neutral (Es_Local_E1 = 0)")
    row = {'Es_Local_E1': 0 if es_final else 1}

    # ELO actual de cada equipo (post-último partido jugado)
    team_elos = results.get('team_elos', {})
    elo_e1 = team_elos.get(equipo1, ELO_BASE)
    elo_e2 = team_elos.get(equipo2, ELO_BASE)
    print(f"   ELO actual: {equipo1} {int(elo_e1)}  vs  {equipo2} {int(elo_e2)}  (Δ {int(elo_e1-elo_e2):+d})")

    # Forma reciente y H2H usando los historiales acumulados durante main()
    historial = results.get('team_historial', {})
    h2h_log = results.get('h2h_log', {})
    forma_e1 = resumen_forma(historial.get(equipo1, []))
    forma_e2 = resumen_forma(historial.get(equipo2, []))
    h2h = resumen_h2h(h2h_log, equipo1, equipo2)
    print(f"   Forma últ.{FORMA_WINDOW}: {equipo1} {forma_e1['w']}W-{forma_e1['d']}D-{forma_e1['l']}L  ·  {equipo2} {forma_e2['w']}W-{forma_e2['d']}D-{forma_e2['l']}L")
    if h2h['n'] > 0:
        print(f"   H2H últ.{h2h['n']}: {equipo1} {h2h['w']}-{h2h['d']}-{h2h['l']}  (GF {h2h['gf']}, GC {h2h['gc']})")

    # xG sintético rolling (últimos XG_WINDOW partidos)
    xg_historial = results.get('xg_historial', {})
    h_xg_e1 = (xg_historial.get(equipo1) or [])[-XG_WINDOW:]
    h_xg_e2 = (xg_historial.get(equipo2) or [])[-XG_WINDOW:]
    xg_e1_roll  = float(np.mean([h['xg']  for h in h_xg_e1])) if h_xg_e1 else float(df_model.get('xG_E1_rolling', pd.Series([np.nan])).mean())
    xga_e1_roll = float(np.mean([h['xga'] for h in h_xg_e1])) if h_xg_e1 else float(df_model.get('xGA_E1_rolling', pd.Series([np.nan])).mean())
    xg_e2_roll  = float(np.mean([h['xg']  for h in h_xg_e2])) if h_xg_e2 else float(df_model.get('xG_E2_rolling', pd.Series([np.nan])).mean())
    xga_e2_roll = float(np.mean([h['xga'] for h in h_xg_e2])) if h_xg_e2 else float(df_model.get('xGA_E2_rolling', pd.Series([np.nan])).mean())
    if h_xg_e1 or h_xg_e2:
        print(f"   xG últ.{XG_WINDOW}: {equipo1} {xg_e1_roll:.2f} (xGA {xga_e1_roll:.2f})  ·  {equipo2} {xg_e2_roll:.2f} (xGA {xga_e2_roll:.2f})")

    # Coeficiente UEFA
    coef_e1 = UEFA_COEF.get(equipo1, UEFA_COEF_DEFAULT)
    coef_e2 = UEFA_COEF.get(equipo2, UEFA_COEF_DEFAULT)
    print(f"   Coef UEFA: {equipo1} {coef_e1:.1f}  ·  {equipo2} {coef_e2:.1f}  (Δ {coef_e1 - coef_e2:+.1f})")

    forma_features = {
        'Forma_W_E1': forma_e1['w'], 'Forma_D_E1': forma_e1['d'], 'Forma_L_E1': forma_e1['l'],
        'Forma_GF_E1': forma_e1['gf'], 'Forma_GC_E1': forma_e1['gc'], 'Forma_Pts_E1': forma_e1['pts'],
        'Forma_W_E2': forma_e2['w'], 'Forma_D_E2': forma_e2['d'], 'Forma_L_E2': forma_e2['l'],
        'Forma_GF_E2': forma_e2['gf'], 'Forma_GC_E2': forma_e2['gc'], 'Forma_Pts_E2': forma_e2['pts'],
        'Diff_Forma_Pts': forma_e1['pts'] - forma_e2['pts'],
        'Diff_Forma_GD':  (forma_e1['gf'] - forma_e1['gc']) - (forma_e2['gf'] - forma_e2['gc']),
        'Dias_Descanso_E1': 7, 'Dias_Descanso_E2': 7,  # default razonable para partidos futuros
        'H2H_W_E1': h2h['w'], 'H2H_D': h2h['d'], 'H2H_L_E1': h2h['l'],
        'H2H_GF_E1': h2h['gf'], 'H2H_GC_E1': h2h['gc'], 'H2H_N': h2h['n'],
        'xG_E1_rolling': xg_e1_roll, 'xGA_E1_rolling': xga_e1_roll,
        'xG_E2_rolling': xg_e2_roll, 'xGA_E2_rolling': xga_e2_roll,
        'Diff_xG_rolling': xg_e1_roll - xg_e2_roll,
        'Coef_UEFA_E1': coef_e1, 'Coef_UEFA_E2': coef_e2,
        'Diff_Coef_UEFA': coef_e1 - coef_e2,
    }

    for col in feat_cols:
        if col == 'ELO_E1':
            row[col] = elo_e1
        elif col == 'ELO_E2':
            row[col] = elo_e2
        elif col == 'Diff_ELO':
            row[col] = elo_e1 - elo_e2
        elif col in forma_features:
            row[col] = forma_features[col]
        elif col in stats_e1.index:
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

    n_feat = len(feat_cols)
    def make_clfs(seed):
        return build_classifiers(seed=seed, n_features=n_feat)

    # Ordenar modelos por F1 del último fold (entrenó con más historial → mejor proxy)
    model_names = list(make_clfs(0).keys())
    cv_df = results.get('cv_results')
    if cv_df is not None:
        for col in ('F1 Last', 'F1 Mean', 'CV Mean'):
            if col in cv_df.columns:
                cv_order = cv_df.set_index('Model')[col].to_dict()
                model_names = sorted(model_names, key=lambda n: cv_order.get(n, 0), reverse=True)
                break

    probas_acum = {name: [] for name in model_names}

    full_models = results.get('full_models')

    print(f"\n{'='*58}")
    print(f"🔮  {equipo1}  vs  {equipo2}")
    print(f"{'='*58}")

    if full_models:
        print("   Prediciendo con modelos pre-entrenados...", end='', flush=True)
        for name in model_names:
            if name in full_models:
                probas_acum[name].append(full_models[name].predict_proba(X_pred)[0])
        print(" listo.")
    else:
        print(f"   Entrenando ensemble ({n_runs} corridas)...", end='', flush=True)
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
    modelos_out = []
    for name in model_names:
        avg  = np.mean(probas_acum[name], axis=0)
        pred = classes[np.argmax(avg)]
        w = float(avg[win_idx])  if win_idx  is not None else 0.0
        d = float(avg[draw_idx]) if draw_idx is not None else 0.0
        l = float(avg[loss_idx]) if loss_idx is not None else 0.0
        print(f"  {name:<22} → {pred:<6}  {w:>5.1%}  {d:>5.1%}  {l:>5.1%}")
        modelos_out.append({'modelo': name, 'pred': pred, 'win': w, 'draw': d, 'loss': l})

    # --- Promedio global entre todos los modelos ---
    todas = np.mean([np.mean(probas_acum[n], axis=0) for n in model_names], axis=0)
    pred_global = classes[np.argmax(todas)]
    w_c = float(todas[win_idx])  if win_idx  is not None else 0.0
    d_c = float(todas[draw_idx]) if draw_idx is not None else 0.0
    l_c = float(todas[loss_idx]) if loss_idx is not None else 0.0
    print(f"  {'-'*55}")
    print(f"  {'CONSENSO':<22} → {pred_global:<6}  {w_c:>5.1%}  {d_c:>5.1%}  {l_c:>5.1%}")

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

    full_regressors = results.get('full_regressors')
    print(f"\n  {'Modelo':<16}  Marcador predicho")
    print(f"  {'-'*42}")
    goles_out = []
    for reg_name in (full_regressors or make_regs(0)).keys():
        if full_regressors and reg_name in full_regressors:
            r1, r2 = full_regressors[reg_name]
            g1v = float(r1.predict(X_pred)[0])
            g2v = float(r2.predict(X_pred)[0])
            if hasattr(r1, 'estimators_'):
                s1 = float(np.std([e.predict(X_pred)[0] for e in r1.estimators_]))
                s2 = float(np.std([e.predict(X_pred)[0] for e in r2.estimators_]))
            else:
                s1, s2 = 0.0, 0.0
            g1 = int(max(0, round(g1v)))
            g2 = int(max(0, round(g2v)))
        else:
            g1_preds, g2_preds = [], []
            for seed in range(n_runs):
                r1, r2 = make_regs(seed)[reg_name]
                r1.fit(X_full, y1_full)
                r2.fit(X_full, y2_full)
                g1_preds.append(r1.predict(X_pred)[0])
                g2_preds.append(r2.predict(X_pred)[0])
            g1 = int(max(0, round(np.mean(g1_preds))))
            g2 = int(max(0, round(np.mean(g2_preds))))
            s1, s2 = float(np.std(g1_preds)), float(np.std(g2_preds))
        print(f"  {reg_name:<16}  {equipo1} {g1} – {g2} {equipo2}  (±{s1:.1f} / ±{s2:.1f})")
        goles_out.append({'modelo': reg_name, 'g1': g1, 'g2': g2, 'std1': s1, 'std2': s2})

    print()
    equipos_conocidos = sorted(set(df['Equipo1'].tolist() + df['Equipo2'].tolist()))
    desconocidos = []
    for e in [equipo1, equipo2]:
        if e not in equipos_conocidos:
            print(f"  ⚠️  '{e}' no está en el historial — se usaron promedios globales")
            desconocidos.append(e)

    return {
        'equipo1': equipo1,
        'equipo2': equipo2,
        'fase': fase,
        'es_final': es_final,
        'n_runs': n_runs,
        'elo_e1': float(elo_e1),
        'elo_e2': float(elo_e2),
        'diff_elo': float(elo_e1 - elo_e2),
        'forma_e1': forma_e1,
        'forma_e2': forma_e2,
        'h2h': h2h,
        'modelos': modelos_out,
        'consenso': {'pred': pred_global, 'win': w_c, 'draw': d_c, 'loss': l_c},
        'goles': goles_out,
        'equipos_desconocidos': desconocidos,
    }


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
        predecir_partido("Kairat Almaty",  "Olympiacos",       results)
        predecir_partido("Inter",    "Liverpool", results)
        predecir_partido("Monaco",    "Galatasaray",          results)

    except FileNotFoundError:
        print(f"❌ No se encontró el archivo: {FILEPATH}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


"""El comando es:

  python agregar_partido.py --fecha YYYY-MM-DD --fase Liga --si

  Ejemplos para las jornadas que vienen:

  # Jornada 5 UCL 2025-26 (25-26 nov)  python agregar_partido.py --fecha 2025-11-25 --fase Liga --si
  python agregar_partido.py --fecha 2025-11-26 --fase Liga --si

  Flags útiles:
  - --si → auto-confirma cada partido (sin esto te pregunta uno por uno)
  - --no-headless → muestra el navegador (para depurar si algo falla)
  - --debug → guarda debug_uefa.png y .txt por cada scrape
  - --fase → omítelo y te lo pregunta (default Liga)

  Recuerda que modo_fecha lanza un subprocess por partido para evitar que Chromium se cuelgue, así que tarda un rato — cada partido toma
  ~15-30s."""