"""
Harness de experimentos: evalúa configuraciones del pipeline con walk-forward
CV y registra resultados a experimentos/resultados.csv.

Cada configuración define:
  - dataset (path al excel)
  - features extra: incluir / excluir xG, UEFA coef, etc
  - clasificadores: ¿calibrar?, ¿pesos de clase custom?
  - n_runs del ensemble

Uso:
    python experimentos/harness.py --listar           # listar runs
    python experimentos/harness.py --run baseline     # un escenario
    python experimentos/harness.py --run-todos        # todos
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

import ml.knime_workflow_converter as pipe
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier


DATASET = str(BASE / "data" / "creando_dataset_modificado.xlsx")
RESULTADOS_CSV = BASE / "experimentos" / "resultados.csv"


def build_classifiers_v2(seed=42, n_features=None,
                          class_weight_draw_boost=1.0,
                          calibrar=False, k=25):
    """Versión configurable de los clasificadores.

    class_weight_draw_boost: multiplica el peso de la clase Draw (default 1.0 = sin cambio).
        Útil porque Draw está subrepresentado y los modelos casi nunca lo predicen.
    calibrar: envuelve cada clasificador en CalibratedClassifierCV(method='isotonic', cv=3).
        Las probabilidades quedan calibradas (un 70% real ≈ acierta 70%).
    """
    k_eff = min(k, n_features) if n_features else k
    sk = lambda: SelectKBest(f_classif, k=k_eff)

    if class_weight_draw_boost != 1.0:
        # clase 0 = Draw (LabelEncoder ordena alfabéticamente)
        cw = {0: class_weight_draw_boost, 1: 1.0, 2: 1.0}
    else:
        cw = 'balanced'

    rf  = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3,
                                 class_weight=cw, random_state=seed, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3,
                                     random_state=seed)
    lr  = LogisticRegression(max_iter=2000, C=0.3, class_weight=cw, random_state=seed)
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
              class_weight=cw, random_state=seed)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                        random_state=seed, verbosity=0, use_label_encoder=False,
                        eval_metric='mlogloss')
    knn = KNeighborsClassifier(n_neighbors=7)

    def wrap(clf, scale=False):
        steps = []
        if scale:
            steps.append(('sc', StandardScaler()))
        steps.append(('sk', sk()))
        steps.append(('clf', clf))
        p = Pipeline(steps)
        if calibrar:
            return CalibratedClassifierCV(p, method='isotonic', cv=3)
        return p

    return {
        'Random Forest':       wrap(rf),
        'Gradient Boosting':   wrap(gb),
        'Logistic Regression': wrap(lr, scale=True),
        'SVM':                 wrap(svm, scale=True),
        'XGBoost':             wrap(xgb),
        'KNN':                 wrap(knn, scale=True),
    }


def evaluar_walkforward(X, y, n_splits=5, **kwargs):
    """TimeSeriesSplit walk-forward sobre features X / target y.
    Devuelve dict {modelo: {acc_mean, acc_std, f1_mean, logloss}}."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    n_feat = X.shape[1]
    resumen = {}

    # Definir conjunto de modelos para esta evaluación
    sample_models = build_classifiers_v2(seed=0, n_features=n_feat, **kwargs)
    for name in sample_models.keys():
        resumen[name] = {'accs': [], 'f1s': [], 'lls': []}

    for fold_idx, (tr, te) in enumerate(tscv.split(X)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        if len(set(ytr)) < 2 or len(set(yte)) < 1:
            continue
        modelos = build_classifiers_v2(seed=fold_idx, n_features=n_feat, **kwargs)
        for name, clf in modelos.items():
            try:
                clf.fit(Xtr, ytr)
                ypred = clf.predict(Xte)
                acc = accuracy_score(yte, ypred)
                f1 = f1_score(yte, ypred, average='macro', zero_division=0)
                resumen[name]['accs'].append(acc)
                resumen[name]['f1s'].append(f1)
                if hasattr(clf, 'predict_proba'):
                    try:
                        prob = clf.predict_proba(Xte)
                        # Asegurar que los labels coincidan
                        labels_clf = clf.classes_ if hasattr(clf, 'classes_') else sorted(set(yte))
                        ll = log_loss(yte, prob, labels=list(labels_clf))
                        resumen[name]['lls'].append(ll)
                    except Exception:
                        pass
            except Exception as e:
                print(f"  ⚠️  {name} fold {fold_idx} falló: {e}")

    out = {}
    for name, r in resumen.items():
        out[name] = {
            'acc_mean': float(np.mean(r['accs'])) if r['accs'] else None,
            'acc_std':  float(np.std(r['accs']))  if r['accs'] else None,
            'f1_mean':  float(np.mean(r['f1s']))  if r['f1s']  else None,
            'logloss':  float(np.mean(r['lls']))  if r['lls']  else None,
            'n_folds':  len(r['accs']),
        }
    return out


def cargar_dataset_procesado(features_extra=True):
    """Aplica el pipeline de preprocesamiento del módulo y devuelve X, y."""
    df = pipe.load_data(DATASET)
    if 'Fecha' in df.columns:
        df['_fecha_orden'] = pd.to_datetime(df['Fecha'], errors='coerce')
        sort_cols = ['_fecha_orden']
        if 'Partido_id' in df.columns:
            sort_cols.append('Partido_id')
        df = df.sort_values(by=sort_cols, na_position='last')
        df = df.drop(columns=['_fecha_orden']).reset_index(drop=True)
    elif 'Partido_id' in df.columns:
        df = df.sort_values('Partido_id').reset_index(drop=True)

    if 'Fase' in df.columns and 'Es_Local_E1' in df.columns:
        m = df['Fase'].astype(str).str.strip().str.lower() == 'final'
        df.loc[m, 'Es_Local_E1'] = 0

    if features_extra:
        df = pipe.compute_uefa_coef_features(df)
        df, _ = pipe.compute_elo_features(df)
        df, _ = pipe.compute_form_features(df)
        df, _ = pipe.compute_h2h_features(df)
        df, _ = pipe.compute_xg_features(df)
    else:
        df, _ = pipe.compute_elo_features(df)
        df, _ = pipe.compute_form_features(df)
        df, _ = pipe.compute_h2h_features(df)

    df = pipe.select_columns(df)
    df = pipe.handle_missing_values(df)
    df = pipe.create_derived_variables(df)
    df = pipe.filter_rows(df)
    team_stats = pipe.aggregate_by_team(df)
    df = pipe.join_team_stats(df, team_stats)
    df_model, le_dict = pipe.prepare_for_modeling(df)

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
    y = df_model['Resultado_E1']
    return X, y, df_model


# ====== ESCENARIOS A EVALUAR ======
ESCENARIOS = {
    'baseline_old':       {'features_extra': False,
                            'kwargs': {}},
    'v1_xg_uefa':         {'features_extra': True,
                            'kwargs': {}},
    'v2_calibrado':       {'features_extra': True,
                            'kwargs': {'calibrar': True}},
    'v3_draw_boost_2x':   {'features_extra': True,
                            'kwargs': {'class_weight_draw_boost': 2.0}},
    'v4_draw_boost_calib':{'features_extra': True,
                            'kwargs': {'calibrar': True, 'class_weight_draw_boost': 2.0}},
    'v5_k15':             {'features_extra': True,
                            'kwargs': {'k': 15}},
    'v6_k40':             {'features_extra': True,
                            'kwargs': {'k': 40}},
    'v7_k15_calib':       {'features_extra': True,
                            'kwargs': {'k': 15, 'calibrar': True}},
    'v8_draw_boost_1.5_calib': {'features_extra': True,
                            'kwargs': {'calibrar': True, 'class_weight_draw_boost': 1.5}},
    'v9_draw_boost_3_calib': {'features_extra': True,
                            'kwargs': {'calibrar': True, 'class_weight_draw_boost': 3.0}},
    'v10_k15_draw_boost_calib': {'features_extra': True,
                            'kwargs': {'calibrar': True, 'class_weight_draw_boost': 2.0, 'k': 15}},
    'v11_k20_draw_boost_calib': {'features_extra': True,
                            'kwargs': {'calibrar': True, 'class_weight_draw_boost': 2.0, 'k': 20}},
    'v12_k25_db1.5_calib': {'features_extra': True,
                            'kwargs': {'calibrar': True, 'class_weight_draw_boost': 1.5, 'k': 25}},
    'v13_k30_db2_calib':   {'features_extra': True,
                            'kwargs': {'calibrar': True, 'class_weight_draw_boost': 2.0, 'k': 30}},
    'v14_k20_db1.5_calib': {'features_extra': True,
                            'kwargs': {'calibrar': True, 'class_weight_draw_boost': 1.5, 'k': 20}},
}


def correr_escenario(nombre):
    cfg = ESCENARIOS[nombre]
    print(f"\n========================================")
    print(f"  Escenario: {nombre}")
    print(f"  Config: features_extra={cfg['features_extra']}  {cfg['kwargs']}")
    print(f"========================================")
    X, y, _ = cargar_dataset_procesado(features_extra=cfg['features_extra'])
    print(f"  X.shape = {X.shape}")
    res = evaluar_walkforward(X, y, n_splits=5, **cfg['kwargs'])

    print(f"\n  Resultados walk-forward (TimeSeriesSplit 5 folds):")
    print(f"  {'Modelo':<22} {'Acc':>7}  {'Acc Std':>8}  {'F1':>7}  {'LogLoss':>8}  Folds")
    print('  ' + '-'*70)
    for name, r in res.items():
        acc = r['acc_mean']
        std = r['acc_std']
        f1 = r['f1_mean']
        ll = r['logloss']
        print(f"  {name:<22} {acc*100 if acc else 0:>6.2f}%  {std*100 if std else 0:>7.2f}%  "
              f"{f1*100 if f1 else 0:>6.2f}%  {ll if ll else 0:>8.3f}  {r['n_folds']}")

    # Guardar
    rows = []
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    n_partidos = len(y)
    for modelo, r in res.items():
        rows.append({
            'timestamp': ts,
            'escenario': nombre,
            'n_partidos': n_partidos,
            'modelo': modelo,
            'acc_mean': r['acc_mean'],
            'acc_std':  r['acc_std'],
            'f1_mean':  r['f1_mean'],
            'logloss':  r['logloss'],
            'n_folds':  r['n_folds'],
            'config': json.dumps(cfg['kwargs']),
            'features_extra': cfg['features_extra'],
        })
    df_row = pd.DataFrame(rows)
    if RESULTADOS_CSV.exists():
        df_row.to_csv(RESULTADOS_CSV, mode='a', header=False, index=False)
    else:
        df_row.to_csv(RESULTADOS_CSV, mode='w', header=True, index=False)
    print(f"  💾 Resultados appendados a {RESULTADOS_CSV.name}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--listar', action='store_true')
    p.add_argument('--run', help='Nombre del escenario')
    p.add_argument('--run-todos', action='store_true')
    args = p.parse_args()

    if args.listar:
        for k, v in ESCENARIOS.items():
            print(f"  {k:<22} {v}")
        return

    if args.run:
        correr_escenario(args.run)
    elif args.run_todos:
        for nombre in ESCENARIOS:
            try:
                correr_escenario(nombre)
            except Exception as e:
                print(f"❌ {nombre} falló: {e}")
                import traceback; traceback.print_exc()
    else:
        p.print_help()


if __name__ == '__main__':
    main()
