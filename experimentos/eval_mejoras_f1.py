"""
Evalúa mejoras propuestas con walk-forward CV (mismas splits que el pipeline).
Solo se commitea lo que SUBE acc o F1 macro sin trampa.

Variantes:
  baseline               — config actual (DRAW_WEIGHT_BOOST=1.5, sin paridad)
  paridad                — agrega abs(Diff_ELO/Coef/Forma/xG) a X
  draw_boost_2.0/2.5     — sube DRAW_WEIGHT_BOOST
  draw_mult_1.3/1.5/1.8  — multiplica P(Draw) antes de argmax (post-hoc)
  dc_rho_*               — Dixon-Coles con ρ fijo en grid
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import ml.knime_workflow_converter as pipe
from ml.dixon_coles import DixonColesModel

DATASET = BASE / "data" / "creando_dataset_modificado.xlsx"
N_SPLITS = 3
SKLEARN_MODELS = ['Random Forest', 'Gradient Boosting', 'Logistic Regression',
                  'SVM', 'XGBoost', 'KNN']


def preprocess():
    """Replica el pipeline hasta sacar X, y, df_dc, class_labels."""
    df = pipe.load_data(str(DATASET))
    if 'Fecha' in df.columns:
        df['_fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.sort_values('_fecha').reset_index(drop=True)

    if 'Fase' in df.columns and 'Es_Local_E1' in df.columns:
        m = df['Fase'].astype(str).str.strip().str.lower() == 'final'
        if m.any():
            df.loc[m, 'Es_Local_E1'] = 0

    df = pipe.compute_uefa_coef_features(df)
    df, _ = pipe.compute_elo_features(df)
    df, _ = pipe.compute_form_features(df)
    df, _ = pipe.compute_h2h_features(df)
    df, _ = pipe.compute_xg_features(df)
    df = pipe.select_columns(df)
    df = pipe.handle_missing_values(df, strategy='mean')
    df = pipe.create_derived_variables(df)
    df = pipe.filter_rows(df)
    ts = pipe.aggregate_by_team(df)
    df = pipe.join_team_stats(df, ts)
    df_model, le_dict = pipe.prepare_for_modeling(df)

    exclude = ['Partido_id', 'Fase', 'Equipo1', 'Equipo2',
               'EQUIPO1_GOLES', 'EQUIPO2_GOLES', 'Resultado_E1', 'Diferencia_Goles',
               'Goles_dentro_area_E1', 'Goles_Fuera_Area_E1',
               'Goles_dentro_area_E2', 'Goles_Fuera_Area_E2',
               'goles_encajados_E1', 'goles_encajados_E2',
               'Goles_encajados_propia_puerta_E1', 'Goles_encajados_propia_puerta_E2',
               'Porterias_a_cero_E1', 'Porterias_a_cero_E2',
               'Eficiencia_Tiros_E1', 'Eficiencia_Tiros_E2']
    X = df_model.drop(columns=[c for c in exclude if c in df_model.columns])
    y = df_model['Resultado_E1']
    class_labels = list(le_dict['Resultado_E1'].classes_)  # ['Draw','Loss','Win']

    df_dc = df.loc[X.index, [c for c in ['Equipo1', 'Equipo2',
                                          'EQUIPO1_GOLES', 'EQUIPO2_GOLES',
                                          'Es_Local_E1']
                              if c in df.columns]].copy()
    if 'Fecha' in df.columns:
        df_dc['Fecha'] = df.loc[X.index, 'Fecha'].values

    return X, y, df_dc, class_labels, le_dict


def cv_sklearn(X, y, draw_mult: float = 1.0) -> dict:
    """Walk-forward CV de los 6 sklearn. Opcionalmente multiplica P(Draw) por
    draw_mult antes del argmax (post-hoc threshold)."""
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    classifiers = pipe.build_classifiers(seed=42, n_features=X.shape[1])
    out = {}
    for name, clf in classifiers.items():
        accs, f1s = [], []
        for tr, te in cv.split(X):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]
            from sklearn.base import clone
            c = clone(clf)
            c.fit(X_tr, y_tr)
            if draw_mult == 1.0:
                pred = c.predict(X_te)
            else:
                proba = c.predict_proba(X_te)
                proba = proba.copy()
                draw_idx = 0  # ['Draw','Loss','Win'] → Draw is index 0
                proba[:, draw_idx] *= draw_mult
                proba = proba / proba.sum(axis=1, keepdims=True)
                pred = np.argmax(proba, axis=1)
            accs.append(accuracy_score(y_te, pred))
            f1s.append(f1_score(y_te, pred, average='macro', zero_division=0))
        out[name] = (np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s),
                     accs[-1], f1s[-1])
    return out


def cv_dixon_coles(df_dc, rho=None) -> tuple:
    """Walk-forward CV de DC. Si rho=None usa MLE; si se pasa, lo fija."""
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    accs, f1s = [], []
    for tr, te in cv.split(df_dc):
        train, test = df_dc.iloc[tr], df_dc.iloc[te]
        dc = DixonColesModel(max_goals=10)
        if rho is not None:
            # Hack: fittear DC normalmente y luego sobreescribir rho
            dc.fit(train)
            dc.params_['rho'] = float(rho)
        else:
            dc.fit(train)
        preds, reals = [], []
        for _, r in test.iterrows():
            e1, e2 = str(r['Equipo1']), str(r['Equipo2'])
            is_home = bool(r.get('Es_Local_E1', 1))
            g1, g2 = int(r['EQUIPO1_GOLES']), int(r['EQUIPO2_GOLES'])
            real = 'Win' if g1 > g2 else ('Loss' if g1 < g2 else 'Draw')
            reals.append(real)
            if e1 not in dc.params_['attack'] or e2 not in dc.params_['attack']:
                preds.append('Win')
            else:
                preds.append(dc.predict(e1, e2, is_home_e1=is_home))
        accs.append(accuracy_score(reals, preds))
        f1s.append(f1_score(reals, preds, labels=['Win', 'Draw', 'Loss'],
                            average='macro', zero_division=0))
    return (np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s),
            accs[-1], f1s[-1])


def add_paridad_features(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    if 'Diff_ELO' in X.columns:
        X2['Abs_Diff_ELO'] = X['Diff_ELO'].abs()
    if 'Diff_Coef_UEFA' in X.columns:
        X2['Abs_Diff_Coef_UEFA'] = X['Diff_Coef_UEFA'].abs()
    if 'Diff_Forma_Pts' in X.columns:
        X2['Abs_Diff_Forma_Pts'] = X['Diff_Forma_Pts'].abs()
    if 'Diff_xG_rolling' in X.columns:
        X2['Abs_Diff_xG_rolling'] = X['Diff_xG_rolling'].abs()
    return X2


def fmt(t):
    am, as_, fm, fs, al, fl = t
    return f"acc {am*100:5.2f}±{as_*100:4.2f}  F1 {fm*100:5.2f}±{fs*100:4.2f}  "\
           f"last acc {al*100:.0f}%  F1 {fl*100:.0f}%"


def main():
    print("📦 Preprocesando dataset...")
    X, y, df_dc, class_labels, le_dict = preprocess()
    print(f"   X: {X.shape[1]} features  ·  y: {len(y)} partidos  ·  clases: {class_labels}")

    # ─── EXPERIMENTOS sklearn ─────────────────────────────────────
    experimentos = {}

    print("\n[1/8] baseline (DRAW_WEIGHT_BOOST=1.5, sin paridad)")
    pipe.DRAW_WEIGHT_BOOST = 1.5
    experimentos['baseline'] = cv_sklearn(X, y)

    print("[2/8] paridad (+4 features abs)")
    Xp = add_paridad_features(X)
    pipe.DRAW_WEIGHT_BOOST = 1.5
    experimentos['paridad'] = cv_sklearn(Xp, y)

    print("[3/8] draw_boost=2.0")
    pipe.DRAW_WEIGHT_BOOST = 2.0
    experimentos['draw_boost_2.0'] = cv_sklearn(X, y)

    print("[4/8] draw_boost=2.5")
    pipe.DRAW_WEIGHT_BOOST = 2.5
    experimentos['draw_boost_2.5'] = cv_sklearn(X, y)

    pipe.DRAW_WEIGHT_BOOST = 1.5  # restore
    print("[5/8] draw_mult post-hoc = 1.3")
    experimentos['draw_mult_1.3'] = cv_sklearn(X, y, draw_mult=1.3)

    print("[6/8] draw_mult post-hoc = 1.5")
    experimentos['draw_mult_1.5'] = cv_sklearn(X, y, draw_mult=1.5)

    print("[7/8] draw_mult post-hoc = 1.8")
    experimentos['draw_mult_1.8'] = cv_sklearn(X, y, draw_mult=1.8)

    # paridad + draw_mult combo (mejor de ambos si fueran independientes)
    print("[8/8] paridad + draw_mult=1.5 (combo)")
    experimentos['combo_paridad_mult1.5'] = cv_sklearn(Xp, y, draw_mult=1.5)

    # ─── EXPERIMENTOS DC ──────────────────────────────────────────
    print("\nDC experiments (ρ grid):")
    dc_exps = {}
    dc_exps['dc_baseline'] = cv_dixon_coles(df_dc, rho=None)
    for rho in (-0.30, -0.20, -0.10, -0.05, 0.0):
        dc_exps[f'dc_rho_{rho}'] = cv_dixon_coles(df_dc, rho=rho)

    # ─── RESULTADOS sklearn ───────────────────────────────────────
    print("\n" + "=" * 100)
    print("RESULTADOS sklearn (promedio del ensemble = avg de los 6 modelos)")
    print("=" * 100)
    base = experimentos['baseline']

    def avg(res):
        accs = [v[0] for v in res.values()]
        f1s = [v[2] for v in res.values()]
        return np.mean(accs), np.mean(f1s)

    base_acc, base_f1 = avg(base)
    print(f"\n  {'Experimento':<32}  {'avg acc':>8}  {'Δ acc':>7}  {'avg F1':>8}  {'Δ F1':>7}")
    print("  " + "-" * 78)
    for nombre, res in experimentos.items():
        a, f = avg(res)
        da = (a - base_acc) * 100
        df_ = (f - base_f1) * 100
        marca = "✅" if (da >= 0 and df_ >= 0) else ("⚠️" if da >= 0 or df_ >= 0 else "❌")
        print(f"  {nombre:<32}  {a*100:7.2f}%  {da:+6.2f}p  {f*100:7.2f}%  {df_:+6.2f}p  {marca}")

    print("\n  Detalle por modelo del MEJOR experimento:")
    best_name = max(experimentos.keys(), key=lambda k: avg(experimentos[k])[1])
    print(f"  (mejor por F1 macro: {best_name})\n")
    for m in SKLEARN_MODELS:
        b = base[m]
        x = experimentos[best_name][m]
        print(f"    {m:<22} baseline: F1 {b[2]*100:.1f}%  acc {b[0]*100:.1f}%  →  "
              f"{best_name}: F1 {x[2]*100:.1f}%  acc {x[0]*100:.1f}%  "
              f"(ΔF1 {(x[2]-b[2])*100:+.1f}p, Δacc {(x[0]-b[0])*100:+.1f}p)")

    # ─── RESULTADOS DC ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("RESULTADOS Dixon-Coles (ρ grid)")
    print("=" * 100)
    for nombre, t in dc_exps.items():
        print(f"  {nombre:<22}  {fmt(t)}")

    # ─── VEREDICTO ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("VEREDICTO")
    print("=" * 100)
    keep = []
    for nombre, res in experimentos.items():
        if nombre == 'baseline':
            continue
        a, f = avg(res)
        if a > base_acc + 0.005 and f > base_f1 + 0.005:
            keep.append((nombre, a - base_acc, f - base_f1))
    if keep:
        print("✅ Variantes que SUBEN acc Y F1 (deltas significativos > 0.5p):")
        for n, da, df_ in keep:
            print(f"    {n:<32}  Δacc {da*100:+.2f}p  ΔF1 {df_*100:+.2f}p")
    else:
        print("❌ Ninguna variante sklearn supera el baseline en AMBAS métricas con margen.")

    # DC
    dc_base = dc_exps['dc_baseline']
    keep_dc = []
    for n, t in dc_exps.items():
        if n == 'dc_baseline':
            continue
        if t[0] > dc_base[0] + 0.005 and t[2] > dc_base[2] + 0.005:
            keep_dc.append((n, t[0] - dc_base[0], t[2] - dc_base[2]))
    if keep_dc:
        print("✅ Variantes DC que mejoran ambos (significativo):")
        for n, da, df_ in keep_dc:
            print(f"    {n:<22}  Δacc {da*100:+.2f}p  ΔF1 {df_*100:+.2f}p")
    else:
        print("❌ El ρ MLE del DC ya es óptimo (ninguna ρ fija lo supera con margen).")


if __name__ == "__main__":
    main()
