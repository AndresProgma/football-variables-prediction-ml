"""
Eval HONESTO de los regresores que predicen goles (E1 y E2).

Estos λ₁/λ₂ son la entrada a simular_mercados → de ahí salen BTTS, total goles,
hándicap, resultado correcto exacto, etc. Si los regresores erran mucho, TODOS
los mercados derivados van a ser malos.

Métricas walk-forward (TimeSeriesSplit n=3) sobre las 372 filas del dataset:
  - MAE goles E1 y E2
  - MAE total goles (g1+g2)
  - MAE diferencia (g1-g2)
  - Accuracy del ganador (W/D/L derivado del marcador predicho)
  - Accuracy del marcador exacto (round y match con real)
  - Accuracy +/-1 gol (predicción "casi acertada")

Compara 3 modelos para los λ:
  - GBR  : GradientBoostingRegressor (lo que usa el pipeline hoy)
  - DC   : Dixon-Coles MLE (modelo clásico de fútbol)
  - mean : baseline = predecir la media del train (ej. 1.7 / 1.2)
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from experimentos.eval_panel_mercados import build_data  # noqa: E402
from ml.dixon_coles import DixonColesModel  # noqa: E402

N_SPLITS = 3
SEED = 42


def label(g1, g2):
    if g1 > g2: return "W"
    if g1 < g2: return "L"
    return "D"


def evaluate(name, lam1_pred, lam2_pred, g1_real, g2_real):
    """Medir un set de predicciones de goles vs realidad."""
    g1_round = np.rint(lam1_pred).astype(int)
    g2_round = np.rint(lam2_pred).astype(int)
    g1_real_i = g1_real.astype(int)
    g2_real_i = g2_real.astype(int)

    mae_g1 = mean_absolute_error(g1_real, lam1_pred)
    mae_g2 = mean_absolute_error(g2_real, lam2_pred)
    mae_total = mean_absolute_error(g1_real + g2_real, lam1_pred + lam2_pred)
    mae_diff  = mean_absolute_error(g1_real - g2_real, lam1_pred - lam2_pred)

    # ganador (W/D/L) del marcador PREDICHO redondeado vs real
    real_lbl = np.array([label(a, b) for a, b in zip(g1_real_i, g2_real_i)])
    pred_lbl = np.array([label(a, b) for a, b in zip(g1_round, g2_round)])
    acc_winner = (real_lbl == pred_lbl).mean()

    # marcador exacto
    acc_exact = ((g1_round == g1_real_i) & (g2_round == g2_real_i)).mean()

    # +/-1 gol en cada lado
    acc_plusminus1 = (
        (np.abs(g1_round - g1_real_i) <= 1) & (np.abs(g2_round - g2_real_i) <= 1)
    ).mean()

    return {
        "name": name,
        "MAE_g1": mae_g1, "MAE_g2": mae_g2,
        "MAE_total": mae_total, "MAE_diff": mae_diff,
        "acc_winner": acc_winner, "acc_exact": acc_exact,
        "acc_pm1": acc_plusminus1,
    }


def main():
    print("📦 Cargando dataset…")
    X, g1, g2, e1, e2, res, is_home, fecha = build_data()
    print(f"   {len(X)} partidos · X={X.shape[1]} features")
    print(f"   goles reales: g1 mean={g1.mean():.2f} std={g1.std():.2f}  ·  "
          f"g2 mean={g2.mean():.2f} std={g2.std():.2f}")
    print(f"   total goles: mean={(g1+g2).mean():.2f} std={(g1+g2).std():.2f}")

    cv = TimeSeriesSplit(n_splits=N_SPLITS)

    # Acumular predicciones de cada variante
    preds = {"GBR": [[], []], "DC": [[], []], "mean": [[], []]}
    reals = [[], []]  # g1, g2

    for fold, (tr, te) in enumerate(cv.split(X), 1):
        print(f"\nfold {fold}: train={len(tr)}  test={len(te)}")
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        g1_tr, g2_tr = g1[tr], g2[tr]
        e1_tr, e2_tr = e1[tr], e2[tr]
        is_home_tr = is_home[tr]
        fecha_tr = fecha[tr]
        g1_te, g2_te = g1[te], g2[te]
        e1_te, e2_te = e1[te], e2[te]
        is_home_te = is_home[te]

        # GBR
        reg1 = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, min_samples_leaf=4,
                                         random_state=SEED).fit(Xtr, g1_tr)
        reg2 = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, min_samples_leaf=4,
                                         random_state=SEED).fit(Xtr, g2_tr)
        gbr_l1 = np.clip(reg1.predict(Xte), 0.05, 10)
        gbr_l2 = np.clip(reg2.predict(Xte), 0.05, 10)
        preds["GBR"][0].extend(gbr_l1); preds["GBR"][1].extend(gbr_l2)

        # DC
        df_dc = pd.DataFrame({
            "Equipo1": e1_tr, "Equipo2": e2_tr,
            "EQUIPO1_GOLES": g1_tr, "EQUIPO2_GOLES": g2_tr,
            "Es_Local_E1": is_home_tr.astype(int),
            "Fecha": fecha_tr,
        })
        dc = DixonColesModel(max_goals=8).fit(df_dc)
        dc_l1 = np.empty(len(te)); dc_l2 = np.empty(len(te))
        for j in range(len(te)):
            ej1, ej2 = str(e1_te[j]), str(e2_te[j])
            if ej1 in dc.params_["attack"] and ej2 in dc.params_["attack"]:
                l1, l2 = dc.expected_score(ej1, ej2, is_home_e1=bool(is_home_te[j]))
                dc_l1[j], dc_l2[j] = float(l1), float(l2)
            else:
                dc_l1[j], dc_l2[j] = g1_tr.mean(), g2_tr.mean()
        preds["DC"][0].extend(dc_l1); preds["DC"][1].extend(dc_l2)

        # mean baseline
        preds["mean"][0].extend([float(g1_tr.mean())] * len(te))
        preds["mean"][1].extend([float(g2_tr.mean())] * len(te))

        reals[0].extend(g1_te); reals[1].extend(g2_te)

    g1_r = np.array(reals[0]); g2_r = np.array(reals[1])
    rows = []
    for name in ("GBR", "DC", "mean"):
        l1 = np.array(preds[name][0]); l2 = np.array(preds[name][1])
        rows.append(evaluate(name, l1, l2, g1_r, g2_r))

    # Tabla
    print(f"\n{'='*90}")
    print(f"REGRESORES DE GOLES — walk-forward sobre {len(g1_r)} partidos de test")
    print(f"{'='*90}")
    print(f"{'Modelo':<10} {'MAE g1':>7} {'MAE g2':>7} {'MAE total':>10} {'MAE diff':>9} "
          f"{'ac winner':>10} {'ac exact':>9} {'ac +/-1':>8}")
    print("-" * 90)
    for r in rows:
        print(f"{r['name']:<10} {r['MAE_g1']:>7.3f} {r['MAE_g2']:>7.3f} "
              f"{r['MAE_total']:>10.3f} {r['MAE_diff']:>9.3f} "
              f"{r['acc_winner']*100:>9.2f}% {r['acc_exact']*100:>8.2f}% "
              f"{r['acc_pm1']*100:>7.2f}%")

    print()
    print("Interpretación rápida:")
    print(" - MAE ~ 1 gol significa que el promedio del error es 1 gol entero (alto).")
    print(" - acc winner: % de aciertos del W/D/L derivado del marcador predicho redondeado.")
    print(" - acc exact:  % de aciertos del marcador exacto (ej. predijo 2-1, real fue 2-1).")
    print(" - acc +/-1:   % donde el error es ≤ 1 gol en cada lado (marcador casi correcto).")
    print(" - SI 'mean' está a la par del modelo → no hay señal real, todos los mercados")
    print("   derivados (BTTS, hándicap, total goles, resultado correcto) heredan ese error.")


if __name__ == "__main__":
    main()
