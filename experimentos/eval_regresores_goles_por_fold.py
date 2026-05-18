"""
Mismo eval que eval_regresores_goles.py pero reporta cada fold por separado.

Por qué: el promedio de los 3 folds mezcla:
  - Fold 1 con SOLO 95 partidos de train (pocos, modelos no entrenan bien)
  - Fold 3 con 279 partidos de train (mucho contexto, escenario real)

El fold 3 es el más representativo de "predecir partidos futuros del 25-26
teniendo todo lo previo".
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


def evaluate(l1, l2, g1_r, g2_r):
    g1_round = np.rint(l1).astype(int)
    g2_round = np.rint(l2).astype(int)
    g1_i = g1_r.astype(int); g2_i = g2_r.astype(int)
    return dict(
        mae_total=mean_absolute_error(g1_r + g2_r, l1 + l2),
        mae_diff=mean_absolute_error(g1_r - g2_r, l1 - l2),
        acc_winner=np.mean([label(a,b)==label(c,d) for a,b,c,d in zip(g1_round,g2_round,g1_i,g2_i)]),
        acc_exact=float(np.mean((g1_round == g1_i) & (g2_round == g2_i))),
    )


def main():
    X, g1, g2, e1, e2, res, is_home, fecha = build_data()
    print(f"Total: {len(X)} partidos · fechas {fecha[0]} → {fecha[-1]}\n")

    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    by_fold = {"GBR": [], "DC": [], "mean": []}

    for fold, (tr, te) in enumerate(cv.split(X), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        g1_tr, g2_tr = g1[tr], g2[tr]
        g1_te, g2_te = g1[te], g2[te]

        reg1 = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, min_samples_leaf=4,
                                         random_state=SEED).fit(Xtr, g1_tr)
        reg2 = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, min_samples_leaf=4,
                                         random_state=SEED).fit(Xtr, g2_tr)
        gbr_l1 = np.clip(reg1.predict(Xte), 0.05, 10)
        gbr_l2 = np.clip(reg2.predict(Xte), 0.05, 10)

        df_dc = pd.DataFrame({
            "Equipo1": e1[tr], "Equipo2": e2[tr],
            "EQUIPO1_GOLES": g1_tr, "EQUIPO2_GOLES": g2_tr,
            "Es_Local_E1": is_home[tr].astype(int),
            "Fecha": fecha[tr],
        })
        dc = DixonColesModel(max_goals=8).fit(df_dc)
        dc_l1 = np.empty(len(te)); dc_l2 = np.empty(len(te))
        for j in range(len(te)):
            ej1, ej2 = str(e1[te][j]), str(e2[te][j])
            if ej1 in dc.params_["attack"] and ej2 in dc.params_["attack"]:
                l1, l2 = dc.expected_score(ej1, ej2, is_home_e1=bool(is_home[te][j]))
                dc_l1[j], dc_l2[j] = float(l1), float(l2)
            else:
                dc_l1[j], dc_l2[j] = g1_tr.mean(), g2_tr.mean()

        mean_l1 = np.full(len(te), float(g1_tr.mean()))
        mean_l2 = np.full(len(te), float(g2_tr.mean()))

        by_fold["GBR"].append((len(tr), len(te), fecha[te[0]], fecha[te[-1]],
                               evaluate(gbr_l1, gbr_l2, g1_te, g2_te)))
        by_fold["DC"].append((len(tr), len(te), fecha[te[0]], fecha[te[-1]],
                              evaluate(dc_l1, dc_l2, g1_te, g2_te)))
        by_fold["mean"].append((len(tr), len(te), fecha[te[0]], fecha[te[-1]],
                                evaluate(mean_l1, mean_l2, g1_te, g2_te)))

    # Tabla
    for fold_i in range(N_SPLITS):
        ntr, nte, fmin, fmax, _ = by_fold["GBR"][fold_i]
        print(f"=== Fold {fold_i+1}  ·  train={ntr}, test={nte}  ·  test={fmin} → {fmax} ===")
        print(f"  {'Modelo':<6} {'MAE total':>10} {'MAE diff':>10} {'acc winner':>11} {'acc exact':>10}")
        for name in ("GBR", "DC", "mean"):
            r = by_fold[name][fold_i][4]
            print(f"  {name:<6} {r['mae_total']:>10.3f} {r['mae_diff']:>10.3f} "
                  f"{r['acc_winner']*100:>10.2f}% {r['acc_exact']*100:>9.2f}%")
        print()


if __name__ == "__main__":
    main()
