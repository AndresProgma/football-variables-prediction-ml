"""
Eval honesto walk-forward para cada mercado del panel del Pick del Día.

Flujo por fold (TimeSeriesSplit, n=3, slice cronológico):
  1. Entrenar SOLO en train:
       - GBR_g1 → predice goles esperados de E1 (target EQUIPO1_GOLES)
       - GBR_g2 → predice goles esperados de E2 (target EQUIPO2_GOLES)
       - LogisticRegression calibrado isotónico → predice prob 1X2
  2. Sobre el test fold, por cada partido:
       - Predecir λ₁, λ₂ y P_win/draw/loss
       - Llamar simular_mercados(λ₁, λ₂, P_w, P_d, P_l, e1, e2)
       - Construir la predicción de cada mercado (prob > 0.5 → "sí")
       - Compararla contra la realidad
  3. Calcular accuracy + baseline (clase mayoritaria del train) + Brier score
     por mercado.

Métricas:
  - acc       — accuracy binaria del mercado (predicho vs real)
  - acc_base  — accuracy de predecir siempre la clase mayoritaria del train
  - Δ         — diferencia (acc − acc_base)
  - brier     — error cuadrático medio (calibración de la probabilidad)

Veredicto: ✅ si Δ ≥ 3pp y brier < baseline_brier ; ⚠️ si solo uno ; ❌ si peor.

Mercados NO evaluados (porque el dataset no tiene info granular):
  - HT/FT y gol_ambas_mitades → necesitan goles por mitad, no están en el Excel.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import ml.knime_workflow_converter as pipe  # noqa: E402
from ml.dixon_coles import DixonColesModel  # noqa: E402
from experimentos.eval_mercados_ml import build_X_and_df  # noqa: E402

N_SPLITS = 3
SEED = 42


def build_data():
    """Replica el pipeline + extrae también goles reales y resultado."""
    X, df_aligned = build_X_and_df()
    mask = df_aligned["EQUIPO1_GOLES"].notna() & df_aligned["EQUIPO2_GOLES"].notna()
    X = X.loc[mask.values]
    d = df_aligned.loc[mask.values].copy()
    g1 = d["EQUIPO1_GOLES"].astype(float).values
    g2 = d["EQUIPO2_GOLES"].astype(float).values
    e1 = d["Equipo1"].astype(str).values
    e2 = d["Equipo2"].astype(str).values
    is_home = (d["Es_Local_E1"].fillna(1).astype(bool).values
               if "Es_Local_E1" in d.columns else np.ones(len(d), dtype=bool))
    fecha = d["Fecha"].values if "Fecha" in d.columns else np.array([None] * len(d))
    # Etiqueta 1X2 desde el resultado real
    res = np.where(g1 > g2, "Win", np.where(g1 < g2, "Loss", "Draw"))
    return X, g1, g2, e1, e2, res, is_home, fecha


def build_lr_classifier(n_features):
    clf = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                              random_state=SEED))])
    return CalibratedClassifierCV(clf, method="isotonic", cv=3)


def _mode_class(arr):
    vals, counts = np.unique(arr, return_counts=True)
    return vals[np.argmax(counts)]


def eval_market_binary(name, real_bool, prob_yes, base_rate):
    """Mide accuracy/Brier de un mercado binario.
    base_rate = P(real=True) en train, define la clase mayoritaria del baseline."""
    pred = (prob_yes > 0.5).astype(int)
    real = real_bool.astype(int)
    acc = accuracy_score(real, pred)
    base_class = 1 if base_rate > 0.5 else 0
    acc_base = accuracy_score(real, np.full_like(real, base_class))
    try:
        brier = brier_score_loss(real, prob_yes)
        brier_base = brier_score_loss(real, np.full_like(real, base_rate, dtype=float))
    except Exception:
        brier = brier_base = float("nan")
    return {
        "name": name, "n": len(real),
        "acc": acc, "acc_base": acc_base, "delta_pp": (acc - acc_base) * 100,
        "brier": brier, "brier_base": brier_base, "brier_delta": brier_base - brier,
        "rate_real": real.mean(), "rate_train": base_rate,
    }


def run_eval(variante: str, X, g1, g2, e1, e2, res, is_home, fecha):
    """Corre el walk-forward para una variante: 'gbr' o 'dc'."""
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    acc_by_market: dict[str, list[tuple[bool, float, float]]] = {}

    def push(name, real, prob, base_rate):
        acc_by_market.setdefault(name, []).append((bool(real), float(prob), float(base_rate)))

    print(f"\n🔬 Variante: {variante.upper()} ({'GradientBoostingRegressor para λ' if variante=='gbr' else 'Dixon-Coles MLE para λ'})")
    for fold, (tr, te) in enumerate(cv.split(X), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        g1_tr, g2_tr = g1[tr], g2[tr]
        res_tr = res[tr]
        g1_te, g2_te = g1[te], g2[te]
        res_te = res[te]
        e1_te, e2_te = e1[te], e2[te]
        is_home_te = is_home[te]

        # ── λ₁, λ₂ según variante ──────────────────────────────────
        if variante == "gbr":
            reg1 = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                             learning_rate=0.05, min_samples_leaf=4,
                                             random_state=SEED).fit(Xtr, g1_tr)
            reg2 = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                             learning_rate=0.05, min_samples_leaf=4,
                                             random_state=SEED).fit(Xtr, g2_tr)
            lam1 = np.clip(reg1.predict(Xte), 0.05, 10)
            lam2 = np.clip(reg2.predict(Xte), 0.05, 10)
        else:  # dc
            df_dc = pd.DataFrame({
                "Equipo1": e1[tr], "Equipo2": e2[tr],
                "EQUIPO1_GOLES": g1_tr, "EQUIPO2_GOLES": g2_tr,
                "Es_Local_E1": is_home[tr].astype(int),
                "Fecha": fecha[tr],
            })
            dc = DixonColesModel(max_goals=8)
            dc.fit(df_dc)
            lam1 = np.empty(len(te)); lam2 = np.empty(len(te))
            unseen = 0
            mean_g1_tr, mean_g2_tr = float(g1_tr.mean()), float(g2_tr.mean())
            for j in range(len(te)):
                e1j, e2j = str(e1_te[j]), str(e2_te[j])
                if e1j in dc.params_["attack"] and e2j in dc.params_["attack"]:
                    l1, l2 = dc.expected_score(e1j, e2j, is_home_e1=bool(is_home_te[j]))
                    lam1[j], lam2[j] = float(l1), float(l2)
                else:
                    # equipo nuevo: fallback a media del train (no podemos hacer mejor)
                    lam1[j], lam2[j] = mean_g1_tr, mean_g2_tr
                    unseen += 1
            lam1 = np.clip(lam1, 0.05, 10)
            lam2 = np.clip(lam2, 0.05, 10)
            if unseen:
                print(f"   fold {fold}: {unseen} partidos con equipo unseen → fallback a media")

        # ── Clasificador 1X2 ───────────────────────────────────────
        clf = build_lr_classifier(Xtr.shape[1]).fit(Xtr, res_tr)
        classes = list(clf.classes_)  # típicamente ['Draw','Loss','Win']
        probs = clf.predict_proba(Xte)
        idx_w = classes.index("Win")
        idx_d = classes.index("Draw")
        idx_l = classes.index("Loss")

        # ── base rates de cada mercado (con datos del train) ───────
        def br(real_arr): return float(np.mean(real_arr))
        btts_tr   = (g1_tr > 0) & (g2_tr > 0)
        over_tr   = {ln: (g1_tr + g2_tr) > ln for ln in (0.5, 1.5, 2.5, 3.5, 4.5)}
        g1_over_tr = {ln: g1_tr > ln for ln in (0.5, 1.5, 2.5)}
        g2_over_tr = {ln: g2_tr > ln for ln in (0.5, 1.5, 2.5)}
        diff_tr = g1_tr - g2_tr
        hcap_tr = {
            "E1-1":   diff_tr > 1,
            "E1-1.5": diff_tr > 1.5,
            "E1-2":   diff_tr > 2,
            "E2-1":   diff_tr < -1,
            "E2-1.5": diff_tr < -1.5,
        }
        dnb_e1_tr = (res_tr == "Win")[res_tr != "Draw"]
        doble_tr = {
            "1X": ((g1_tr > g2_tr) | (g1_tr == g2_tr)),
            "X2": ((g2_tr > g1_tr) | (g1_tr == g2_tr)),
            "12": (g1_tr != g2_tr),
        }
        wb_e1_tr = (g1_tr > g2_tr) & (g1_tr > 0) & (g2_tr > 0)
        wb_e2_tr = (g2_tr > g1_tr) & (g1_tr > 0) & (g2_tr > 0)

        # ── recorrer test fold ─────────────────────────────────────
        for i in range(len(te)):
            pw, pd_, pl = probs[i, idx_w], probs[i, idx_d], probs[i, idx_l]
            m = pipe.simular_mercados(
                float(lam1[i]), float(lam2[i]),
                float(pw), float(pd_), float(pl),
                str(e1_te[i]), str(e2_te[i]),
                n_sims=5000,  # más rápido, suficiente para accuracy
            )
            g1r, g2r = int(g1_te[i]), int(g2_te[i])
            diff_r = g1r - g2r

            # BTTS sí
            push("BTTS · sí",   (g1r > 0 and g2r > 0), m["btts"]["si"], br(btts_tr))
            # Total goles O/U
            for ln in (0.5, 1.5, 2.5, 3.5, 4.5):
                key = f"over_{int(ln)}_{int((ln-int(ln))*10)}".replace("over_0_5","over_0_5")
                k = f"over_{int(ln)}_5"
                p = m["total_goles"][k]
                push(f"Total goles +{ln}", ((g1r + g2r) > ln), p, br(over_tr[ln]))
            # Goles por equipo
            for ln in (0.5, 1.5, 2.5):
                p1 = m["goles_e1"][f"over_{int(ln)}_5"]
                p2 = m["goles_e2"][f"over_{int(ln)}_5"]
                push(f"E1 marca >{ln}", (g1r > ln), p1, br(g1_over_tr[ln]))
                push(f"E2 marca >{ln}", (g2r > ln), p2, br(g2_over_tr[ln]))
            # Hándicap
            keys_hcap = {
                f"{e1_te[i]} -1":   ("E1-1",   diff_r > 1),
                f"{e1_te[i]} -1.5": ("E1-1.5", diff_r > 1.5),
                f"{e1_te[i]} -2":   ("E1-2",   diff_r > 2),
                f"{e2_te[i]} -1":   ("E2-1",   diff_r < -1),
                f"{e2_te[i]} -1.5": ("E2-1.5", diff_r < -1.5),
            }
            for k_real, (k_tr, real_v) in keys_hcap.items():
                # buscar key en m['handicap'] que matchee el equipo y la línea
                # el dict de simular_mercados usa los nombres reales como key
                if k_real in m["handicap"]:
                    push(f"Hándicap {k_tr}", real_v, m["handicap"][k_real], br(hcap_tr[k_tr]))
            # DNB
            mask_nd = (res_te[i] != "Draw")
            if mask_nd:
                real_dnb_e1 = (g1r > g2r)
                push("DNB · E1", real_dnb_e1, m["dnb"][e1_te[i]], br(dnb_e1_tr))
            # Doble oportunidad
            for k, real_v in {
                "1X": (g1r >= g2r),
                "X2": (g2r >= g1r),
                "12": (g1r != g2r),
            }.items():
                # find key starting with k in m['doble_oportunidad']
                found = next((v for kk, v in m["doble_oportunidad"].items() if kk.startswith(k)), None)
                if found is not None:
                    push(f"Doble oportunidad {k}", real_v, found, br(doble_tr[k]))
            # Win + BTTS
            push("Win+BTTS · E1", ((g1r > g2r) and g1r > 0 and g2r > 0),
                 m["win_btts"][e1_te[i]], br(wb_e1_tr))
            push("Win+BTTS · E2", ((g2r > g1r) and g1r > 0 and g2r > 0),
                 m["win_btts"][e2_te[i]], br(wb_e2_tr))
            # Resultado correcto top-1
            top_score = m["resultado_correcto"][0]
            real_score_str = f"{g1r}-{g2r}"
            push("Resultado correcto top-1", (top_score["score"] == real_score_str),
                 top_score["prob"], 0.0)  # baseline pésimo (score exacto es ruidoso)

        print(f"   fold {fold}: train={len(tr)} test={len(te)}  predicciones OK")

    return acc_by_market


def summarize(acc_by_market):
    rows = []
    for name, data in acc_by_market.items():
        real_arr = np.array([x[0] for x in data], dtype=int)
        prob_arr = np.array([x[1] for x in data], dtype=float)
        base_rate = float(np.mean([x[2] for x in data]))
        r = eval_market_binary(name, real_arr.astype(bool), prob_arr, base_rate)
        rows.append(r)
    return {r["name"]: r for r in rows}


def print_table(title, table):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(f"{'Mercado':<26} {'n':>4} {'acc real':>9} {'acc base':>9} {'Δ':>7} "
          f"{'rate real':>10} {'brier':>8} {'brier base':>11} Veredicto")
    print("-" * 110)
    for r in sorted(table.values(), key=lambda r: -r["delta_pp"]):
        v = "✅" if r["delta_pp"] >= 3 and r["brier_delta"] > 0 else (
            "⚠️" if (r["delta_pp"] >= 0 or r["brier_delta"] > 0) else "❌")
        print(f"{r['name']:<26} {r['n']:>4d} {r['acc']*100:>8.2f}% "
              f"{r['acc_base']*100:>8.2f}% {r['delta_pp']:>+6.2f}p "
              f"{r['rate_real']*100:>9.1f}% {r['brier']:>8.3f} {r['brier_base']:>11.3f}  {v}")


def print_comparison(gbr_t, dc_t):
    print("\n" + "=" * 110)
    print("COMPARACIÓN GBR vs DIXON-COLES (por mercado)")
    print("=" * 110)
    print(f"{'Mercado':<26} {'acc GBR':>9} {'acc DC':>8} {'Δacc':>7} "
          f"{'brier GBR':>10} {'brier DC':>9} {'Δbrier':>9}  Ganador")
    print("-" * 110)
    diffs = []
    for name in sorted(set(gbr_t.keys()) | set(dc_t.keys())):
        g = gbr_t.get(name); d = dc_t.get(name)
        if g is None or d is None:
            continue
        d_acc = (d["acc"] - g["acc"]) * 100
        d_brier = g["brier"] - d["brier"]  # positivo = DC mejor (brier menor)
        diffs.append((name, d_acc, d_brier, g, d))
    # ordenar por mejora de DC sobre GBR
    diffs.sort(key=lambda x: -(x[1] + x[2] * 100))
    deltas_acc, deltas_brier = [], []
    for name, d_acc, d_brier, g, d in diffs:
        win = "DC" if (d_acc > 0 and d_brier > 0) else (
            "GBR" if (d_acc < 0 and d_brier < 0) else "≈")
        print(f"{name:<26} {g['acc']*100:>8.2f}% {d['acc']*100:>7.2f}% "
              f"{d_acc:>+6.2f}p {g['brier']:>10.3f} {d['brier']:>9.3f} "
              f"{d_brier:>+8.3f}  {win}")
        deltas_acc.append(d_acc); deltas_brier.append(d_brier)
    print("-" * 110)
    print(f"   PROMEDIO sobre {len(deltas_acc)} mercados: Δacc {np.mean(deltas_acc):+.2f}pp · "
          f"Δbrier {np.mean(deltas_brier):+.4f}  (positivo = DC mejor)")
    wins_dc = sum(1 for d_acc, d_brier, *_ in [(d_acc, d_brier) for _, d_acc, d_brier, _, _ in diffs] if d_acc > 0 and d_brier > 0)
    wins_gbr = sum(1 for d_acc, d_brier, *_ in [(d_acc, d_brier) for _, d_acc, d_brier, _, _ in diffs] if d_acc < 0 and d_brier < 0)
    print(f"   GANA DC en ambas métricas: {wins_dc}/{len(diffs)}  ·  "
          f"Gana GBR en ambas: {wins_gbr}/{len(diffs)}  ·  Empate o mixto: {len(diffs)-wins_dc-wins_gbr}")


def main():
    print("📦 Preprocesando dataset…")
    X, g1, g2, e1, e2, res, is_home, fecha = build_data()
    print(f"   X: {X.shape}  ·  partidos jugados: {len(g1)}")
    print(f"   distribución 1X2: Win={(res=='Win').mean():.0%}  "
          f"Draw={(res=='Draw').mean():.0%}  Loss={(res=='Loss').mean():.0%}")

    # ── 1X2 referencia ──
    cv2 = TimeSeriesSplit(n_splits=N_SPLITS)
    clf_accs = []
    for tr, te in cv2.split(X):
        clf = build_lr_classifier(X.shape[1]).fit(X.iloc[tr], res[tr])
        pred = clf.predict(X.iloc[te])
        clf_accs.append(accuracy_score(res[te], pred))
    print(f"   1X2 directo (LR calibrado): acc walk-forward = {np.mean(clf_accs)*100:.2f}%")

    # ── Variantes ──
    gbr_raw = run_eval("gbr", X, g1, g2, e1, e2, res, is_home, fecha)
    dc_raw  = run_eval("dc",  X, g1, g2, e1, e2, res, is_home, fecha)
    gbr_t = summarize(gbr_raw)
    dc_t  = summarize(dc_raw)

    print_table("TABLA — Variante GBR (regresores GradientBoosting para λ)", gbr_t)
    print_table("TABLA — Variante DC (Dixon-Coles MLE para λ)", dc_t)
    print_comparison(gbr_t, dc_t)

    print("\nLeyenda:")
    print("  acc real / acc base / Δ = accuracy del modelo, baseline mayoritario, diferencia")
    print("  brier                   = error cuadrático medio de la probabilidad (menor = mejor)")
    print("  ✅ Δ≥3pp Y brier mejora ; ⚠️ uno solo ; ❌ peor que el baseline")


if __name__ == "__main__":
    main()
