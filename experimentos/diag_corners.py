"""
Diagnóstico: ¿por qué falla el regresor de corners totales?

1. ¿Hay correlación entre las features pre-partido y los corners reales?
   (si no la hay, el modelo no puede aprender — no es problema de modelo
    sino de señal en el dataset)
2. ¿Qué features importa el GBR cuando lo entrenamos? ¿Son débiles?
3. ¿Cómo se comportan las predicciones vs el real? ¿predice siempre lo mismo
   (cerca de la media), o predice algo pero erra mucho?
4. ¿Hay patrones temporales o por fase?
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from experimentos.eval_mercados_ml import build_X_and_df  # noqa: E402

def main():
    print("📦 Preparando datos…")
    X, df = build_X_and_df()
    y = (df["Saques_de_esquina_sacados_E1"] + df["Saques_de_esquina_sacados_E2"]).astype(float)
    mask = y.notna()
    X, df, y = X.loc[mask.values], df.loc[mask.values], y.loc[mask.values]
    print(f"X: {X.shape} · y (corners totales): mean={y.mean():.2f}  std={y.std():.2f}  (filtrados {(~mask).sum()} NaN)")

    # ─── 1. CORRELACIONES (Pearson) ──────────────────────────────
    print("\n" + "=" * 70)
    print("1. CORRELACIONES Pearson de features pre-partido con corners totales")
    print("=" * 70)
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    print(f"\n   Top 15 features por |corr|:\n")
    for name, c in corrs.head(15).items():
        bar = "█" * int(c * 50)
        print(f"   {name:<40} |corr|={c:.3f}  {bar}")
    print(f"\n   |corr| MAX: {corrs.iloc[0]:.3f}  (señal: 0.7+ fuerte, 0.3-0.7 moderada, <0.3 débil)")
    print(f"   |corr| mediana: {corrs.median():.3f}")
    print(f"   features con |corr| > 0.2: {(corrs > 0.2).sum()} / {len(corrs)}")
    print(f"   features con |corr| > 0.3: {(corrs > 0.3).sum()} / {len(corrs)}")

    # ─── 2. BASELINE TRIVIAL: ¿qué tan estable es la media? ──────
    print("\n" + "=" * 70)
    print("2. ¿La media del train predice bien? (cuánto varía corners por partido)")
    print("=" * 70)
    print(f"\n   Distribución real:")
    print(f"   min={y.min():.0f}  p10={np.percentile(y,10):.1f}  "
          f"p25={np.percentile(y,25):.1f}  p50={np.percentile(y,50):.1f}  "
          f"p75={np.percentile(y,75):.1f}  p90={np.percentile(y,90):.1f}  max={y.max():.0f}")
    print(f"   Si predigo siempre 9.5 (media), MAE ≈ desviación media:")
    print(f"     MAD (|y - mediana|).mean() = {(y - y.median()).abs().mean():.2f}")
    print(f"     MAE vs media               = {(y - y.mean()).abs().mean():.2f}")
    print(f"     Eso es el FLOOR del error — un buen modelo solo puede bajar de ahí si")
    print(f"     hay señal sistemática en las features.")

    # ─── 3. FEATURE IMPORTANCE del GBR entrenado en todo ─────────
    print("\n" + "=" * 70)
    print("3. FEATURE IMPORTANCE del GBR (entrenado en TODO el dataset)")
    print("=" * 70)
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=4, random_state=42,
    )
    gbr.fit(X, y)
    fi = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n   Top 15 features que el modelo usa:\n")
    for name, imp in fi.head(15).items():
        bar = "█" * int(imp * 200)
        print(f"   {name:<40} imp={imp:.4f}  {bar}")
    print(f"\n   Importance acumulada top-5: {fi.head(5).sum():.2%}")
    print(f"   Importance acumulada top-10: {fi.head(10).sum():.2%}")
    print(f"   (si las top concentran <50% → ninguna feature manda, modelo perdido)")

    # ─── 4. ANÁLISIS DE PREDICCIONES VS REAL ─────────────────────
    print("\n" + "=" * 70)
    print("4. ¿QUÉ PREDICE el modelo? — walk-forward, último fold")
    print("=" * 70)
    cv = TimeSeriesSplit(n_splits=3)
    splits = list(cv.split(X))
    tr, te = splits[-1]
    gbr2 = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=4, random_state=42,
    )
    gbr2.fit(X.iloc[tr], y.iloc[tr])
    pred = gbr2.predict(X.iloc[te])
    real = y.iloc[te].values

    print(f"\n   Estadísticas de las predicciones (fold final, n={len(te)}):")
    print(f"   real:  mean={real.mean():.2f}  std={real.std():.2f}  rango=[{real.min():.0f}, {real.max():.0f}]")
    print(f"   pred:  mean={pred.mean():.2f}  std={pred.std():.2f}  rango=[{pred.min():.2f}, {pred.max():.2f}]")
    print(f"\n   👉 Si pred.std << real.std, el modelo está prediciendo casi siempre lo mismo")
    print(f"      (es decir, no encuentra señal y se refugia en la media).")
    print(f"      Ratio std(pred)/std(real) = {pred.std() / real.std():.2f}")
    print(f"      → 0.0-0.3: predice casi constante (sin señal)")
    print(f"      → 0.3-0.7: predice algo pero limitado")
    print(f"      → 0.7+   : predice con varianza real")

    # casos donde más se equivoca
    err = np.abs(real - pred)
    idx_worst = np.argsort(-err)[:8]
    df_te = df.iloc[te].iloc[idx_worst]
    print(f"\n   Peores 8 errores del fold final:")
    print(f"   {'real':>5}  {'pred':>6}  {'err':>5}  partido")
    for j, i in enumerate(idx_worst):
        r = df_te.iloc[j]
        e1 = str(r.get("Equipo1", "?"))
        e2 = str(r.get("Equipo2", "?"))
        f = r.get("Fecha", "")
        print(f"   {real[i]:>5.0f}  {pred[i]:>6.2f}  {err[i]:>5.2f}  {e1} vs {e2} ({f})")

    print(f"\n   👉 Si los peores errores son partidos 'raros' (muchos corners por")
    print(f"      remate dominante de un solo equipo), eso confirma que corners")
    print(f"      depende del flujo del partido, no de la fuerza pre-partido.")

if __name__ == "__main__":
    main()
