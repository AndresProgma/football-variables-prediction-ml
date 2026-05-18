"""
Evalúa qué mercados de apuestas son predecibles con ML a partir del dataset.

Para cada target candidato (corners, tarjetas, disparos a puerta, faltas, etc.):
  1. Entrena GradientBoostingRegressor con las MISMAS features pre-partido
     que usa el clasificador 1X2 del pipeline (ELO, Forma, H2H, xG, Coef UEFA
     + team_stats agregadas).
  2. Quita las columnas post-partido del propio target para evitar leakage
     directo.
  3. Mide R²/MAE walk-forward (TimeSeriesSplit n_splits=3) vs baseline
     "predecir la media del train".
  4. Para cada línea O/U sugerida: deriva clasificación Over/Under con
     pred > line y mide accuracy vs baseline (clase mayoritaria del train).

Criterio: un mercado vale la pena solo si MAE_modelo < MAE_baseline (R² > 0)
Y al menos una línea O/U supera al baseline mayoritario por > 5pp.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import ml.knime_workflow_converter as pipe  # noqa: E402

DATASET = BASE / "data" / "creando_dataset_modificado.xlsx"
N_SPLITS = 3
SEED = 42

# columnas post-partido del Excel — todas estas son leakage si están en X y
# se usan para predecir algo de la misma familia.
POST_PARTIDO_BASE = [
    "EQUIPO1_GOLES", "EQUIPO2_GOLES",
    "Goles_dentro_area_E1", "Goles_dentro_area_E2",
    "Goles_Fuera_Area_E1", "Goles_Fuera_Area_E2",
    "goles_encajados_E1", "goles_encajados_E2",
    "Goles_encajados_propia_puerta_E1", "Goles_encajados_propia_puerta_E2",
    "Porterias_a_cero_E1", "Porterias_a_cero_E2",
    "Eficiencia_Tiros_E1", "Eficiencia_Tiros_E2",
    "Saques_de_esquina_sacados_E1", "Saques_de_esquina_sacados_E2",
    "Tarjetas_amarillas_E1", "Tarjetas_amarillas_E2",
    "Tarjetas_rojas_E1", "Tarjetas_rojas_E2",
    "Faltas_cometidas_E1", "Faltas_cometidas_E2",
    "Faltas_cometidas_tercio_def_E1", "Faltas_cometidas_tercio_def_E2",
    "Faltas_cometidas_en_campo_propio_E1", "Faltas_cometidas_en_campo_propio_E2",
    "Disparos_totales_E1", "Disparos_totales_E2",
    "Disparos_a_puerta_E1", "Disparos_a_puerta_E2",
    "Disparos_fuera_E1", "Disparos_fuera_E2",
    "Disparo_Bloqueados_E1", "Disparo_Bloqueados_E2",
    "Disparos_Al palo_E1", "Disparos_Al palo_E2",
    "Disparos_Larguero_E1", "Disparos_Larguero_E2",
    "Disparos_Poste_E1", "Disparos_Poste_E2",
    "Disparos_a_puerta_fuera_del_area_E1", "Disparos_a_puerta_fuera_del_area_E2",
    "Disparos_fuera_desde_fuera_del_area_E1", "Disparos_fuera_desde_fuera_del_area_E2",
    "Posesion_E1", "Posesion_E2",
    "Tiempo_de_posesion_E1", "Tiempo_de_posesion_E2",
    "Precision_pase_E1", "Precision_pase_E2",
    "Pases_completados_E1", "Pases_completados_E2",
    "Pases_realizados_E1", "Pases_realizados_E2",
    "Pases_cortos_completados_E1", "Pases_cortos_completados_E2",
    "Pases_media_distancia_completados_E1", "Pases_media_distancia_completados_E2",
    "Pases_en_largo_completados_E1", "Pases_en_largo_completados_E2",
    "Pases_completados_atras_E1", "Pases_completados_atras_E2",
    "Pases_completadosa_izquierda_E1", "Pases_completadosa_izquierda_E2",
    "Pases_completados_derecha_E1", "Pases_completados_derecha_E2",
    "Pases_zonas_clave_E1", "Pases_zonas_clave_E2",
    "Pases_al_area_E1", "Pases_al_area_E2",
    "Fueras_de_juego_E1", "Fueras_de_juego_E2",
    "Resultado_E1", "Diferencia_Goles",
]

# (nombre, función para construir target desde df, líneas O/U a evaluar)
MERCADOS = [
    ("Corners totales",
     lambda d: d["Saques_de_esquina_sacados_E1"] + d["Saques_de_esquina_sacados_E2"],
     [7.5, 8.5, 9.5, 10.5, 11.5]),
    ("Corners E1",
     lambda d: d["Saques_de_esquina_sacados_E1"],
     [3.5, 4.5, 5.5, 6.5]),
    ("Corners E2",
     lambda d: d["Saques_de_esquina_sacados_E2"],
     [2.5, 3.5, 4.5, 5.5]),
    ("Diff corners (E1-E2)",
     lambda d: d["Saques_de_esquina_sacados_E1"] - d["Saques_de_esquina_sacados_E2"],
     [-2.5, -0.5, 0.5, 2.5]),
    ("Tarjetas totales",
     lambda d: (d["Tarjetas_amarillas_E1"] + d["Tarjetas_amarillas_E2"]
                + d["Tarjetas_rojas_E1"] + d["Tarjetas_rojas_E2"]),
     [2.5, 3.5, 4.5, 5.5]),
    ("Disparos a puerta totales",
     lambda d: d["Disparos_a_puerta_E1"] + d["Disparos_a_puerta_E2"],
     [8.5, 9.5, 10.5, 11.5, 12.5]),
    ("Faltas totales",
     lambda d: d["Faltas_cometidas_E1"] + d["Faltas_cometidas_E2"],
     [19.5, 21.5, 22.5, 23.5, 24.5]),
    ("Posesion E1 (>55%)",
     lambda d: d["Posesion_E1"],
     [45.0, 50.0, 55.0, 60.0]),
    ("Disparos totales (E1+E2)",
     lambda d: d["Disparos_totales_E1"] + d["Disparos_totales_E2"],
     [24.5, 26.5, 28.5, 30.5]),
]


def build_X_and_df():
    """Replica el pipeline hasta tener X (features) y df completo (con targets)."""
    df = pipe.load_data(str(DATASET))
    df["_fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.sort_values("_fecha").reset_index(drop=True)

    if "Fase" in df.columns and "Es_Local_E1" in df.columns:
        m = df["Fase"].astype(str).str.strip().str.lower() == "final"
        if m.any():
            df.loc[m, "Es_Local_E1"] = 0

    df = pipe.compute_uefa_coef_features(df)
    df, _ = pipe.compute_elo_features(df)
    df, _ = pipe.compute_form_features(df)
    df, _ = pipe.compute_h2h_features(df)
    df, _ = pipe.compute_xg_features(df)
    df, _ = pipe.compute_market_rolling_features(df)

    df_raw = df.copy()  # guardar todas las columnas originales antes del filter

    df = pipe.select_columns(df)
    df = pipe.handle_missing_values(df, strategy="mean")
    df = pipe.create_derived_variables(df)
    df = pipe.filter_rows(df)
    ts = pipe.aggregate_by_team(df)
    df = pipe.join_team_stats(df, ts)
    df_model, le_dict = pipe.prepare_for_modeling(df)

    # quitar TODO lo post-partido como feature, dejar solo features pre-partido
    drop = [c for c in POST_PARTIDO_BASE if c in df_model.columns]
    extra_meta = ["Partido_id", "Fase", "Equipo1", "Equipo2"]
    drop += [c for c in extra_meta if c in df_model.columns]
    X = df_model.drop(columns=drop).select_dtypes(include=[np.number])

    # alinear df_raw a X.index — df_raw mantiene los targets post-partido
    df_aligned = df_raw.loc[X.index].copy()

    return X, df_aligned


def evaluate_target(name, target_series, X, lines):
    """Walk-forward CV de GBR vs baseline (mean del train) sobre el target."""
    y = target_series.values.astype(float)

    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    maes_m, maes_b, r2s = [], [], []
    line_acc = {ln: {"model": [], "baseline": [], "rate_over_train": []} for ln in lines}

    for tr, te in cv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        gbr = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=4, random_state=SEED,
        )
        gbr.fit(Xtr, ytr)
        pred = gbr.predict(Xte)

        mean_train = float(ytr.mean())
        base_pred = np.full_like(yte, mean_train, dtype=float)

        maes_m.append(mean_absolute_error(yte, pred))
        maes_b.append(mean_absolute_error(yte, base_pred))
        r2s.append(r2_score(yte, pred))

        # accuracy O/U por línea
        for ln in lines:
            real_over = (yte > ln).astype(int)
            model_over = (pred > ln).astype(int)
            # baseline: clase mayoritaria de Over en train
            rate_train = float((ytr > ln).mean())
            base_class = 1 if rate_train > 0.5 else 0
            base_over = np.full_like(real_over, base_class)

            line_acc[ln]["model"].append(accuracy_score(real_over, model_over))
            line_acc[ln]["baseline"].append(accuracy_score(real_over, base_over))
            line_acc[ln]["rate_over_train"].append(rate_train)

    mae_m, mae_b = float(np.mean(maes_m)), float(np.mean(maes_b))
    r2 = float(np.mean(r2s))
    improv_pct = 100.0 * (mae_b - mae_m) / mae_b if mae_b > 0 else 0.0

    print(f"\n📊 {name}")
    print(f"   target mean={target_series.mean():.2f}  std={target_series.std():.2f}  "
          f"n={len(target_series)}")
    print(f"   MAE modelo: {mae_m:.3f}   MAE baseline (predecir mean): {mae_b:.3f}   "
          f"R²: {r2:+.3f}   mejora MAE: {improv_pct:+.1f}%")

    best_line = None
    best_delta = -1e9
    for ln in lines:
        acc_m = float(np.mean(line_acc[ln]["model"]))
        acc_b = float(np.mean(line_acc[ln]["baseline"]))
        rate = float(np.mean(line_acc[ln]["rate_over_train"]))
        delta = (acc_m - acc_b) * 100
        flag = "✅" if delta >= 3 else ("⚠️" if delta >= 0 else "❌")
        print(f"     O/U {ln:+5.1f}  acc modelo {acc_m*100:5.2f}%  "
              f"acc baseline {acc_b*100:5.2f}%  Δ {delta:+5.2f}pp  "
              f"(rate_over_train {rate*100:4.1f}%) {flag}")
        if delta > best_delta:
            best_delta = delta
            best_line = ln

    veredicto = (improv_pct > 0) and (best_delta >= 3)
    return {
        "name": name,
        "mae_model": mae_m,
        "mae_baseline": mae_b,
        "r2": r2,
        "mae_improv_pct": improv_pct,
        "best_line": best_line,
        "best_delta_pp": best_delta,
        "veredicto": "VALE LA PENA ✅" if veredicto else "NO VALE LA PENA ❌",
    }


def main():
    print("📦 Preprocesando dataset…")
    X, df_aligned = build_X_and_df()
    print(f"   X: {X.shape[1]} features  ·  {len(X)} partidos  ·  "
          f"target columns disponibles del df: {sum(1 for c in POST_PARTIDO_BASE if c in df_aligned.columns)}")

    resultados = []
    for nombre, target_fn, lines in MERCADOS:
        try:
            target = target_fn(df_aligned)
        except KeyError as e:
            print(f"\n⚠️  {nombre}: falta columna {e}, salteo")
            continue
        target = pd.to_numeric(target, errors="coerce")
        mask = target.notna()
        if mask.sum() < len(X) - 5:
            print(f"\n⚠️  {nombre}: {(~mask).sum()} NaNs, salteando filas")
        # alinear X y target por máscara
        Xc = X.loc[mask.values]
        tc = target.loc[mask.values]
        if len(Xc) < 60:
            print(f"\n⚠️  {nombre}: solo {len(Xc)} filas, no evaluable")
            continue
        resultados.append(evaluate_target(nombre, tc, Xc, lines))

    print("\n" + "=" * 90)
    print("RESUMEN — qué mercados merecen subir al pipeline")
    print("=" * 90)
    print(f"\n  {'Mercado':<30} {'MAE mod':>8} {'MAE base':>9} {'R²':>7} "
          f"{'mej%':>6} {'mejor línea':>13} {'Δacc':>7}  Veredicto")
    print("  " + "-" * 100)
    for r in sorted(resultados, key=lambda r: -r["best_delta_pp"]):
        print(f"  {r['name']:<30} {r['mae_model']:>8.3f} {r['mae_baseline']:>9.3f} "
              f"{r['r2']:>+7.3f} {r['mae_improv_pct']:>+5.1f}% "
              f"{r['best_line']:>+12.1f}  {r['best_delta_pp']:>+5.2f}pp  "
              f"{r['veredicto']}")


if __name__ == "__main__":
    main()
