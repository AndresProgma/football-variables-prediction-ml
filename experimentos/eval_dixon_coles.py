"""
Evalúa Dixon-Coles vs los 6 modelos sklearn actuales con walk-forward CV.

Mide accuracy / F1 macro sobre el dataset UCL 2025-26 con la misma splits
TimeSeriesSplit que usa el pipeline en knime_workflow_converter (n_splits=3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from ml.dixon_coles import DixonColesModel

DATASET = BASE / "data" / "creando_dataset_modificado.xlsx"


def label_from_score(g1: int, g2: int) -> str:
    if g1 > g2:
        return "Win"
    if g1 < g2:
        return "Loss"
    return "Draw"


def load_df() -> pd.DataFrame:
    df = pd.read_excel(DATASET)
    df = df.dropna(subset=['EQUIPO1_GOLES', 'EQUIPO2_GOLES']).copy()
    df['EQUIPO1_GOLES'] = df['EQUIPO1_GOLES'].astype(int)
    df['EQUIPO2_GOLES'] = df['EQUIPO2_GOLES'].astype(int)
    if 'Fecha' in df.columns:
        df['_fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.sort_values('_fecha').reset_index(drop=True)
    return df


def run_walk_forward(df: pd.DataFrame, n_splits: int = 3, xi: float = 0.0):
    n = len(df)
    test_size = n // (n_splits + 1)
    print(f"\nDataset: {n} partidos, {df['Equipo1'].nunique()} equipos distintos")
    print(f"TimeSeriesSplit n_splits={n_splits}, test ≈ {test_size} partidos por fold, xi={xi}")
    print("=" * 70)

    cv = TimeSeriesSplit(n_splits=n_splits)
    accs, f1s = [], []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(df), 1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model = DixonColesModel(max_goals=10)
        ref = train_df['_fecha'].max() if '_fecha' in train_df.columns else None
        model.fit(train_df, xi=xi, ref_date=ref)

        preds, reals = [], []
        unseen = 0
        for _, r in test_df.iterrows():
            e1, e2 = str(r['Equipo1']), str(r['Equipo2'])
            is_home = bool(r.get('Es_Local_E1', 1))
            real = label_from_score(int(r['EQUIPO1_GOLES']), int(r['EQUIPO2_GOLES']))
            reals.append(real)
            if e1 not in model.params_['attack'] or e2 not in model.params_['attack']:
                unseen += 1
                preds.append("Win")  # baseline: predict home win
                continue
            preds.append(model.predict(e1, e2, is_home_e1=is_home))

        acc = accuracy_score(reals, preds)
        f1m = f1_score(reals, preds, labels=['Win', 'Draw', 'Loss'], average='macro', zero_division=0)
        accs.append(acc)
        f1s.append(f1m)
        cnt = pd.Series(preds).value_counts().to_dict()
        unseen_str = f", {unseen} unseen-team" if unseen else ""
        print(f"  fold {fold_i}: train={len(train_df):3d} test={len(test_df):3d}  "
              f"acc={acc:.2%}  F1m={f1m:.2%}  preds={cnt}{unseen_str}")

    print("-" * 70)
    print(f"  CV mean acc: {np.mean(accs):.2%} ± {np.std(accs):.2%}")
    print(f"  CV mean F1m: {np.mean(f1s):.2%} ± {np.std(f1s):.2%}")
    print(f"  último fold (más historial): acc {accs[-1]:.2%}  F1m {f1s[-1]:.2%}")
    return accs, f1s


def evaluate_track_record_subset(df: pd.DataFrame, n_test: int = 35):
    """Entrena con todos menos los últimos n_test, predice esos n_test.
    Mismo set que ml/_generar_record.py para poder comparar con los otros modelos."""
    print("\n" + "=" * 70)
    print(f"EVALUACIÓN ESTILO RECORD (train: primeros {len(df) - n_test}, test: últimos {n_test})")
    print("=" * 70)
    train_df = df.iloc[:-n_test]
    test_df = df.iloc[-n_test:]
    model = DixonColesModel(max_goals=10)
    ref = train_df['_fecha'].max() if '_fecha' in train_df.columns else None
    model.fit(train_df, xi=0.0, ref_date=ref)

    preds, reals = [], []
    correct = 0
    for _, r in test_df.iterrows():
        e1, e2 = str(r['Equipo1']), str(r['Equipo2'])
        is_home = bool(r.get('Es_Local_E1', 1))
        real = label_from_score(int(r['EQUIPO1_GOLES']), int(r['EQUIPO2_GOLES']))
        reals.append(real)
        if e1 not in model.params_['attack'] or e2 not in model.params_['attack']:
            preds.append("Win")
            continue
        p = model.predict(e1, e2, is_home_e1=is_home)
        preds.append(p)
        if p == real:
            correct += 1

    print(f"  Resultado: {correct}/{len(test_df)} = {correct / len(test_df):.2%}")
    print(f"  F1 macro:  {f1_score(reals, preds, labels=['Win','Draw','Loss'], average='macro', zero_division=0):.2%}")
    return correct / len(test_df)


def main():
    df = load_df()
    print(f"\n📊 DIXON-COLES — evaluación honesta sin leakage")
    run_walk_forward(df, n_splits=3, xi=0.0)
    run_walk_forward(df, n_splits=3, xi=0.0019)
    evaluate_track_record_subset(df, n_test=35)


if __name__ == "__main__":
    main()
