"""
Driver que reanuda 2025-26 desde donde quedó: J8 resto → playoffs → octavos →
cuartos → semis → final. Llama predecir_y_agregar.py --fecha por cada fecha.

Cada partido se predice con dataset previo (track record honesto) y luego
se agrega.
"""

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
PREDICT = BASE / "predecir_y_agregar.py"
DATASET = BASE / "creando_dataset_modificado.xlsx"

# Calendario UCL 2025-26 (lo que falta a partir del estado actual)
FECHAS = [
    ('2026-01-28', 'Liga'),          # J8 — 13 partidos restantes
    # Playoffs (febrero)
    ('2026-02-17', 'Playoffs'), ('2026-02-18', 'Playoffs'),
    ('2026-02-24', 'Playoffs'), ('2026-02-25', 'Playoffs'),
    # Octavos (marzo)
    ('2026-03-10', 'Octavos'), ('2026-03-11', 'Octavos'),
    ('2026-03-17', 'Octavos'), ('2026-03-18', 'Octavos'),
    # Cuartos (abril)
    ('2026-04-07', 'Cuartos'), ('2026-04-08', 'Cuartos'),
    ('2026-04-14', 'Cuartos'), ('2026-04-15', 'Cuartos'),
    # Semifinales (abril-mayo)
    ('2026-04-28', 'Semifinal'), ('2026-04-29', 'Semifinal'),
    ('2026-05-05', 'Semifinal'), ('2026-05-06', 'Semifinal'),
    # Final
    ('2026-05-30', 'Final'),
]


def main():
    n_antes = len(pd.read_excel(DATASET))
    print(f"=== Reanudando UCL 2025-26: {len(FECHAS)} fechas ===")
    print(f"Dataset antes: {n_antes} partidos\n")

    for i, (fecha, fase) in enumerate(FECHAS, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(FECHAS)}] {fecha}  (fase={fase})")
        print(f"{'='*60}")
        cmd = [sys.executable, '-u', str(PREDICT),
               '--fecha', fecha, '--fase', fase, '--n-runs', '20']
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("Interrumpido.")
            return

    n_despues = len(pd.read_excel(DATASET))
    print(f"\nDataset: {n_antes} → {n_despues}  ({n_despues - n_antes:+d})")


if __name__ == '__main__':
    main()
