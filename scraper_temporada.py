"""
Driver para scrapear temporadas completas de UCL al dataset.

Itera por fechas conocidas de una temporada, llamando agregar_partido.py
--fecha YYYY-MM-DD --si --fase <fase> en subprocess por cada fecha.

Cada partido (dentro de cada fecha) ya viene aislado en subprocess por el
propio agregar_partido.py modo_fecha, así que esto es estable.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
AGREGAR = BASE / "agregar_partido.py"
DATASET = BASE / "creando_dataset_modificado.xlsx"

# Calendario UCL 2024-25 (primer año con nuevo formato Liga)
UCL_2024_25 = [
    # League phase
    ('2024-09-17', 'Liga'), ('2024-09-18', 'Liga'), ('2024-09-19', 'Liga'),
    ('2024-10-01', 'Liga'), ('2024-10-02', 'Liga'),
    ('2024-10-22', 'Liga'), ('2024-10-23', 'Liga'),
    ('2024-11-05', 'Liga'), ('2024-11-06', 'Liga'),
    ('2024-11-26', 'Liga'), ('2024-11-27', 'Liga'),
    ('2024-12-10', 'Liga'), ('2024-12-11', 'Liga'),
    ('2025-01-21', 'Liga'), ('2025-01-22', 'Liga'),
    ('2025-01-29', 'Liga'),  # MD8 all 18 matches same day
    # Knockout phase playoffs (ida + vuelta)
    ('2025-02-11', 'Playoffs'), ('2025-02-12', 'Playoffs'),
    ('2025-02-18', 'Playoffs'), ('2025-02-19', 'Playoffs'),
    # Round of 16
    ('2025-03-04', 'Octavos'), ('2025-03-05', 'Octavos'),
    ('2025-03-11', 'Octavos'), ('2025-03-12', 'Octavos'),
    # Quarter-finals
    ('2025-04-08', 'Cuartos'), ('2025-04-09', 'Cuartos'),
    ('2025-04-15', 'Cuartos'), ('2025-04-16', 'Cuartos'),
    # Semi-finals
    ('2025-04-29', 'Semifinal'), ('2025-04-30', 'Semifinal'),
    ('2025-05-06', 'Semifinal'), ('2025-05-07', 'Semifinal'),
    # Final
    ('2025-05-31', 'Final'),
]

# Calendario UCL 2023-24 (último año con formato grupos)
UCL_2023_24 = [
    # Group stage (6 jornadas)
    ('2023-09-19', 'Grupos'), ('2023-09-20', 'Grupos'),
    ('2023-10-03', 'Grupos'), ('2023-10-04', 'Grupos'),
    ('2023-10-24', 'Grupos'), ('2023-10-25', 'Grupos'),
    ('2023-11-07', 'Grupos'), ('2023-11-08', 'Grupos'),
    ('2023-11-28', 'Grupos'), ('2023-11-29', 'Grupos'),
    ('2023-12-12', 'Grupos'), ('2023-12-13', 'Grupos'),
    # Round of 16
    ('2024-02-13', 'Octavos'), ('2024-02-14', 'Octavos'),
    ('2024-02-20', 'Octavos'), ('2024-02-21', 'Octavos'),
    ('2024-03-05', 'Octavos'), ('2024-03-06', 'Octavos'),
    ('2024-03-12', 'Octavos'), ('2024-03-13', 'Octavos'),
    # Quarter-finals
    ('2024-04-09', 'Cuartos'), ('2024-04-10', 'Cuartos'),
    ('2024-04-16', 'Cuartos'), ('2024-04-17', 'Cuartos'),
    # Semi-finals
    ('2024-04-30', 'Semifinal'), ('2024-05-01', 'Semifinal'),
    ('2024-05-07', 'Semifinal'), ('2024-05-08', 'Semifinal'),
    # Final
    ('2024-06-01', 'Final'),
]

TEMPORADAS = {'2024-25': UCL_2024_25, '2023-24': UCL_2023_24}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--temporada', required=True, choices=list(TEMPORADAS.keys()))
    p.add_argument('--desde', help='Reanudar desde esta fecha YYYY-MM-DD')
    args = p.parse_args()

    fechas = TEMPORADAS[args.temporada]
    if args.desde:
        fechas = [f for f in fechas if f[0] >= args.desde]

    print(f"=== Scrape UCL {args.temporada}: {len(fechas)} fechas ===")
    n_antes = len(pd.read_excel(DATASET))
    print(f"Dataset antes: {n_antes} partidos\n")

    for i, (fecha, fase) in enumerate(fechas, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(fechas)}] Procesando {fecha}  (fase={fase})")
        print(f"{'='*60}")
        cmd = [sys.executable, '-u', str(AGREGAR),
               '--fecha', fecha, '--fase', fase, '--si']
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("Interrumpido por usuario.")
            return

    n_despues = len(pd.read_excel(DATASET))
    print(f"\n{'='*60}")
    print(f"Temporada {args.temporada} completada.")
    print(f"Dataset: {n_antes} → {n_despues}  ({n_despues - n_antes:+d} partidos)")


if __name__ == '__main__':
    main()
