"""
Rellena la columna Fecha de los partidos existentes manualmente.

Uso:
    python agregar_fechas.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

import pandas as pd

DATASET = r"C:\Users\fehgb\OneDrive\Desktop\prediccion futybol\creando_dataset_modificado.xlsx"

df = pd.read_excel(DATASET)

print("=" * 50)
print("  AGREGAR FECHAS — formato YYYY-MM-DD")
print("  (Enter para saltar un partido)")
print("=" * 50 + "\n")

for idx, row in df.iterrows():
    if pd.notna(row.get('Fecha')):
        continue

    e1 = row['Equipo1']
    e2 = row['Equipo2']
    g1 = int(row['EQUIPO1_GOLES'])
    g2 = int(row['EQUIPO2_GOLES'])

    fecha = input(f"[{int(row['Partido_id']):2}] {e1} {g1}-{g2} {e2}  → ").strip()
    if fecha:
        df.at[idx, 'Fecha'] = fecha

df.to_excel(DATASET, index=False)
print(f"\n✅ Guardado. Fechas completadas: {df['Fecha'].notna().sum()} de {len(df)}")
