"""
Genera record_historico.json con los últimos 35 partidos completados.
Entrena UNA VEZ con los partidos anteriores, luego predice los 35.
"""
import json, tempfile, os, sys
import pandas as pd
import numpy as np
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

DATASET = _PROJECT_ROOT / "data" / "creando_dataset_modificado.xlsx"
OUTPUT  = _PROJECT_ROOT / "data" / "record_historico.json"

from ml.knime_workflow_converter import main as run_pipeline, predecir_partido

N_TEST = 35

# 1. Cargar y ordenar dataset
df = pd.read_excel(DATASET)
df['_f'] = pd.to_datetime(df.get('Fecha'), errors='coerce')
df = df.sort_values('_f').reset_index(drop=True)

# 2. Filtrar solo partidos con resultado
completados = df.dropna(subset=['EQUIPO1_GOLES','EQUIPO2_GOLES']).reset_index(drop=True)
print(f"Partidos completados: {len(completados)}")

# 3. Split: entrenamiento = todos menos los últimos 35
n_train = len(completados) - N_TEST
train_df = completados.iloc[:n_train].copy()
test_df  = completados.iloc[n_train:].copy()
print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")

# 4. Entrenar pipeline solo con datos de entrenamiento
tmp = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
tmp.close()
train_df.drop(columns=['_f']).to_excel(tmp.name, index=False)
print("\n🧠 Entrenando pipeline con datos de entrenamiento...")
results = run_pipeline(tmp.name)
os.remove(tmp.name)
print("✅ Pipeline entrenado\n")

# 5. Predecir cada partido del test
registros = []
for i, (_, row) in enumerate(test_df.iterrows()):
    e1   = str(row['Equipo1'])
    e2   = str(row['Equipo2'])
    fase = str(row.get('Fase','Liga'))
    g1r  = int(row['EQUIPO1_GOLES'])
    g2r  = int(row['EQUIPO2_GOLES'])
    fecha = str(row.get('Fecha',''))[:10] if pd.notna(row.get('Fecha')) else '—'

    resultado_real = 'Win' if g1r > g2r else 'Loss' if g1r < g2r else 'Draw'
    print(f"[{i+1}/{N_TEST}] Prediciendo {e1} vs {e2} ({fecha})... ", end='', flush=True)

    try:
        pred = predecir_partido(e1, e2, results, n_runs=20, fase=fase)
        cons = pred['consenso']
        ok   = cons['pred'] == resultado_real
        print(f"{'✅' if ok else '❌'}  pred={cons['pred']}  real={resultado_real}  {g1r}-{g2r}")

        registros.append({
            'equipo1':        e1,
            'equipo2':        e2,
            'fecha':          fecha,
            'fase':           fase,
            'goles_real':     f"{g1r}–{g2r}",
            'resultado_real': resultado_real,
            'consenso': {
                'pred': cons['pred'],
                'win':  round(cons['win'], 4),
                'draw': round(cons['draw'], 4),
                'loss': round(cons['loss'], 4),
            },
            'modelos': [
                {
                    'modelo': m['modelo'],
                    'pred':   m['pred'],
                    'win':    round(m['win'], 4),
                    'draw':   round(m['draw'], 4),
                    'loss':   round(m['loss'], 4),
                }
                for m in pred.get('modelos', [])
            ],
            'acierto': ok,
        })
    except Exception as exc:
        print(f"ERROR: {exc}")
        registros.append({'equipo1':e1,'equipo2':e2,'fecha':fecha,'error':str(exc)})

# 6. Estadísticas
aciertos  = [r for r in registros if r.get('acierto')]
resueltos = [r for r in registros if 'acierto' in r]
pct       = len(aciertos)/len(resueltos)*100 if resueltos else 0
print(f"\n📊 Resultado: {len(aciertos)}/{len(resueltos)} aciertos = {pct:.1f}%")

# 7. Guardar
OUTPUT.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"✅ Guardado en {OUTPUT}")
