"""
Para cada partido de una fecha:
  1) Predice con el dataset tal como está AHORA (antes de agregar el match)
  2) Guarda la predicción a track_record_predictions.csv
  3) Agrega el partido al dataset

Uso:
    python predecir_y_agregar.py --fecha YYYY-MM-DD [--fase Liga] [--n-runs 20]

Comportamiento:
  - Entrena el modelo UNA sola vez por fecha (todos los partidos de esa jornada
    se predicen con el mismo estado del dataset, antes de agregar ninguno).
  - El scrape de cada URL corre en un subprocess Python aislado para que
    Chromium no se acumule y nada se cuelgue.
  - La predicción y la inserción al dataset corren en el proceso padre.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
else:
    sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).parent
DATASET = BASE_DIR / "creando_dataset_modificado.xlsx"
PRED_CSV = BASE_DIR / "track_record_predictions.csv"
PREDICTIONS_SIMPLE_CSV = BASE_DIR / "predictions.csv"

# Asegurar import de módulos hermanos
sys.path.insert(0, str(BASE_DIR))


PRED_HEADERS = [
    'fecha_prediccion', 'fecha_partido', 'fase',
    'equipo1', 'equipo2',
    'partido_id_nuevo',
    'elo_e1', 'elo_e2', 'diff_elo',
    'forma_e1_w', 'forma_e1_d', 'forma_e1_l', 'forma_e1_gf', 'forma_e1_gc', 'forma_e1_pts',
    'forma_e2_w', 'forma_e2_d', 'forma_e2_l', 'forma_e2_gf', 'forma_e2_gc', 'forma_e2_pts',
    'h2h_n', 'h2h_w', 'h2h_d', 'h2h_l', 'h2h_gf', 'h2h_gc',
    # Por modelo (6 clasificadores): pred + win/draw/loss
    'rf_pred', 'rf_win', 'rf_draw', 'rf_loss',
    'gb_pred', 'gb_win', 'gb_draw', 'gb_loss',
    'lr_pred', 'lr_win', 'lr_draw', 'lr_loss',
    'svm_pred', 'svm_win', 'svm_draw', 'svm_loss',
    'xgb_pred', 'xgb_win', 'xgb_draw', 'xgb_loss',
    'knn_pred', 'knn_win', 'knn_draw', 'knn_loss',
    # Consenso
    'consenso_pred', 'consenso_win', 'consenso_draw', 'consenso_loss',
    # Regresores de goles
    'rf_goles_g1', 'rf_goles_g2', 'rf_goles_std1', 'rf_goles_std2',
    'xgb_goles_g1', 'xgb_goles_g2', 'xgb_goles_std1', 'xgb_goles_std2',
    # Resultado real (rellenado tras agregar al dataset)
    'g1_real', 'g2_real', 'resultado_real_e1', 'acierto_consenso',
    'n_runs', 'n_partidos_dataset',
]

MODEL_KEYS = {
    'Random Forest': 'rf',
    'Gradient Boosting': 'gb',
    'Logistic Regression': 'lr',
    'SVM': 'svm',
    'XGBoost': 'xgb',
    'KNN': 'knn',
}


def asegurar_header_csv():
    if not PRED_CSV.exists():
        df = pd.DataFrame(columns=PRED_HEADERS)
        df.to_csv(PRED_CSV, index=False, encoding='utf-8')


def scrape_subprocess(url, headless=True):
    """Lanza un subprocess Python que hace obtener_info_partido() y devuelve el
    JSON con la info del partido. Aislamiento total de Chromium."""
    code = (
        "import json, sys\n"
        "sys.path.insert(0, r'" + str(BASE_DIR) + "')\n"
        "from scraper_uefa import obtener_info_partido\n"
        f"info = obtener_info_partido({url!r}, headless={headless})\n"
        "print('<<<JSON>>>' + json.dumps({\n"
        "    'url': info['url'], 'match_id': info['match_id'],\n"
        "    'equipo1': info['equipo1'], 'equipo2': info['equipo2'],\n"
        "    'fecha': info['fecha'], 'goles_e1': info['goles_e1'],\n"
        "    'goles_e2': info['goles_e2'], 'texto_stats': info['texto_stats'],\n"
        "}) + '<<<END>>>')\n"
    )
    proc = subprocess.run([sys.executable, '-u', '-c', code],
                          capture_output=True, text=True, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError(f"Scrape falló:\n{proc.stderr[:1000]}")
    out = proc.stdout
    i = out.find('<<<JSON>>>')
    j = out.find('<<<END>>>')
    if i < 0 or j < 0:
        raise RuntimeError(f"No se encontró bloque JSON en la salida:\n{out[-1000:]}")
    return json.loads(out[i + len('<<<JSON>>>'):j])


def construir_fila_prediccion(pred_dict, fecha_partido, n_dataset, goles_reales):
    """Convierte el dict que devuelve predecir_partido() en una fila plana."""
    fila = {col: None for col in PRED_HEADERS}
    fila['fecha_prediccion'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fila['fecha_partido'] = fecha_partido
    fila['fase'] = pred_dict['fase']
    fila['equipo1'] = pred_dict['equipo1']
    fila['equipo2'] = pred_dict['equipo2']
    fila['elo_e1'] = pred_dict['elo_e1']
    fila['elo_e2'] = pred_dict['elo_e2']
    fila['diff_elo'] = pred_dict['diff_elo']

    for k, v in pred_dict['forma_e1'].items():
        fila[f'forma_e1_{k}'] = v
    for k, v in pred_dict['forma_e2'].items():
        fila[f'forma_e2_{k}'] = v

    h2h = pred_dict['h2h']
    fila['h2h_n']  = h2h['n']
    fila['h2h_w']  = h2h['w']
    fila['h2h_d']  = h2h['d']
    fila['h2h_l']  = h2h['l']
    fila['h2h_gf'] = h2h['gf']
    fila['h2h_gc'] = h2h['gc']

    for m in pred_dict['modelos']:
        key = MODEL_KEYS.get(m['modelo'])
        if not key:
            continue
        fila[f'{key}_pred'] = m['pred']
        fila[f'{key}_win']  = m['win']
        fila[f'{key}_draw'] = m['draw']
        fila[f'{key}_loss'] = m['loss']

    c = pred_dict['consenso']
    fila['consenso_pred'] = c['pred']
    fila['consenso_win']  = c['win']
    fila['consenso_draw'] = c['draw']
    fila['consenso_loss'] = c['loss']

    for g in pred_dict['goles']:
        if g['modelo'] == 'Random Forest':
            fila['rf_goles_g1']   = g['g1']
            fila['rf_goles_g2']   = g['g2']
            fila['rf_goles_std1'] = g['std1']
            fila['rf_goles_std2'] = g['std2']
        elif g['modelo'] == 'XGBoost':
            fila['xgb_goles_g1']   = g['g1']
            fila['xgb_goles_g2']   = g['g2']
            fila['xgb_goles_std1'] = g['std1']
            fila['xgb_goles_std2'] = g['std2']

    fila['n_runs'] = pred_dict['n_runs']
    fila['n_partidos_dataset'] = n_dataset

    if goles_reales is not None and goles_reales[0] is not None and goles_reales[1] is not None:
        g1, g2 = goles_reales
        fila['g1_real'] = g1
        fila['g2_real'] = g2
        if g1 > g2:
            fila['resultado_real_e1'] = 'Win'
        elif g1 < g2:
            fila['resultado_real_e1'] = 'Loss'
        else:
            fila['resultado_real_e1'] = 'Draw'
        fila['acierto_consenso'] = (fila['resultado_real_e1'] == c['pred'])

    return fila


def guardar_prediccion(fila):
    asegurar_header_csv()
    df_row = pd.DataFrame([fila], columns=PRED_HEADERS)
    df_row.to_csv(PRED_CSV, mode='a', header=False, index=False, encoding='utf-8')


def reapendar_predictions_csv():
    """main() de knime_workflow_converter reescribe predictions.csv con las
    predicciones del test-set. Tras correrlo, re-agregamos las predicciones
    forward ya hechas (desde track_record_predictions.csv) para no perderlas
    a lo largo de varios runs."""
    if not PRED_CSV.exists():
        return
    try:
        df_track = pd.read_csv(PRED_CSV, encoding='utf-8')
    except Exception as e:
        print(f"   ⚠️  No se pudo leer {PRED_CSV.name}: {e}")
        return
    if df_track.empty:
        return
    cols_simple = ['Equipo1', 'Equipo2', 'Resultado_Real',
                   'Random Forest', 'Gradient Boosting', 'Logistic Regression',
                   'SVM', 'XGBoost', 'KNN']
    forward = pd.DataFrame({
        'Equipo1': df_track['equipo1'],
        'Equipo2': df_track['equipo2'],
        'Resultado_Real': df_track['resultado_real_e1'].fillna(''),
        'Random Forest':       df_track['rf_pred'].fillna(''),
        'Gradient Boosting':   df_track['gb_pred'].fillna(''),
        'Logistic Regression': df_track['lr_pred'].fillna(''),
        'SVM':                 df_track['svm_pred'].fillna(''),
        'XGBoost':             df_track['xgb_pred'].fillna(''),
        'KNN':                 df_track['knn_pred'].fillna(''),
    }, columns=cols_simple)
    forward.to_csv(PREDICTIONS_SIMPLE_CSV, mode='a', header=False,
                   index=False, encoding='utf-8')
    print(f"   🔁 Re-apendadas {len(forward)} predicciones forward a {PREDICTIONS_SIMPLE_CSV.name}")


def guardar_prediccion_simple(fila):
    """Append a una fila al predictions.csv con el formato simple histórico:
    Equipo1, Equipo2, Resultado_Real, Random Forest, Gradient Boosting,
    Logistic Regression, SVM, XGBoost, KNN
    """
    columnas = ['Equipo1', 'Equipo2', 'Resultado_Real',
                'Random Forest', 'Gradient Boosting', 'Logistic Regression',
                'SVM', 'XGBoost', 'KNN']
    row = {
        'Equipo1': fila['equipo1'],
        'Equipo2': fila['equipo2'],
        'Resultado_Real': fila.get('resultado_real_e1') or '',
        'Random Forest':       fila.get('rf_pred')  or '',
        'Gradient Boosting':   fila.get('gb_pred')  or '',
        'Logistic Regression': fila.get('lr_pred')  or '',
        'SVM':                 fila.get('svm_pred') or '',
        'XGBoost':             fila.get('xgb_pred') or '',
        'KNN':                 fila.get('knn_pred') or '',
    }
    write_header = not PREDICTIONS_SIMPLE_CSV.exists() or PREDICTIONS_SIMPLE_CSV.stat().st_size == 0
    pd.DataFrame([row], columns=columnas).to_csv(
        PREDICTIONS_SIMPLE_CSV, mode='a', header=write_header,
        index=False, encoding='utf-8'
    )


def procesar_fecha(fecha, fase='Liga', n_runs=20, headless=True):
    from scraper_uefa import listar_partidos_por_fecha
    from knime_workflow_converter import main as entrenar_main, predecir_partido
    import agregar_partido as agp

    print("=" * 60)
    print(f"  PREDECIR + AGREGAR — fecha {fecha}  |  fase {fase}")
    print("=" * 60)

    # 1. URLs del día
    urls = listar_partidos_por_fecha(fecha, headless=headless)
    if not urls:
        print(f"No se encontraron partidos para {fecha}.")
        return
    print(f"Encontrados {len(urls)} partidos:")
    for u in urls:
        print(f"  - {u}")

    # 2. Entrenar UNA sola vez con el dataset actual (antes de agregar nada)
    print(f"\n🧠 Entrenando modelos con el dataset actual ({DATASET.name})...")
    n_antes = len(pd.read_excel(DATASET))
    results = entrenar_main(str(DATASET))
    print(f"   Modelos entrenados sobre {n_antes} partidos.\n")

    # 2b. main() acaba de reescribir predictions.csv con las predicciones del
    # test-set. Re-apendeamos las predicciones forward acumuladas para no
    # perderlas a través de varios runs.
    reapendar_predictions_csv()

    # 3. Por cada URL: scrape (subprocess) → predecir → guardar → agregar
    for i, url in enumerate(urls, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(urls)}] {url}")
        print(f"{'─' * 60}")

        try:
            info = scrape_subprocess(url, headless=headless)
        except Exception as e:
            print(f"❌ Error scrapeando {url}: {e}")
            continue

        equipo1 = info['equipo1'] or ''
        equipo2 = info['equipo2'] or ''
        fecha_p = info['fecha'] or fecha
        if not equipo1 or not equipo2:
            print(f"⚠️  No se pudo extraer equipos del URL. Skip.")
            continue

        # Si el partido ya está en el dataset, NO predecir (contaminaría track
        # record con leakage) NI agregar (duplicaría). Sólo skip.
        df_check, _ = agp.cargar_dataset()
        if agp.partido_ya_existe(df_check, equipo1, equipo2, fecha_p):
            print(f"⏭️   {equipo1} vs {equipo2} ya está en el dataset → skip (no predict, no add)")
            continue

        print(f"\n📅 {equipo1} vs {equipo2}  |  {fecha_p}")

        # Parsear stats temprano para tener fallback de goles si el scraper
        # no extrajo el aria-label (pasó con algunos partidos).
        stats = agp.parsear_stats(info['texto_stats'])
        goles_reales = (info['goles_e1'], info['goles_e2'])
        if (goles_reales[0] is None or goles_reales[1] is None) and 'goles' in stats:
            g1g, g2g = stats['goles']
            goles_reales = (int(g1g), int(g2g))
            print(f"   ℹ️  Goles tomados del bloque de stats: {goles_reales[0]}–{goles_reales[1]}")

        # Predicción (capturamos stdout para que quede en la consola igual)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                pred = predecir_partido(equipo1, equipo2, results,
                                        n_runs=n_runs, fase=fase)
            sys.stdout.write(buf.getvalue())
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Guardar predicción
        fila = construir_fila_prediccion(
            pred, fecha_p, n_antes, goles_reales,
        )
        guardar_prediccion(fila)
        guardar_prediccion_simple(fila)
        print(f"💾 Predicción guardada en {PRED_CSV.name} y {PREDICTIONS_SIMPLE_CSV.name}")

        # Agregar al dataset (reusa lógica de agregar_partido.py)
        df, siguiente_id = agp.cargar_dataset()
        if agp.partido_ya_existe(df, equipo1, equipo2, fecha_p):
            print(f"⚠️  Ya existe {equipo1} vs {equipo2} en {fecha_p}. No se agrega de nuevo.")
            continue

        fila_partido = agp.construir_fila(siguiente_id, fase, equipo1, equipo2,
                                          stats, df.columns)
        fila_partido['Fecha'] = fecha_p
        df = agp.guardar_partido(df, fila_partido)
        g1, g2 = info['goles_e1'], info['goles_e2']
        g1_str = int(g1) if g1 is not None else '?'
        g2_str = int(g2) if g2 is not None else '?'
        print(f"✅ Agregado Partido_id={siguiente_id}: "
              f"{equipo1} {g1_str}–{g2_str} {equipo2}  ({len(df)} en total)")

    # Resumen
    df_final = pd.read_excel(DATASET)
    print(f"\n{'=' * 60}")
    print(f"  Fecha {fecha} completada.")
    print(f"  Dataset: {len(df_final)} partidos.")
    print(f"  Predicciones acumuladas en: {PRED_CSV}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--fecha', required=True,
                        help='Fecha YYYY-MM-DD a procesar')
    parser.add_argument('--fase', default='Liga',
                        help='Fase del torneo (Liga, Octavos, Cuartos, Semifinal, Final)')
    parser.add_argument('--n-runs', type=int, default=20,
                        help='Cantidad de seeds por modelo en la predicción (default 20)')
    parser.add_argument('--no-headless', action='store_true',
                        help='Mostrar Chromium (debug)')
    args = parser.parse_args()

    procesar_fecha(args.fecha, fase=args.fase, n_runs=args.n_runs,
                   headless=not args.no_headless)


if __name__ == '__main__':
    main()
