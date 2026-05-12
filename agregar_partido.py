"""
Agrega un partido nuevo al dataset desde las estadísticas de UEFA.

Modos:
    python agregar_partido.py
        Modo manual: pregunta datos y pide que pegues las stats (flujo original).

    python agregar_partido.py --url <URL_PARTIDO>
        Scrapea automáticamente la URL de un partido de UEFA.com.

    python agregar_partido.py --fecha YYYY-MM-DD
        Scrapea todos los partidos jugados en esa fecha.

    Flags opcionales:
        --fase <Liga|Octavos|Cuartos|Semifinal|Final>   Fase por defecto.
        --no-headless                                   Muestra el navegador.
        --si                                            Auto-confirma sin preguntar.
        --debug                                         Guarda debug_uefa.png/txt.
"""

import argparse
import pandas as pd
import numpy as np
import os
import re
import subprocess
import unicodedata
import sys

# Forzar UTF-8 y line-buffering en la terminal (para ver progreso en vivo)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
else:
    sys.stdout.reconfigure(line_buffering=True)
if sys.stdin.encoding != 'utf-8':
    sys.stdin.reconfigure(encoding='utf-8')

DATASET = r"C:\Users\fehgb\OneDrive\Desktop\prediccion futybol\creando_dataset_modificado.xlsx"

SECTION_HEADERS = {
    'Estadísticas clave', 'Ataque', 'Distribución',
    'Defensa', 'Portería', 'Amonestaciones', 'Estadísticas'
}

# Mapeo: nombre UEFA → (columna_E1, columna_E2)
STAT_MAP = {
    # Goles
    'Goles':                                    ('EQUIPO1_GOLES',                        'EQUIPO2_GOLES'),
    'Goles dentro del área':                    ('Goles_dentro_area_E1',                 'Goles_dentro_area_E2'),
    'Goles fuera del área':                     ('Goles_Fuera_Area_E1',                  'Goles_Fuera_Area_E2'),
    # Disparos
    'Disparos totales':                         ('Disparos_totales_E1',                  'Disparos_totales_E2'),
    'Disparos a puerta':                        ('Disparos_a_puerta_E1',                 'Disparos_a_puerta_E2'),
    'Disparos fuera':                           ('Disparos_fuera_E1',                    'Disparos_fuera_E2'),
    'Bloqueados':                               ('Disparo_Bloqueados_E1',                'Disparo_Bloqueados_E2'),
    'Al palo':                                  ('Disparos_Al palo_E1',                  'Disparos_Al palo_E2'),
    'Larguero':                                 ('Disparos_Larguero_E1',                 'Disparos_Larguero_E2'),
    'Poste':                                    ('Disparos_Poste_E1',                    'Disparos_Poste_E2'),
    'Disparos a puerta fuera del área':         ('Disparos_a_puerta_fuera_del_area_E1',  'Disparos_a_puerta_fuera_del_area_E2'),
    'Disparos fuera desde fuera del área':      ('Disparos_fuera_desde_fuera_del_area_E1','Disparos_fuera_desde_fuera_del_area_E2'),
    # Ataque
    'Asistencias':                              ('Asistencias_E1',                       'Asistencias_E2'),
    'Penaltis marcados':                        ('Penaltis_marcados_E1',                 'Penaltis_marcados_E2'),
    'Penaltis fallados':                        ('Penaltis_fallados_E1',                 'Penaltis_fallados_E2'),
    'Penaltis forzados':                        ('Penaltis_forzados_E1',                 'Penaltis_forzados_E2'),
    'Ataques':                                  ('Ataques_E1',                           'Ataques_E2'),
    'Oportunidades claras':                     ('Oportunidades_claras_E1',              'Oportunidades_claras_E2'),
    'Saques de esquina sacados':                ('Saques_de_esquina_sacados_E1',         'Saques_de_esquina_sacados_E2'),
    'Fueras de juego':                          ('Fueras_de_juego_E1',                   'Fueras_de_juego_E2'),
    'Regates':                                  ('Regates_E1',                           'Regates_E2'),
    'Ataques en el tercio ofensivo':            ('Ataques_tercio_ofensivo_E1',           'Ataques_tercio_ofensivo_E2'),
    'Ataques en zonas clave':                   ('Ataques_zonas_clave_E1',               'Ataques_zonas_clave_E2'),
    'Carreras hacia el área':                   ('Carreras_hacia_el_area_E1',            'Carreras_hacia_el_area_E2'),
    # Distribución
    'Posesión (%)':                             ('Posesion_E1',                          'Posesion_E2'),
    'Precisión en el pase (%)':                 ('Precision_pase_E1',                    'Precision_pase_E2'),
    'Pases completados':                        ('Pases_completados_E1',                 'Pases_completados_E2'),
    'Pases realizados':                         ('Pases_realizados_E1',                  'Pases_realizados_E2'),
    'Pases en corto completados':               ('Pases_cortos_completados_E1',          'Pases_cortos_completados_E2'),
    'Pases de media distancia completados':     ('Pases_media_distancia_completados_E1', 'Pases_media_distancia_completados_E2'),
    'Pases en largo completados':               ('Pases_en_largo_completados_E1',        'Pases_en_largo_completados_E2'),
    'Pases completados atrás':                  ('Pases_completados_atras_E1',           'Pases_completados_atras_E2'),
    'Pases completados a la izquierda':         ('Pases_completadosa_izquierda_E1',      'Pases_completadosa_izquierda_E2'),
    'Pases completados a la derecha':           ('Pases_completados_derecha_E1',         'Pases_completados_derecha_E2'),
    'Libres directos sacados':                  ('Libres_directos_sacados_E1',           'Libres_directos_sacados_E2'),
    'Centros al tercio ofensivo':               ('Centros__tercio_ofensivo_E1',          'Centros__tercio_ofensivo_E2'),
    'Pases a zonas clave':                      ('Pases_zonas_clave_E1',                 'Pases_zonas_clave_E2'),
    'Pases al área':                            ('Pases_al_area_E1',                     'Pases_al_area_E2'),
    'Precisión en el centro (%)':               ('Precision_en_el_centro_E1',            'Precision_en_el_centro_E2'),
    'Centros completados':                      ('Centros_completados_E1',               'Centros_completados_E2'),
    'Centros realizados':                       ('Centros_realizados_E1',                'Centros_realizados_E2'),
    'Tiempo de posesión':                       ('Tiempo_de_posesion_E1',                'Tiempo_de_posesion_E2'),
    # Defensa
    'Balones recuperados':                      ('Balones_recuperados_E1',               'Balones_recuperados_E2'),
    'Disparos bloqueados':                      ('Bloqueos_E1',                          'Bloqueos_E2'),
    'Penaltis cometidos':                       ('Penaltis_cometidos_E1',                'Penaltis_cometidos_E2'),
    'Duelos':                                   ('Entradas_E1',                          'Entradas_E2'),
    'Entradas con éxito':                       ('Entradas_con_exito_E1',                'Entradas_con_exito_E2'),
    'Entradas perdidas':                        ('Entradas_perdidas_E1',                 'Entradas_perdidas_E2'),
    'Despejes completados':                     ('Despejes_completados_E1',              'Despejes_completados_E2'),
    'Despejes realizados':                      ('Despejes_realizados_E1',               'Despejes_realizados_E2'),
    # Portería
    'Goles encajados':                          ('goles_encajados_E1',                   'goles_encajados_E2'),
    'Goles encajados en propia puerta':         ('Goles_encajados_propia_puerta_E1',     'Goles_encajados_propia_puerta_E2'),
    'Porterías a cero':                         ('Porterias_a_cero_E1',                  'Porterias_a_cero_E2'),
    'Paradas':                                  ('Paradas_E1',                           'Paradas_E2'),
    'Paradas en libres directo':                ('Paradas_en_libres_directo_E1',         'Paradas_en_libres_directo_E2'),
    'Paradas tras libre indirecto':             ('Paradas-tras_libre_indirecto_E1',      'Paradas-tras_libre_indirecto_E2'),
    'Penaltis parados':                         ('Penaltis_parados_E1',                  'Penaltis_parados_E2'),
    'Balones blocados':                         ('Balones_blocados_E1',                  'Balones_blocados_E2'),
    'Balones blocados por arriba':              ('Balones_blocados_por arriba_E1',        'Balones_blocados_por arriba_E2'),
    'Balones blocados por abajo':               ('Balones_blocados_por_abajo_E1',         'Balones_blocados_por_abajo_E2'),
    'Despejes de puños':                        ('Despejes_de_puños_E1',                 'Despejes_de_puños_E2'),
    # Amonestaciones
    'Tarjetas amarillas':                       ('Tarjetas_amarillas_E1',                'Tarjetas_amarillas_E2'),
    'Tarjetas rojas':                           ('Tarjetas_rojas_E1',                    'Tarjetas_rojas_E2'),
    'Faltas cometidas':                         ('Faltas_cometidas_E1',                  'Faltas_cometidas_E2'),
    'Faltas cometidas en el tercio defensivo':  ('Faltas_cometidas_tercio_def_E1',       'Faltas_cometidas_tercio_def_E2'),
    'Faltas cometidas en campo propio':         ('Faltas_cometidas_en_campo_propio_E1',  'Faltas_cometidas_en_campo_propio_E2'),
}

IGNORAR = {'Distancia recorrida (km)'}


def norm(texto):
    """Quita acentos y pasa a minúsculas para comparar sin importar codificación."""
    return unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode().lower().strip()

# Mapa normalizado para búsqueda robusta
STAT_MAP_NORM = {norm(k): v for k, v in STAT_MAP.items()}
SECTION_HEADERS_NORM = {norm(h) for h in SECTION_HEADERS}
IGNORAR_NORM = {norm(i) for i in IGNORAR}


def es_numero(texto):
    try:
        float(texto.replace(',', '.'))
        return True
    except ValueError:
        return False


def parsear_stats(texto):
    """
    Parsea el texto copiado de UEFA.
    Formato: valor_E1 / nombre_stat / valor_E2  (en líneas consecutivas)
    Usa comparación normalizada para tolerar problemas de codificación.
    """
    lines = [l.strip() for l in texto.splitlines() if l.strip()]
    stats = {}
    i = 0
    while i < len(lines):
        linea = lines[i]
        linea_n = norm(linea)
        if linea_n in SECTION_HEADERS_NORM or linea_n in IGNORAR_NORM:
            i += 1
            continue

        # Patrón: número → nombre → número
        if es_numero(linea) and i + 2 < len(lines):
            val1 = float(linea.replace(',', '.'))
            nombre = lines[i + 1]
            nombre_n = norm(nombre)
            if not es_numero(nombre) and nombre_n not in SECTION_HEADERS_NORM:
                if i + 2 < len(lines) and es_numero(lines[i + 2]):
                    val2 = float(lines[i + 2].replace(',', '.'))
                    if nombre_n not in IGNORAR_NORM:
                        stats[nombre_n] = (val1, val2)
                    i += 3
                    continue
        i += 1
    return stats


def construir_fila(partido_id, fase, equipo1, equipo2, stats, columnas_dataset):
    fila = {col: np.nan for col in columnas_dataset}

    fila['Partido_id'] = partido_id
    fila['Fase'] = fase
    fila['Equipo1'] = equipo1
    fila['Equipo2'] = equipo2
    fila['Es_Local_E1'] = 1

    no_mapeados = []
    for nombre_n, (val1, val2) in stats.items():
        if nombre_n in STAT_MAP_NORM:
            col1, col2 = STAT_MAP_NORM[nombre_n]
            fila[col1] = val1
            fila[col2] = val2
        else:
            no_mapeados.append(nombre_n)

    if no_mapeados:
        print(f"\n⚠️  Estadísticas de UEFA no mapeadas al dataset: {no_mapeados}")

    return fila



def leer_texto_multilinea():
    print("Pega las estadísticas de UEFA y escribe FIN en una línea nueva cuando termines:\n")
    lineas = []
    while True:
        try:
            linea = input()
        except EOFError:
            break
        if linea.strip().upper() == 'FIN':
            break
        lineas.append(linea)
    return "\n".join(lineas)


def cargar_dataset():
    df = pd.read_excel(DATASET)
    siguiente_id = int(df['Partido_id'].max()) + 1
    return df, siguiente_id


def guardar_partido(df, fila):
    nueva_fila = pd.DataFrame([fila])
    df = pd.concat([df, nueva_fila], ignore_index=True)
    df.to_excel(DATASET, index=False)
    return df


def _normalizar_fecha(f):
    """Acepta '2025-11-4', '2025-11-04', '04/11/2025' → '2025-11-04'."""
    if f is None or (isinstance(f, float) and pd.isna(f)):
        return None
    s = str(f).strip()
    m = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})', s)
    if m:
        y, mo, d = m.groups()
        return f'{y}-{int(mo):02d}-{int(d):02d}'
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})', s)
    if m:
        d, mo, y = m.groups()
        return f'{y}-{int(mo):02d}-{int(d):02d}'
    return s


def partido_ya_existe(df, equipo1, equipo2, fecha):
    """Comprueba si ya hay una fila con los mismos equipos y fecha (normalizada)."""
    if not fecha:
        return False
    f_norm = _normalizar_fecha(fecha)
    fechas_df = df['Fecha'].map(_normalizar_fecha)
    mask = (
        (df['Equipo1'].astype(str).str.lower() == str(equipo1).lower()) &
        (df['Equipo2'].astype(str).str.lower() == str(equipo2).lower()) &
        (fechas_df == f_norm)
    )
    return bool(mask.any())


def procesar_partido(df, siguiente_id, fase, equipo1, equipo2, fecha,
                     texto_stats, auto_confirmar=False):
    """Parsea las stats, construye la fila y la guarda. Devuelve el df actualizado."""
    stats = parsear_stats(texto_stats)
    print(f"  ✓ {len(stats)} estadísticas parseadas")

    fila = construir_fila(siguiente_id, fase, equipo1, equipo2, stats, df.columns)
    fila['Fecha'] = fecha if fecha else None

    g1 = fila.get('EQUIPO1_GOLES')
    g2 = fila.get('EQUIPO2_GOLES')
    g1_str = int(g1) if pd.notna(g1) else '?'
    g2_str = int(g2) if pd.notna(g2) else '?'
    print(f"  Resultado: {equipo1} {g1_str} – {g2_str} {equipo2}  |  {fecha or 'sin fecha'}  |  Fase: {fase}")

    if not auto_confirmar:
        confirmar = input("  ¿Agregar al dataset? (s/n): ").strip().lower()
        if confirmar != 's':
            print("  ⤷ Cancelado")
            return df

    df = guardar_partido(df, fila)
    print(f"  ✅ Guardado (Partido_id={siguiente_id})")
    return df


def modo_manual():
    print("=" * 55)
    print("  AGREGAR PARTIDO AL DATASET — Champions League")
    print("=" * 55)

    df, siguiente_id = cargar_dataset()

    print(f"\nNuevo Partido_id: {siguiente_id}")
    fase    = input("Fase (ej. Grupos, Octavos, Cuartos, Semifinal, Final): ").strip()
    equipo1 = input("Equipo 1 (local / izquierda en la página): ").strip()
    equipo2 = input("Equipo 2 (visitante / derecha en la página): ").strip()

    print(f"\nPartido: {equipo1} vs {equipo2}  |  Fase: {fase}\n")

    texto = leer_texto_multilinea()
    fecha = input("\nFecha del partido (YYYY-MM-DD) o Enter para omitir: ").strip() or None

    df = procesar_partido(df, siguiente_id, fase, equipo1, equipo2, fecha, texto)

    print(f"\n   Dataset ahora tiene {len(df)} partidos.")
    print(f"   Archivo: {DATASET}")


def modo_url(url, fase, headless=True, auto_confirmar=False, debug=False):
    from scraper_uefa import obtener_info_partido

    print("=" * 55)
    print("  AGREGAR PARTIDO DESDE URL")
    print("=" * 55)
    print(f"URL: {url}\n")

    info = obtener_info_partido(url, headless=headless, debug=debug)

    df, siguiente_id = cargar_dataset()

    equipo1 = info['equipo1'] or ''
    equipo2 = info['equipo2'] or ''
    fecha   = info['fecha']
    fase_final = fase or 'Liga'

    if not auto_confirmar:
        nuevo_e1 = input(f"Equipo 1 [{equipo1}]: ").strip()
        if nuevo_e1:
            equipo1 = nuevo_e1
        nuevo_e2 = input(f"Equipo 2 [{equipo2}]: ").strip()
        if nuevo_e2:
            equipo2 = nuevo_e2
        nueva_f = input(f"Fecha [{fecha}]: ").strip()
        if nueva_f:
            fecha = nueva_f
        nueva_fase = input(f"Fase [{fase_final}]: ").strip()
        if nueva_fase:
            fase_final = nueva_fase

    if partido_ya_existe(df, equipo1, equipo2, fecha):
        print(f"⚠️  Ya existe un partido {equipo1} vs {equipo2} en {fecha}. Se omite.")
        return

    print(f"\nProcesando Partido_id {siguiente_id}: {equipo1} vs {equipo2}")
    df = procesar_partido(
        df, siguiente_id, fase_final, equipo1, equipo2, fecha,
        info['texto_stats'], auto_confirmar=auto_confirmar,
    )
    print(f"\n   Dataset ahora tiene {len(df)} partidos.")


def modo_fecha(fecha, fase, headless=True, auto_confirmar=False, debug=False):
    """
    Procesa los partidos de una fecha llamando a este mismo script en subprocess
    para cada URL. Cada partido = un proceso Python fresco, así no se acumulan
    recursos de Chromium y el proceso largo nunca se cuelga.
    """
    from scraper_uefa import listar_partidos_por_fecha

    print("=" * 55)
    print(f"  PROCESAR PARTIDOS DEL {fecha}")
    print("=" * 55)

    urls = listar_partidos_por_fecha(fecha, headless=headless)
    if not urls:
        print(f"No se encontraron partidos para {fecha}. ¿Está bien escrita la fecha?")
        return
    print(f"Encontrados {len(urls)} partidos del {fecha}:\n")
    for u in urls:
        print(f"  - {u}")
    print()

    fase_final = fase or 'Liga'
    script = os.path.abspath(__file__)

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")
        cmd = [sys.executable, '-u', script, '--url', url, '--fase', fase_final]
        if auto_confirmar:
            cmd.append('--si')
        if not headless:
            cmd.append('--no-headless')
        if debug:
            cmd.append('--debug')
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario.")
            return

    # Resumen final leyendo el dataset
    df, _ = cargar_dataset()
    print(f"\n{'=' * 55}")
    print(f"  Procesamiento completado.")
    print(f"  Dataset ahora tiene {len(df)} partidos.")
    print(f"  Archivo: {DATASET}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', help='URL de un partido de UEFA.com')
    parser.add_argument('--fecha', help='Fecha YYYY-MM-DD: procesa todos los partidos del día')
    parser.add_argument('--fase', help='Fase del torneo (Liga, Octavos, Cuartos, Semifinal, Final)')
    parser.add_argument('--no-headless', action='store_true', help='Muestra el navegador (para depurar)')
    parser.add_argument('--si', action='store_true', help='Auto-confirma cada inserción')
    parser.add_argument('--debug', action='store_true', help='Guarda debug_uefa.png y .txt en cada scrape')
    args = parser.parse_args()

    headless = not args.no_headless

    if args.url:
        modo_url(args.url, args.fase, headless=headless, auto_confirmar=args.si, debug=args.debug)
    elif args.fecha:
        modo_fecha(args.fecha, args.fase, headless=headless, auto_confirmar=args.si, debug=args.debug)
    else:
        modo_manual()


if __name__ == "__main__":
    main()
