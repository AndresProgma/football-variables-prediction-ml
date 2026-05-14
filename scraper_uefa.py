"""
Scraper de UEFA.com para automatizar la carga de partidos al dataset.

Funciones públicas:
    obtener_info_partido(url, headless=True) -> dict
        Descarga la página de stats de un partido y devuelve un dict con:
            - texto_stats : str  (formato compatible con parsear_stats() de agregar_partido.py)
            - match_id    : str
            - equipo1, equipo2 : str (ya mapeados a los nombres del dataset)
            - fecha       : str (YYYY-MM-DD)
            - goles_e1, goles_e2 : int|None
            - url         : str

    listar_partidos_por_fecha(fecha, headless=True) -> list[str]
        Devuelve la lista de URLs de stats de todos los partidos jugados en una fecha.

Requiere:
    pip install playwright
    playwright install chromium
"""

import re
import sys
import html as html_module
from datetime import datetime

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except ImportError:
    print("ERROR: Playwright no está instalado.")
    print("Ejecuta: pip install playwright && playwright install chromium")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Mapeo de nombres UEFA → nombres exactos del dataset
# ---------------------------------------------------------------------------
# El dataset usa nombres cortos / con typos históricos. UEFA muestra los
# nombres completos. Esta tabla convierte uno al otro.
MAPEO_EQUIPOS = {
    'atletico de madrid': 'Atleti',
    'atletico madrid': 'Atleti',
    'atletico': 'Atleti',
    'atleti': 'Atleti',
    'borussia dortmund': 'BDortmund',
    'dortmund': 'BDortmund',
    'bayer leverkusen': 'Leverkursen',
    'leverkusen': 'Leverkursen',
    'bayern munich': 'Bayern Munchen',
    'bayern munchen': 'Bayern Munchen',
    'bayern munich fc': 'Bayern Munchen',
    'fc bayern munich': 'Bayern Munchen',
    'manchester city': 'Man City',
    'man city': 'Man City',
    'paris saint-germain': 'Paris',
    'paris saint germain': 'Paris',
    'psg': 'Paris',
    'paris': 'Paris',
    'fc copenhagen': 'Copengagen',
    'copenhagen': 'Copengagen',
    'kobenhavn': 'Copengagen',
    'eintracht frankfurt': 'Frankfurt',
    'frankfurt': 'Frankfurt',
    'real madrid': 'Real Madrid',
    'real madrid cf': 'Real Madrid',
    'inter milan': 'Inter',
    'internazionale': 'Inter',
    'inter': 'Inter',
    'fc internazionale milano': 'Inter',
    'fc barcelona': 'Barcelona',
    'barcelona': 'Barcelona',
    'olympique de marseille': 'Marseille',
    'olympique marseille': 'Marseille',
    'marseille': 'Marseille',
    'as monaco': 'Monaco',
    'monaco': 'Monaco',
    'union sg': 'Union SG',
    'union saint-gilloise': 'Union SG',
    'royale union saint-gilloise': 'Union SG',
    'slavia prague': 'Slavia Praha',
    'slavia praha': 'Slavia Praha',
    'sk slavia praha': 'Slavia Praha',
    'pafos fc': 'Pafos',
    'pafos': 'Pafos',
    'qarabag fk': 'Qarabag',
    'qarabag': 'Qarabag',
    'fk bodo glimt': 'Bodo',
    'bodo glimt': 'Bodo',
    'bodo/glimt': 'Bodo',
    'bodo': 'Bodo',
    'kairat almaty': 'Kairat Almaty',
    'fc kairat': 'Kairat Almaty',
    'fc kairat almaty': 'Kairat Almaty',
    'kairat': 'Kairat Almaty',
    'tottenham hotspur': 'Tottenham',
    'tottenham': 'Tottenham',
    'newcastle united': 'Newcastle',
    'newcastle': 'Newcastle',
    'club brugge': 'Club Brugge',
    'club brugge kv': 'Club Brugge',
    'galatasaray': 'Galatasaray',
    'galatasaray sk': 'Galatasaray',
    'ajax': 'Ajax',
    'ajax amsterdam': 'Ajax',
    'afc ajax': 'Ajax',
    'olympiacos': 'Olympiacos',
    'olympiacos fc': 'Olympiacos',
    'olympiakos': 'Olympiacos',
    'benfica': 'Benfica',
    'sl benfica': 'Benfica',
    'sporting cp': 'Sporting CP',
    'sporting clube de portugal': 'Sporting CP',
    'sporting': 'Sporting CP',
    'athletic club': 'Athletic Club',
    'athletic bilbao': 'Athletic Club',
    'villarreal': 'Villareal',
    'villarreal cf': 'Villareal',
    'atalanta': 'Atlanta',
    'atalanta bc': 'Atlanta',
    'juventus': 'Juventus',
    'juventus fc': 'Juventus',
    'arsenal': 'Arsenal',
    'arsenal fc': 'Arsenal',
    'chelsea': 'Chelsea',
    'chelsea fc': 'Chelsea',
    'liverpool': 'Liverpool',
    'liverpool fc': 'Liverpool',
    'psv': 'PSV',
    'psv eindhoven': 'PSV',
    'napoli': 'Napoli',
    'ssc napoli': 'Napoli',
}


def normalizar_nombre_equipo(nombre):
    """Mapea un nombre UEFA al nombre exacto del dataset."""
    if not nombre:
        return nombre
    clean = nombre.lower().strip()
    if clean in MAPEO_EQUIPOS:
        return MAPEO_EQUIPOS[clean]
    for k, v in MAPEO_EQUIPOS.items():
        if k in clean or clean in k:
            return v
    return nombre


def extraer_match_id(url):
    m = re.search(r'/match/(\d+)', url)
    return m.group(1) if m else None


def extraer_equipos_de_url(url):
    """Saca los nombres del slug: /match/2045960--juventus-vs-sporting-cp/..."""
    m = re.search(r'/match/\d+--([a-z0-9-]+)-vs-([a-z0-9-]+)', url, re.IGNORECASE)
    if not m:
        return None, None
    s1 = m.group(1).replace('-', ' ').strip()
    s2 = m.group(2).replace('-', ' ').strip()
    return normalizar_nombre_equipo(s1), normalizar_nombre_equipo(s2)


def _aceptar_cookies(page):
    """Cierra el banner de cookies si aparece (en página o iframe)."""
    selectores = [
        '#onetrust-accept-btn-handler',
        'button:has-text("Aceptar todo")',
        'button:has-text("Aceptar todas")',
        'button:has-text("Aceptar")',
        'button:has-text("Accept all")',
        'button:has-text("Accept")',
        '[data-testid*="accept"]',
        '[aria-label*="Aceptar"]',
        '[aria-label*="Accept"]',
    ]
    # Página principal
    for sel in selectores:
        try:
            page.locator(sel).first.click(timeout=1500)
            page.wait_for_timeout(800)
            return True
        except Exception:
            continue
    # Iframes (OneTrust suele meterse en uno)
    for frame in page.frames:
        for sel in selectores:
            try:
                frame.locator(sel).first.click(timeout=1000)
                page.wait_for_timeout(800)
                return True
            except Exception:
                continue
    return False


def _extraer_stats_del_dom(page):
    """
    Lee todos los <pk-list-stat-item> y devuelve texto en formato compatible con
    parsear_stats(): cada stat ocupa 3 líneas (valor_E1 / nombre / valor_E2).
    También funciona con acordeones colapsados, porque los datos están en atributos.
    """
    items = page.eval_on_selector_all(
        'pk-list-stat-item',
        """elements => elements.map(el => ({
            label: el.getAttribute('label'),
            data: el.getAttribute('data'),
            second: el.getAttribute('second-data'),
            type: el.getAttribute('data-stat-type'),
        }))"""
    )
    lineas = []
    for it in items:
        label = it.get('label')
        d1 = it.get('data')
        d2 = it.get('second')
        if not label or d1 is None or d2 is None:
            continue
        # Normalizar coma → punto si fuera necesario, pero el parser ya lo hace
        lineas.append(str(d1))
        lineas.append(label)
        lineas.append(str(d2))
    return '\n'.join(lineas)


def _crear_contexto(p, headless):
    """Crea un contexto de Playwright con perfil de navegador real."""
    browser = p.chromium.launch(
        headless=headless,
        args=['--disable-blink-features=AutomationControlled'],
    )
    context = browser.new_context(
        user_agent=(
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/131.0.0.0 Safari/537.36'
        ),
        locale='es-ES',
        viewport={'width': 1366, 'height': 900},
    )
    return browser, context


def obtener_info_partido(url, headless=True, debug=False):
    """
    Carga la página de stats de un partido de UEFA y devuelve la info para construir
    una fila del dataset.
    """
    if '/statistics' not in url:
        url = url.rstrip('/') + '/statistics/'

    info = {
        'url': url,
        'match_id': extraer_match_id(url),
        'equipo1': None,
        'equipo2': None,
        'fecha': None,
        'goles_e1': None,
        'goles_e2': None,
        'texto_stats': '',
    }

    with sync_playwright() as p:
        browser, context = _crear_contexto(p, headless)
        page = context.new_page()
        page.goto(url, wait_until='domcontentloaded', timeout=45000)
        # Después de redirects, la URL final tiene el slug --equipo1-vs-equipo2
        final_url = page.url
        info['url'] = final_url
        if not info['match_id']:
            info['match_id'] = extraer_match_id(final_url)
        e1_url, e2_url = extraer_equipos_de_url(final_url)
        info['equipo1'] = e1_url
        info['equipo2'] = e2_url
        _aceptar_cookies(page)

        # Esperar a que carguen las stats
        try:
            page.wait_for_selector('text=Estadísticas clave', timeout=20000)
        except PWTimeout:
            try:
                page.wait_for_selector('text=Key stats', timeout=5000)
            except PWTimeout:
                pass

        # Esperar a que la red termine de cargar las stats
        try:
            page.wait_for_load_state('networkidle', timeout=15000)
        except PWTimeout:
            pass

        # Esperar a que aparezcan los stat items custom
        try:
            page.wait_for_selector('pk-list-stat-item', timeout=10000)
            page.wait_for_timeout(800)
        except PWTimeout:
            pass

        # Extraer todas las stats directamente de los atributos del DOM
        info['texto_stats'] = _extraer_stats_del_dom(page)

        # Fallback: si no se encontraron stats custom, tirar al inner_text de main
        if not info['texto_stats'].strip():
            try:
                info['texto_stats'] = page.locator('main').first.inner_text(timeout=5000)
            except Exception:
                info['texto_stats'] = page.locator('body').inner_text()

        # Fecha y marcador desde el HTML completo (JSON embebido + aria-label).
        # UEFA mete JSONs dentro de atributos data-options con &quot; → desescapamos.
        html_full = html_module.unescape(page.content())

        # Fecha: "kickOffTime":{"date":"YYYY-MM-DD..."}
        m = re.search(r'kickOffTime"\s*:\s*\{\s*"date"\s*:\s*"(\d{4}-\d{2}-\d{2})', html_full)
        if m:
            info['fecha'] = m.group(1)

        # Marcador: aria-label=" Equipo1 - Equipo2 X-Y"  o  "name":" ... X-Y"
        m = re.search(r'aria-label="\s*[^"]+?\s(\d+)\s*[-–]\s*(\d+)\s*"', html_full)
        if not m:
            m = re.search(r'"name":"\s*[^"]+?\s(\d+)\s*[-–]\s*(\d+)\s*"', html_full)
        if m:
            info['goles_e1'] = int(m.group(1))
            info['goles_e2'] = int(m.group(2))

        if debug:
            page.screenshot(path='debug_uefa.png', full_page=True)
            with open('debug_uefa.txt', 'w', encoding='utf-8') as f:
                f.write(info['texto_stats'])
            print('[debug] Guardado debug_uefa.png y debug_uefa.txt')

        browser.close()

    return info


_MESES_ES = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
    'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
    'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
}


def _parsear_fecha_es(texto):
    """'martes 4 noviembre 2025' → '2025-11-04'."""
    if not texto:
        return None
    m = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', texto.lower())
    if not m:
        return None
    dia = int(m.group(1))
    mes_nombre = m.group(2)
    año = int(m.group(3))
    mes = _MESES_ES.get(mes_nombre)
    if not mes:
        return None
    return f'{año}-{mes:02d}-{dia:02d}'


def _season_de_fecha(fecha):
    """UCL season = año del primer partido del torneo (agosto). Sep-Dic año X
    y Ene-May año X+1 pertenecen a la temporada X."""
    y, m, _ = (int(x) for x in fecha.split('-'))
    return y if m >= 7 else y - 1


def listar_partidos_por_fecha(fecha, headless=True):
    """
    Devuelve solo las URLs de los partidos jugados en la fecha YYYY-MM-DD.
    Lee los H2 de fecha en español (ej. 'martes 4 noviembre 2025') y asocia
    cada anchor de partido al H2 anterior más cercano en el DOM.

    Para fechas pasadas se pasa ?season=YYYY a la URL (necesario para que
    UEFA cargue el calendario de esa temporada).
    """
    try:
        datetime.strptime(fecha, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Fecha inválida: {fecha!r}. Debe ser YYYY-MM-DD.")

    season = _season_de_fecha(fecha)
    url = f'https://es.uefa.com/uefachampionsleague/fixtures-results/?season={season}#/d/{fecha}'

    with sync_playwright() as p:
        browser, context = _crear_contexto(p, headless)
        page = context.new_page()
        page.goto(url, wait_until='domcontentloaded', timeout=45000)
        _aceptar_cookies(page)
        page.wait_for_timeout(3500)  # SPA: dejar que renderice

        # Recorrer H2 (fechas) y anchors de partido en orden de documento.
        # A cada anchor le asignamos el H2 anterior más cercano cuyo texto sea fecha.
        items = page.evaluate("""() => {
            const dateRe = /^(?:lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)\\s+\\d+\\s+\\w+\\s+\\d{4}$/i;
            const all = Array.from(document.querySelectorAll('h2, a[href*="/match/"]'));
            const out = [];
            let cur = null;
            for (const el of all) {
                if (el.tagName === 'H2') {
                    const t = (el.innerText || '').trim();
                    if (dateRe.test(t)) cur = t;
                } else {
                    out.push({ href: el.getAttribute('href'), dateRaw: cur });
                }
            }
            return out;
        }""")
        browser.close()

    por_id = {}
    for it in items:
        href = it.get('href') or ''
        if '/match/' not in href:
            continue
        fecha_url = _parsear_fecha_es(it.get('dateRaw') or '')
        if fecha_url != fecha:
            continue
        if not href.startswith('http'):
            href = 'https://es.uefa.com' + href
        href = re.sub(r'/(statistics|lineups|line-ups|summary|info)/?$', '', href)
        href = href.rstrip('/') + '/statistics/'
        mid = extraer_match_id(href)
        if not mid:
            continue
        es_canonica = '/uefachampionsleague/match/' in href and '--' in href
        if mid not in por_id or (es_canonica and '--' not in por_id[mid]):
            por_id[mid] = href

    return sorted(set(por_id.values()))


if __name__ == '__main__':
    # Smoke test
    if len(sys.argv) > 1:
        if sys.argv[1] == '--fecha' and len(sys.argv) > 2:
            for u in listar_partidos_por_fecha(sys.argv[2]):
                print(u)
        else:
            info = obtener_info_partido(sys.argv[1], debug=True)
            for k, v in info.items():
                if k == 'texto_stats':
                    print(f'{k}: ({len(v)} caracteres)')
                else:
                    print(f'{k}: {v}')
    else:
        print('Uso:')
        print('  python scraper_uefa.py <url-partido>')
        print('  python scraper_uefa.py --fecha YYYY-MM-DD')
