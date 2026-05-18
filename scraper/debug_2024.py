"""Buscar JSON embebido y endpoints reales de UEFA para listado completo."""
import re
import json
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--disable-blink-features=AutomationControlled'])
    context = browser.new_context(
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        locale='es-ES',
        viewport={'width': 1366, 'height': 800},
    )
    page = context.new_page()

    # Capturar requests de red mientras carga
    api_calls = []
    page.on('request', lambda r: api_calls.append(r.url) if 'api' in r.url or 'match' in r.url else None)

    print("=== Cargar history/seasons/2025/matches/ y capturar XHRs ===")
    page.goto("https://es.uefa.com/uefachampionsleague/history/seasons/2025/matches/",
              wait_until='networkidle', timeout=60000)
    for sel in ['#onetrust-accept-btn-handler', 'button:has-text("Aceptar")']:
        try: page.locator(sel).first.click(timeout=1500); break
        except Exception: pass
    page.wait_for_timeout(5000)

    # Filtro: API calls relevantes
    api_relevant = [u for u in api_calls
                    if ('api' in u.lower() or 'match' in u.lower())
                    and 'analytics' not in u.lower()
                    and 'cookielaw' not in u.lower()
                    and 'optanon' not in u.lower()
                    and '.png' not in u.lower() and '.jpg' not in u.lower()]
    print(f"\nXHRs relevantes ({len(api_relevant)} de {len(api_calls)} totales):")
    for u in api_relevant[:30]:
        print(f"  {u[:120]}")

    # Buscar JSON embebido (__NEXT_DATA__, __INITIAL_STATE__, etc.)
    print("\n=== Buscar JSON embebido en la página ===")
    html = page.content()
    for tag_id in ['__NEXT_DATA__', '__INITIAL_STATE__', '__APP_STATE__']:
        m = re.search(rf'<script[^>]*id="{tag_id}"[^>]*>(.*?)</script>', html, re.S)
        if m:
            try:
                data = json.loads(m.group(1))
                print(f"  ✓ Encontrado <script id='{tag_id}'>: {len(m.group(1))} bytes JSON")
                # Buscar partidos
                serialized = json.dumps(data)
                ids = set(re.findall(r'"matchId":\s*"?(\d{6,})"?', serialized))
                ids |= set(re.findall(r'/match/(\d+)', serialized))
                print(f"  IDs de partido encontrados en el JSON: {len(ids)}")
                if ids:
                    print(f"  Muestra: {sorted(ids)[:10]}")
                # Top-level keys
                if isinstance(data, dict):
                    print(f"  Top keys: {list(data.keys())[:10]}")
            except Exception as e:
                print(f"  Error parsing {tag_id}: {e}")
        else:
            print(f"  ✗ No hay <script id='{tag_id}'>")

    # Otros patrones de inyección: window.__STATE__ = {...} en script inline
    m = re.search(r'window\.(?:__\w+__|__STATE__)\s*=\s*({.+?});</script>', html, re.S)
    if m:
        print(f"\n  Encontrado window.__STATE__: {len(m.group(1))} bytes")
        ids = set(re.findall(r'/match/(\d+)', m.group(1)))
        print(f"  IDs: {len(ids)}")

    browser.close()
