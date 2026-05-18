"""
Driver v2: scrapea temporada UCL via API JSON oficial de UEFA.

Cambio clave vs v1: en lugar de iterar fechas hardcodeadas y leer el DOM del
listado (que UEFA rompió para temporadas pasadas — el SPA ignora ?season=YYYY
y redirige a 2025-26), llamamos al endpoint REST:

    https://match.uefa.com/v5/matches
        ?competitionId=1            ← UEFA Champions League
        &seasonYear=YYYY            ← año FINAL de la temporada (2025 = 2024/25)
        &phase=TOURNAMENT
        &limit=200&offset=0
        &order=ASC

Eso devuelve los 189 partidos de la temporada en un solo request, con id,
fecha, equipos, score y matchday (MD1..MD17 según formato Liga+KO).

Para cada match, lanzamos subprocess `agregar_partido.py --url ... --fase X
--si` con la URL canónica `/match/{id}/statistics/`. agregar_partido.py ya
hace dedup por (Equipo1, Equipo2, Fecha) — partidos repetidos se saltean.

Uso:
    python scripts/scraper_temporada_v2.py --season 2025          # UCL 2024-25
    python scripts/scraper_temporada_v2.py --season 2024          # UCL 2023-24
    python scripts/scraper_temporada_v2.py --season 2025 --desde 2024-11-01
    python scripts/scraper_temporada_v2.py --season 2025 --dry    # solo lista, no scrapea
"""
import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
_PROJECT_ROOT = BASE.parent
AGREGAR = BASE / "agregar_partido.py"
DATASET = _PROJECT_ROOT / "data" / "creando_dataset_modificado.xlsx"

API_URL = ("https://match.uefa.com/v5/matches"
           "?competitionId=1&seasonYear={season}&phase=TOURNAMENT"
           "&limit=200&offset=0&order=ASC")

# Mapeo matchday → fase del dataset (formato Liga + KO, válido desde 2024-25)
MD_TO_FASE_LIGA_FORMAT = {
    "MD1": "Liga", "MD2": "Liga", "MD3": "Liga", "MD4": "Liga",
    "MD5": "Liga", "MD6": "Liga", "MD7": "Liga", "MD8": "Liga",
    "MD9": "Playoffs", "MD10": "Playoffs",
    "MD11": "Octavos", "MD12": "Octavos",
    "MD13": "Cuartos", "MD14": "Cuartos",
    "MD15": "Semifinal", "MD16": "Semifinal",
    "MD17": "Final",
}
# Para 2023-24 y anteriores (formato grupos + KO)
MD_TO_FASE_GRUPOS_FORMAT = {
    "MD1": "Grupos", "MD2": "Grupos", "MD3": "Grupos",
    "MD4": "Grupos", "MD5": "Grupos", "MD6": "Grupos",
    "MD7": "Octavos", "MD8": "Octavos",
    "MD9": "Cuartos", "MD10": "Cuartos",
    "MD11": "Semifinal", "MD12": "Semifinal",
    "MD13": "Final",
}


def fetch_matches(season_year: int) -> list[dict]:
    url = API_URL.format(season=season_year)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def fase_from_match(m: dict, formato: str) -> str:
    md_name = (m.get("matchday") or {}).get("name", "")
    table = MD_TO_FASE_LIGA_FORMAT if formato == "liga" else MD_TO_FASE_GRUPOS_FORMAT
    return table.get(md_name, "Liga")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True,
                   help="Año final de la temporada (2025 = UCL 2024-25, 2024 = 2023-24)")
    p.add_argument("--formato", choices=["liga", "grupos", "auto"], default="auto",
                   help="Formato del torneo. 'auto' = liga si season ≥ 2025")
    p.add_argument("--desde", help="Reanudar desde fecha YYYY-MM-DD")
    p.add_argument("--dry", action="store_true", help="Solo lista, no scrapea")
    args = p.parse_args()

    formato = args.formato
    if formato == "auto":
        formato = "liga" if args.season >= 2025 else "grupos"

    print(f"📡 Llamando API UEFA para season={args.season} (formato={formato})…")
    matches = fetch_matches(args.season)
    print(f"   {len(matches)} partidos en la respuesta")

    matches.sort(key=lambda m: m["kickOffTime"]["date"])

    if args.desde:
        matches = [m for m in matches if m["kickOffTime"]["date"] >= args.desde]
        print(f"   filtrados a {len(matches)} desde {args.desde}")

    # Resumen por fase
    from collections import Counter
    fases = Counter(fase_from_match(m, formato) for m in matches)
    print(f"   resumen por fase: {dict(fases)}")
    print()

    if args.dry:
        print("=== DRY RUN — primeros 8 partidos a procesar ===")
        for m in matches[:8]:
            home = m["homeTeam"]["internationalName"]
            away = m["awayTeam"]["internationalName"]
            s = m["score"]["total"]
            f = m["kickOffTime"]["date"]
            fase = fase_from_match(m, formato)
            print(f"  {f}  {home} {s['home']}-{s['away']} {away}  "
                  f"id={m['id']} fase={fase}")
        return

    n_antes = len(pd.read_excel(DATASET))
    print(f"Dataset antes: {n_antes} partidos\n")

    errores = []
    for i, m in enumerate(matches, 1):
        home = m["homeTeam"]["internationalName"]
        away = m["awayTeam"]["internationalName"]
        s = m["score"]["total"]
        fecha = m["kickOffTime"]["date"]
        fase = fase_from_match(m, formato)
        url = f"https://es.uefa.com/uefachampionsleague/match/{m['id']}/statistics/"

        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(matches)}] {fecha} · {home} {s['home']}-{s['away']} {away}  "
              f"(fase={fase}, id={m['id']})")
        print(f"{'=' * 60}")

        cmd = [sys.executable, "-u", str(AGREGAR),
               "--url", url, "--fase", fase, "--si"]
        try:
            r = subprocess.run(cmd, check=False, timeout=180)
            if r.returncode != 0:
                errores.append((m["id"], home, away, f"exit {r.returncode}"))
        except KeyboardInterrupt:
            print("\nInterrumpido por usuario.")
            break
        except Exception as e:
            errores.append((m["id"], home, away, str(e)))
            print(f"  ERROR: {e}")

    n_despues = len(pd.read_excel(DATASET))
    print(f"\n{'=' * 60}")
    print(f"Temporada {args.season} completada.")
    print(f"Dataset: {n_antes} → {n_despues}  ({n_despues - n_antes:+d} partidos)")
    if errores:
        print(f"\n⚠️  {len(errores)} errores:")
        for mid, h, a, e in errores[:20]:
            print(f"   id={mid}  {h} vs {a}  → {e}")


if __name__ == "__main__":
    main()
