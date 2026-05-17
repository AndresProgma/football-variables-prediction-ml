"""Scraper Microservice — FastAPI wrapper de scraper_uefa.py."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scraper_uefa import obtener_info_partido, listar_partidos_por_fecha

app = FastAPI(title="Scraper Service — UEFA")


class ScrapeRequest(BaseModel):
    url: str
    headless: bool = True


@app.post("/scrape")
def scrape(data: ScrapeRequest):
    try:
        return obtener_info_partido(data.url, headless=data.headless)
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/listar/{fecha}")
def listar(fecha: str, headless: bool = True):
    try:
        urls = listar_partidos_por_fecha(fecha, headless=headless)
        return {"fecha": fecha, "partidos": urls, "total": len(urls)}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}
