# Dockerfile multi-stage para minimizar imagen final
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias del sistema (build tools para sklearn/xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar deps primero para cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copiar código de la app
COPY . .

# Variables que el usuario puede override en Render/Railway/Fly:
#   PORT             — el host normalmente lo provee, default 8000
#   ALLOWED_ORIGINS  — dominios para CORS, default *
ENV PORT=8000 \
    ALLOWED_ORIGINS=*

EXPOSE 8000

# Healthcheck interno
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/api/health || exit 1

# `sh -c` para que ${PORT} se expanda
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
