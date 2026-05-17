FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código fuente del pipeline (desde raíz del proyecto)
COPY knime_workflow_converter.py predecir_v2.py ./
COPY services/ml/main.py ./

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
