# Build from the repository root:
#   docker build -f serving/Dockerfile -t phmsa-api .
#   docker run -p 8000:8000 -v phmsa-models:/home/app/srv/.models phmsa-api
# Bake the pinned model weights into the image (bigger image, no download at start-up):
#   docker build -f serving/Dockerfile --build-arg BAKE_MODELS=1 -t phmsa-api:baked .
FROM python:3.11-slim

ARG BAKE_MODELS=0
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 app
WORKDIR /home/app/srv

# CPU-only torch (the default wheel pulls several GB of CUDA libraries)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY serving/requirements.txt serving/requirements.txt
RUN pip install -r serving/requirements.txt

# Only the code the pipeline imports, plus the pinned manifest
COPY pipeline/incident_extractor.py pipeline/
COPY relation_extraction/extract_relations.py relation_extraction/
COPY ner_model/predict.py ner_model/
COPY severity_model/severity_predict.py severity_model/
COPY cause_model/cause_predict.py cause_model/
COPY rag/__init__.py rag/config.py rag/
COPY mlops/__init__.py mlops/common.py mlops/model_manifest.json mlops/
COPY serving/__init__.py serving/api.py serving/
RUN chown -R app:app /home/app
USER app

RUN if [ "$BAKE_MODELS" = "1" ]; then \
      python -c "import sys; sys.path.insert(0,'mlops'); import common as c; print(c.download_models(c.load_manifest()))"; \
    fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl --fail http://localhost:8000/health || exit 1
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
