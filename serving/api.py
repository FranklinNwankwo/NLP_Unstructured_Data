"""
FastAPI service for the incident extraction pipeline.

    uvicorn serving.api:app --host 0.0.0.0 --port 8000          # from the repository root
    docker build -f serving/Dockerfile -t phmsa-api . && docker run -p 8000:8000 phmsa-api

Endpoints
  GET  /health          liveness: the process is up (never touches the models)
  GET  /ready           readiness: 200 once the models are loaded, 503 before (for load balancers)
  GET  /version         pinned model repo + commit, expected scores, code version
  POST /extract         one narrative  -> entities, relations, severity, cause
  POST /extract/batch   up to 64 narratives in one call

Models: local folders if PHMSA_MODEL_DIR is set (with ner/, severity/, cause/ inside), otherwise the
revision pinned in mlops/model_manifest.json, downloaded from Hugging Face once and cached.
Loading starts in a background thread at startup (PHMSA_PRELOAD=0 turns that off; the first
request then loads them).
"""

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mlops"))
from common import load_manifest  # noqa: E402

MAX_CHARS = 20_000
MAX_BATCH = 64
API_VERSION = "1.0.0"


# ---------------------------------------------------------------- schemas

class ExtractRequest(BaseModel):
    narrative: str = Field(..., min_length=1, max_length=MAX_CHARS, description="Incident narrative text")
    commodity_type: Optional[str] = Field(None, description="gas_distribution, gas_transmission_gathering "
                                                            "or hazardous_liquid (optional, echoed back)")


class BatchRequest(BaseModel):
    items: List[ExtractRequest] = Field(..., min_length=1, max_length=MAX_BATCH)


class Span(BaseModel):
    text: str
    label: str
    start: int
    end: int
    sentence_index: Optional[int] = None
    unit_type: Optional[str] = None


class Relation(BaseModel):
    type: str
    confidence: str
    sentence_index: int
    head: Span
    tail: Span


class Classification(BaseModel):
    label: str
    probabilities: dict


class ExtractResponse(BaseModel):
    commodity_type: Optional[str] = None
    narrative: str
    entities: List[Span]
    relations: List[Relation]
    dropped_negated: List[Span]
    severity: Optional[Classification] = None
    cause: Optional[Classification] = None
    note: Optional[str] = None
    model_revision: str
    latency_ms: float


class BatchResponse(BaseModel):
    results: List[ExtractResponse]
    model_revision: str
    latency_ms: float


# ---------------------------------------------------------------- model state

class ModelState:
    def __init__(self):
        self.extractor = None
        self.source = None
        self.error = None
        self.loading = False
        self.loaded_in = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.extractor is not None:
                return self.extractor
            self.loading, t0 = True, time.time()
            try:
                from common import get_extractor
                model_dir = os.environ.get("PHMSA_MODEL_DIR")
                self.extractor, self.source = get_extractor(model_dir, load_manifest())
                self.error, self.loaded_in = None, round(time.time() - t0, 1)
            except Exception as e:                       # surfaced by /ready and the extract endpoints
                self.error = f"{type(e).__name__}: {e}"
            finally:
                self.loading = False
            return self.extractor

    def get(self):
        ext = self.extractor or self.load()
        if ext is None:
            raise HTTPException(status_code=503, detail=f"models not loaded: {self.error}")
        return ext


state = ModelState()


@asynccontextmanager
async def lifespan(_app):
    if os.environ.get("PHMSA_PRELOAD", "1") != "0" and state.extractor is None:
        threading.Thread(target=state.load, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan, title="PHMSA incident extraction API", version=API_VERSION,
              description="Entities, relations, severity and cause from PHMSA pipeline incident narratives. "
                          "A portfolio project, not for operational, safety or regulatory decisions.")


def _revision():
    return state.source or "not loaded"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if state.extractor is not None:
        return {"ready": True, "models": state.source, "load_seconds": state.loaded_in}
    raise HTTPException(status_code=503, detail={"ready": False, "loading": state.loading, "error": state.error})


@app.get("/version")
def version():
    m = load_manifest()
    return {"api_version": API_VERSION, "model_repo": m["model_repo"], "pinned_revision": m["revision"],
            "pinned_at": m.get("pinned_at"), "loaded": state.source,
            "expected_scores": {k: v["expected"] for k, v in m["models"].items()}}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    ext = state.get()
    t0 = time.perf_counter()
    rec = ext.extract(req.narrative, req.commodity_type)
    return {**rec, "model_revision": _revision(), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/extract/batch", response_model=BatchResponse)
def extract_batch(req: BatchRequest):
    ext = state.get()
    t0 = time.perf_counter()
    recs = ext.extract_many([i.narrative for i in req.items], [i.commodity_type for i in req.items])
    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"results": [{**r, "model_revision": _revision(), "latency_ms": ms} for r in recs],
            "model_revision": _revision(), "latency_ms": ms}
