"""Offline tests for the FastAPI service: real pipeline code, fake models, no downloads.

    python serving/test_api.py
"""

import os
import sys
from pathlib import Path

os.environ["PHMSA_PRELOAD"] = "0"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

from fastapi.testclient import TestClient  # noqa: E402

from incident_extractor import IncidentExtractor  # noqa: E402
from serving import api  # noqa: E402

failures = 0


def check(name, ok, detail=""):
    global failures
    failures += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n        {detail}" if not ok else ""))


class FakeNER:
    def predict(self, sentences):
        out = []
        for s in sentences:
            ents = []
            for word, label in (("METER SET", "EQUIPMENT"), ("CORROSION", "FAILURE_MODE"), ("INJURY", "CONSEQUENCE")):
                if word in s:
                    i = s.index(word)
                    ents.append({"text": word, "label": label, "start": i, "end": i + len(word)})
            out.append(ents)
        return out


class FakeClassifier:
    def __init__(self, label):
        self.label = label

    def predict(self, texts):
        return [{"label": self.label, "probabilities": {self.label: 0.9, "other": 0.1}} for _ in texts]


TEXT = ("THE METER SET WAS FOUND LEAKING DUE TO EXTERNAL CORROSION. THERE WAS NO REPORTABLE INJURY. "
        "THE OPERATOR REPLACED THE METER SET AND RESTORED SERVICE.")

with TestClient(api.app) as client:
    # before models load
    r = client.get("/health")
    check("/health is up without models", r.status_code == 200 and r.json() == {"status": "ok"}, r.text)
    r = client.get("/ready")
    check("/ready is 503 before models load", r.status_code == 503, r.text)
    r = client.get("/version")
    v = r.json()
    check("/version reports the pinned manifest", r.status_code == 200 and v["model_repo"] and "pinned_revision" in v
          and v["expected_scores"]["severity"]["macro_f1"] > 0, r.text)

    # failed load surfaces as 503, not a crash
    api.state.extractor, api.state.error = None, None
    old_env = os.environ.get("PHMSA_MODEL_DIR")
    os.environ["PHMSA_MODEL_DIR"] = str(ROOT / "does-not-exist")
    r = client.post("/extract", json={"narrative": TEXT})
    check("missing models give 503 with the reason", r.status_code == 503 and "not loaded" in r.text, r.text)
    if old_env is None:
        os.environ.pop("PHMSA_MODEL_DIR")

    # inject the fake pipeline
    api.state.extractor = IncidentExtractor(FakeNER(), FakeClassifier("minor"), FakeClassifier("CORROSION FAILURE"))
    api.state.source = "fake-models@test"
    r = client.get("/ready")
    check("/ready is 200 once models are in", r.status_code == 200 and r.json()["models"] == "fake-models@test", r.text)

    r = client.post("/extract", json={"narrative": TEXT, "commodity_type": "gas_distribution"})
    body = r.json()
    check("/extract returns 200", r.status_code == 200, r.text)
    labels = sorted(e["label"] for e in body.get("entities", []))
    check("entities come back with original offsets",
          labels == ["EQUIPMENT", "EQUIPMENT", "FAILURE_MODE"]
          and all(TEXT[e["start"]:e["end"]] == e["text"] for e in body["entities"]), labels)
    check("negated INJURY is dropped, and reported", [d["text"] for d in body["dropped_negated"]] == ["INJURY"],
          body.get("dropped_negated"))
    check("severity and cause present", body["severity"]["label"] == "minor"
          and body["cause"]["label"] == "CORROSION FAILURE", body)
    check("revision and latency echoed", body["model_revision"] == "fake-models@test" and body["latency_ms"] >= 0)
    check("commodity echoed", body["commodity_type"] == "gas_distribution")

    r = client.post("/extract", json={"narrative": "PIPE LEAK."})
    body = r.json()
    check("short narrative: no severity/cause, with a note",
          r.status_code == 200 and body["severity"] is None and body["cause"] is None and "fewer than" in body["note"], body)

    r = client.post("/extract/batch", json={"items": [{"narrative": TEXT}, {"narrative": TEXT.lower()},
                                                      {"narrative": "PIPE LEAK."}]})
    body = r.json()
    check("/extract/batch keeps order and count",
          r.status_code == 200 and len(body["results"]) == 3 and body["results"][2]["severity"] is None, r.text[:300])

    # validation
    check("empty narrative is 422", client.post("/extract", json={"narrative": ""}).status_code == 422)
    check("missing field is 422", client.post("/extract", json={"text": TEXT}).status_code == 422)
    check("oversized narrative is 422",
          client.post("/extract", json={"narrative": "A" * (api.MAX_CHARS + 1)}).status_code == 422)
    check("batch over the limit is 422",
          client.post("/extract/batch", json={"items": [{"narrative": TEXT}] * (api.MAX_BATCH + 1)}).status_code == 422)
    check("empty batch is 422", client.post("/extract/batch", json={"items": []}).status_code == 422)
    check("OpenAPI schema served", client.get("/openapi.json").status_code == 200)

print(f"\n{'All API tests passed' if not failures else f'{failures} API test(s) FAILED'}")
sys.exit(1 if failures else 0)
