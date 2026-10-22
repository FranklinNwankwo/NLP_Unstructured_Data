"""
Builds the reference profile the drift report compares new batches against: the full pipeline
run on the frozen test narratives (data the models never trained on, from the same period).

    python mlops/build_reference.py                       # pinned models from Hugging Face
    python mlops/build_reference.py --model-dir local     # the repo's own model folders, no download
    python mlops/build_reference.py --n 500               # faster: a seeded sample of 500 narratives

The PHMSA-derived labels stored with each narrative (severity from the fatality / injury / fire /
explosion flags, the coded apparent cause) give the reference agreement rates, which equal the
models' test accuracy. Writes mlops/reference_profile.json; commit it.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DOCS_TEST, REFERENCE_PATH, ROOT, get_extractor, load_manifest, profile, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir")
ap.add_argument("--n", type=int, default=0, help="sample size (default: all test narratives)")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--out", default=str(REFERENCE_PATH))
args = ap.parse_args()

docs = read_jsonl(DOCS_TEST)
if args.n and args.n < len(docs):
    docs = random.Random(42).sample(docs, args.n)
m = load_manifest()
extractor, source = get_extractor(args.model_dir, m)

records, t0 = [], time.time()
for i in range(0, len(docs), args.batch):
    chunk = docs[i:i + args.batch]
    records += extractor.extract_many([d["text"] for d in chunk], [d["commodity_type"] for d in chunk])
    print(f"  {min(i + args.batch, len(docs)):,} / {len(docs):,}  ({time.time() - t0:.0f}s)", end="\r")
print()

ref = profile(records, [d["severity"] for d in docs], [d.get("cause") for d in docs])
ref["source"] = "eval/frozen/docs_test.jsonl" + (f" (seeded sample of {args.n})" if args.n else "")
ref["models"] = source
Path(args.out).write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
print(f"wrote {Path(args.out).relative_to(ROOT) if Path(args.out).is_relative_to(ROOT) else args.out}: "
      f"{ref['n_narratives']} narratives, zero-entity share {ref['zero_entity_share']:.1%}, "
      f"severity agreement {ref.get('severity_agreement') or 0:.3f}, cause agreement {ref.get('cause_agreement') or 0:.3f}")
