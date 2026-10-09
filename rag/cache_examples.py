"""
Pre-compute answers for the demo page's example questions, so visitors clicking them
use no LLM quota. Uses 2 LLM calls per example (6 total for 3 examples).

    python -m rag.cache_examples --index-dir rag_index

Already-cached examples are skipped, so if a run stops on a quota error just run it
again later. Use --force to recompute everything (after rebuilding the index or
changing the prompts). Commit rag/example_answers.json afterwards.
"""
import argparse
import json
import os
from pathlib import Path

from rag.config import EXAMPLES, EXAMPLES_CACHE
from rag.engine import RAGEngine


def load_llm_config() -> dict:
    cfg = {}
    secrets = Path(".streamlit") / "secrets.toml"
    if secrets.exists():
        import tomllib
        cfg = tomllib.loads(secrets.read_text(encoding="utf-8"))
    for k in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "LLM_ANSWER_MODEL"):
        cfg[k] = os.environ.get(k, cfg.get(k))
    if not cfg["LLM_API_KEY"] or not cfg["LLM_MODEL"]:
        raise SystemExit("Set LLM_API_KEY and LLM_MODEL (env vars or .streamlit/secrets.toml).")
    return cfg


def main(args):
    path = Path(EXAMPLES_CACHE)
    cache = {} if args.force or not path.exists() else json.loads(path.read_text(encoding="utf-8"))
    todo = [q for q in EXAMPLES if q not in cache]
    if not todo:
        print(f"All {len(EXAMPLES)} examples already cached in {path}. Use --force to recompute.")
        return

    cfg = load_llm_config()
    engine = RAGEngine(
        api_key=cfg["LLM_API_KEY"], model=cfg["LLM_MODEL"], base_url=cfg.get("LLM_BASE_URL"),
        answer_model=cfg.get("LLM_ANSWER_MODEL"), index_dir=args.index_dir,
    )

    # Drop cached entries for questions no longer in EXAMPLES.
    cache = {q: v for q, v in cache.items() if q in EXAMPLES}
    for q in todo:
        print(f"- {q}")
        try:
            res = engine.ask(q)
        except Exception as e:
            print(f"  FAILED: {str(e)[:300]}\n  Saved what finished; run again later for the rest.")
            break
        cache[q] = res.to_dict()
        path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  cached ({res.stats['matching_incidents']} matching incidents, {len(res.sources)} sources)")

    print(f"\n{len(cache)}/{len(EXAMPLES)} examples cached in {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=None, help="Local rag_index folder; omit to download from HF")
    ap.add_argument("--force", action="store_true", help="Recompute all examples")
    main(ap.parse_args())
