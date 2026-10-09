"""
Hybrid RAG over extracted PHMSA incidents.

Pipeline for each question:
  1. LLM turns the question into structured filters (state, system type, cause, years)
     plus a short search phrase. Values are validated against what exists in the data.
  2. Filters are applied with pandas over ALL incidents -> exact statistics.
  3. Semantic search over narrative chunks, restricted to the filtered incidents.
  4. LLM writes the answer from the statistics + top reports, citing [report_id].
"""
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from openai import AuthenticationError, OpenAI, RateLimitError
from sentence_transformers import SentenceTransformer

from rag.config import (
    EMBED_MODEL, EXTRA_TEXT_COLS, INDEX_REPO, MAX_CHARS_PER_REPORT,
    MAX_REPORTS_IN_CONTEXT, QUERY_PREFIX, STAT_COLS,
)

FILTER_PROMPT = """You convert questions about US pipeline incidents into database filters.
Return ONLY a JSON object with exactly these keys:
  "state": list of 2-letter codes taken from ALLOWED_STATES, or an empty list
  "system_type": list of values taken from ALLOWED_SYSTEM_TYPES, or an empty list
  "cause": list of values taken from ALLOWED_CAUSES, or an empty list
  "year_min": integer or null
  "year_max": integer or null
  "search_text": a short phrase describing what to look for in incident narratives
Rules:
- Leave a field empty/null when the question does not constrain it.
- Copy allowed values exactly as written.
- "since 2015" means year_min=2015. "before 2015" means year_max=2014. "in 2018" means year_min=2018 and year_max=2018.
- Data covers years {year_lo} to {year_hi}.

ALLOWED_STATES: {states}
ALLOWED_SYSTEM_TYPES: {system_types}
ALLOWED_CAUSES: {causes}

Question: {question}"""

ANSWER_SYSTEM = """You are an analyst answering questions about US PHMSA pipeline incident reports.
Rules:
- Use only the STATISTICS and REPORTS provided. Do not use outside knowledge about specific incidents.
- Every count, total or trend must come from STATISTICS, which cover ALL matching incidents.
  REPORTS are only the most relevant sample, so never count them to answer "how many".
- Cite reports inline as [report_id] for every claim drawn from a narrative.
- If the provided material is not enough to answer, say so plainly.
- Start with a direct 2-3 sentence answer, then supporting detail grouped by theme."""


@dataclass
class Filters:
    state: list = field(default_factory=list)
    system_type: list = field(default_factory=list)
    cause: list = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    search_text: str = ""

    def describe(self) -> str:
        parts = []
        if self.state: parts.append(f"state in {self.state}")
        if self.system_type: parts.append(f"system type in {self.system_type}")
        if self.cause: parts.append(f"cause in {self.cause}")
        if self.year_min is not None: parts.append(f"year >= {self.year_min}")
        if self.year_max is not None: parts.append(f"year <= {self.year_max}")
        return "; ".join(parts) or "none (all incidents)"


@dataclass
class Result:
    answer: str
    filters: Filters
    stats: dict
    sources: pd.DataFrame

    def to_dict(self) -> dict:
        src = self.sources.copy()
        if "year" in src:
            src["year"] = src["year"].astype(object).where(src["year"].notna(), None)
        return {
            "answer": self.answer,
            "filters": asdict(self.filters),
            "stats": self.stats,
            "sources": src.to_dict(orient="records"),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Result":
        stats = dict(d["stats"])
        if "by_year" in stats:  # JSON turns int keys into strings
            stats["by_year"] = {int(k): v for k, v in stats["by_year"].items()}
        return cls(
            answer=d["answer"],
            filters=Filters(**d["filters"]),
            stats=stats,
            sources=pd.DataFrame(d["sources"]),
        )


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _to_int(v):
    try:
        return int(v) if v is not None and str(v).strip() != "" else None
    except (TypeError, ValueError):
        return None


def _tally(series: pd.Series, top: int = 10) -> dict:
    counts = series.explode().dropna().astype(str).value_counts().head(top)
    return {k: int(v) for k, v in counts.items()}


class RAGEngine:
    def __init__(self, api_key: str, model: str, base_url: str | None = None,
                 index_dir: str | None = None, answer_model: str | None = None):
        """model: used for question -> filters. answer_model: used to write the answer
        (defaults to model). Two different models = two separate free-tier daily quotas."""
        files = ["incidents.parquet", "chunks.parquet", "embeddings.npy"]
        if index_dir:
            paths = {f: Path(index_dir) / f for f in files}
        else:
            paths = {f: hf_hub_download(INDEX_REPO, f, repo_type="dataset") for f in files}

        self.incidents = pd.read_parquet(paths["incidents.parquet"])
        self.incidents["report_id"] = self.incidents["report_id"].astype(str)
        self.incidents["year"] = pd.to_numeric(self.incidents["year"], errors="coerce")
        self.by_id = self.incidents.set_index("report_id")

        self.chunks = pd.read_parquet(paths["chunks.parquet"]).reset_index(drop=True)
        self.chunks["report_id"] = self.chunks["report_id"].astype(str)
        self.emb = np.load(paths["embeddings.npy"])
        if len(self.emb) != len(self.chunks):
            raise ValueError("embeddings.npy and chunks.parquet are out of sync; rebuild the index")

        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.llm = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.answer_model = answer_model or model

        self.allowed = {
            c: sorted(self.incidents[c].dropna().astype(str).unique().tolist())
            for c in ("state", "system_type", "cause")
        }
        years = self.incidents["year"].dropna()
        self.year_range = (int(years.min()), int(years.max()))

    # ---------- LLM ----------
    def _chat(self, messages: list[dict], json_mode: bool = False, model: str | None = None) -> str:
        kwargs = {"model": model or self.model, "messages": messages, "temperature": 0}
        if json_mode:
            try:
                resp = self.llm.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs)
                return resp.choices[0].message.content or ""
            except (RateLimitError, AuthenticationError):
                raise  # quota/key problems: retrying without JSON mode would just fail again
            except Exception:
                pass  # provider may not support JSON mode; fall through to plain call
        resp = self.llm.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    # ---------- 1. question -> filters ----------
    def _match(self, key: str, values) -> list[str]:
        if isinstance(values, str):
            values = [values]
        out = []
        for v in values or []:
            v = str(v).strip().upper()
            exact = [a for a in self.allowed[key] if a.upper() == v]
            if exact:
                out += exact
            elif key != "state":  # partial match for long category names only
                out += [a for a in self.allowed[key] if v and (v in a.upper() or a.upper() in v)]
        return sorted(set(out))

    def parse_filters(self, question: str) -> Filters:
        prompt = FILTER_PROMPT.format(
            year_lo=self.year_range[0], year_hi=self.year_range[1],
            states=", ".join(self.allowed["state"]),
            system_types=", ".join(self.allowed["system_type"]),
            causes=", ".join(self.allowed["cause"][:80]),
            question=question,
        )
        raw = _extract_json(self._chat([{"role": "user", "content": prompt}], json_mode=True))
        return Filters(
            state=self._match("state", raw.get("state")),
            system_type=self._match("system_type", raw.get("system_type")),
            cause=self._match("cause", raw.get("cause")),
            year_min=_to_int(raw.get("year_min")),
            year_max=_to_int(raw.get("year_max")),
            search_text=str(raw.get("search_text") or question),
        )

    # ---------- 2. filters -> subset + exact stats ----------
    def apply_filters(self, f: Filters) -> pd.DataFrame:
        d = self.incidents
        m = pd.Series(True, index=d.index)
        if f.state: m &= d["state"].isin(f.state)
        if f.system_type: m &= d["system_type"].isin(f.system_type)
        if f.cause: m &= d["cause"].isin(f.cause)
        if f.year_min is not None: m &= d["year"] >= f.year_min
        if f.year_max is not None: m &= d["year"] <= f.year_max
        return d[m]

    def compute_stats(self, sub: pd.DataFrame) -> dict:
        stats = {"matching_incidents": int(len(sub))}
        if len(sub):
            by_year = sub["year"].dropna().astype(int).value_counts().sort_index()
            stats["by_year"] = {int(k): int(v) for k, v in by_year.items()}
            stats["by_cause"] = _tally(sub["cause"])
            stats["by_state"] = _tally(sub["state"])
            for c in STAT_COLS:
                stats[f"by_{c}"] = _tally(sub[c])
        return stats

    # ---------- 3. semantic retrieval inside the subset ----------
    def retrieve(self, sub: pd.DataFrame, query: str) -> pd.DataFrame:
        mask = self.chunks["report_id"].isin(set(sub["report_id"])).to_numpy()
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return pd.DataFrame(columns=["report_id", "score"])

        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = self.emb[idx].astype(np.float32) @ q.astype(np.float32)

        picked, seen = [], set()
        for p in np.argsort(-scores):
            rid = self.chunks.at[idx[p], "report_id"]
            if rid in seen:
                continue
            seen.add(rid)
            picked.append({"report_id": rid, "score": round(float(scores[p]), 3)})
            if len(picked) >= MAX_REPORTS_IN_CONTEXT:
                break
        return pd.DataFrame(picked)

    # ---------- 4. answer ----------
    def _report_block(self, rid: str) -> str:
        r = self.by_id.loc[rid]
        year = "" if pd.isna(r["year"]) else int(r["year"])
        lines = [f"[{rid}] {r['state']} | {year} | {r['system_type']} | cause: {r['cause']}"]
        for c in EXTRA_TEXT_COLS:
            v = r[c]
            if isinstance(v, (list, tuple, np.ndarray)):
                v = ", ".join(map(str, v))
            if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v):
                lines.append(f"{c}: {v}")
        lines.append("Narrative: " + str(r["narrative"])[:MAX_CHARS_PER_REPORT])
        return "\n".join(lines)

    def ask(self, question: str) -> Result:
        f = self.parse_filters(question)
        sub = self.apply_filters(f)
        stats = self.compute_stats(sub)

        if stats["matching_incidents"] == 0:
            return Result(
                answer=f"No incidents in the dataset match these filters: {f.describe()}. "
                       "Try widening the year range, state or cause.",
                filters=f, stats=stats, sources=pd.DataFrame(),
            )

        hits = self.retrieve(sub, f.search_text)
        reports = "\n\n".join(self._report_block(rid) for rid in hits["report_id"])
        user_msg = (
            f"QUESTION: {question}\n\n"
            f"FILTERS APPLIED: {f.describe()}\n\n"
            f"STATISTICS (exact, over all {stats['matching_incidents']} matching incidents):\n"
            f"{json.dumps(stats, indent=1)}\n\n"
            f"REPORTS (top {len(hits)} most relevant):\n{reports}"
        )
        answer = self._chat([
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user_msg},
        ], model=self.answer_model)

        sources = hits.merge(
            self.incidents[["report_id", "year", "state", "system_type", "cause", "narrative"]],
            on="report_id", how="left",
        )
        return Result(answer=answer, filters=f, stats=stats, sources=sources)
