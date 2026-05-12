"""Ingest: pull HF dataset, filter, normalize topics, write to seed table."""
from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
from datasets import load_dataset
from rich.console import Console
from sklearn.cluster import KMeans

from .config import Config
from .db import Store
from .llm import EmbeddingClient, LLMClient
from .trace import trace_event

console = Console()

_IMG_MD = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_FIG_LANG = re.compile(
    r"\b(as\s+shown|in\s+the\s+figure|in\s+the\s+drawing|the\s+diagram|see\s+the\s+figure)\b",
    re.I,
)


def _seed_id(question: str, answer: str) -> str:
    return hashlib.sha256(f"{question}\x1e{answer}".encode()).hexdigest()[:16]


def _is_english(text: str) -> bool:
    if not text:
        return False
    asc = sum(1 for c in text if ord(c) < 128)
    return asc / len(text) >= 0.80


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _cell_str(row: dict[str, Any], key: str | None) -> str:
    if not key:
        return ""
    v = row.get(key)
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


def _cell_list_first_str(row: dict[str, Any], key: str | None) -> str:
    if not key:
        return ""
    v = row.get(key)
    if v is None:
        return ""
    if isinstance(v, list):
        if not v:
            return ""
        x = v[0]
        return str(x).strip() if x is not None else ""
    return str(v).strip()


def _topic_from_row(row: dict[str, Any], topic_field: str | None, default_topic: str) -> str:
    if not topic_field or topic_field not in row:
        return default_topic
    v = row.get(topic_field)
    if isinstance(v, list):
        if not v:
            return "uncategorized"
        first = str(v[0]).strip()
        if ">" in first:
            return first.split(">", 1)[0].strip() or "uncategorized"
        return first or "uncategorized"
    s = (v or "").strip() if isinstance(v, str) else str(v).strip()
    if not s:
        return "uncategorized"
    if ">" in s:
        return s.split(">", 1)[0].strip() or "uncategorized"
    return s


def _solution_from_row(row: dict[str, Any], solution_field: str) -> str:
    if not solution_field or solution_field not in row:
        return ""
    v = row.get(solution_field)
    if isinstance(v, list):
        parts = [str(x).strip() for x in v if x is not None and str(x).strip()]
        return "\n\n".join(parts) if parts else ""
    return str(v or "").strip()


def _norm_lower(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _passes_language_row(cfg: Config, row: dict[str, Any]) -> bool:
    lf, lv = cfg.input.language_field, cfg.input.language_filter
    if lf and lf not in row:
        return True
    # MathNet leaves `language` blank on most rows; only enforce the column when it is non-empty.
    if lf and lv is not None:
        got = _cell_str(row, lf)
        if got and _norm_lower(got) != _norm_lower(str(lv)):
            return False
    if cfg.filters.language == "en":
        qf = cfg.input.question_field
        q = _cell_str(row, qf) or _cell_list_first_str(row, qf)
        if not _is_english(q):
            return False
    return True


def _canonical_problem_type(label: str) -> str:
    """Normalize MathNet labels so YAML can say 'answer only' for 'final answer only' rows."""
    m = _norm_lower(label)
    if m in ("final answer only", "answer only"):
        return "answer_only"
    if m == "proof and answer":
        return "proof_and_answer"
    if m == "proof only":
        return "proof_only"
    return m


def _passes_problem_type_row(cfg: Config, row: dict[str, Any]) -> bool:
    pf, allowed = cfg.input.problem_type_field, cfg.input.problem_type_filter
    if not pf or not allowed:
        return True
    if pf not in row:
        return True
    got = _canonical_problem_type(_cell_str(row, pf))
    allowed_canon = {_canonical_problem_type(x) for x in allowed}
    return got in allowed_canon


def _passes_country_row(cfg: Config, row: dict[str, Any]) -> bool:
    cf, cv = cfg.input.country_field, cfg.input.country_filter
    if not cf or cv is None:
        return True
    if cf not in row:
        return True
    got = _cell_str(row, cf)
    return _norm_lower(got) == _norm_lower(str(cv))


def _clean_question_markdown(raw: str) -> tuple[str, bool]:
    """Strip image markdown; return (text, had_image_markdown)."""
    had = bool(_IMG_MD.search(raw))
    q = _IMG_MD.sub("", raw).strip()
    q = re.sub(r"\n{3,}", "\n\n", q)
    return q, had


def ingest_dataset(cfg: Config, store: Store) -> None:
    """Load HF dataset → apply filters → write seeds to DB."""
    trace_event(store, "ingest", "ingest.start", message=f"HF {cfg.input.dataset} split={cfg.input.split}")
    console.print(f"[bold]Loading dataset[/]: {cfg.input.dataset} (split={cfg.input.split})")
    if cfg.input.config_name:
        ds = load_dataset(cfg.input.dataset, cfg.input.config_name, split=cfg.input.split)
    else:
        ds = load_dataset(cfg.input.dataset, split=cfg.input.split)

    qf, af, tf = cfg.input.question_field, cfg.input.answer_field, cfg.input.topic_field
    cols = set(ds.column_names)
    need = {qf, cfg.input.solution_field}
    missing = need - cols
    if missing:
        raise ValueError(
            "dataset missing required input columns "
            f"{sorted(missing)}; required fields are question_field and solution_field; have {sorted(cols)}"
        )

    kept, dropped = 0, 0
    cap = cfg.input.max_seeds

    for row in ds:
        if cap is not None and kept >= cap:
            break
        raw_q = _cell_str(row, qf) or _cell_list_first_str(row, qf)
        a = _cell_str(row, af) or _cell_list_first_str(row, af)
        topic_raw = _topic_from_row(row, tf, cfg.input.default_topic)
        sol = _solution_from_row(row, cfg.input.solution_field)

        if not raw_q or not sol:
            dropped += 1
            continue
        if not _passes_language_row(cfg, row):
            dropped += 1
            continue
        if not _passes_problem_type_row(cfg, row):
            dropped += 1
            continue
        if not _passes_country_row(cfg, row):
            dropped += 1
            continue

        q, had_img = _clean_question_markdown(raw_q)
        if not q.strip():
            dropped += 1
            continue
        if had_img and _FIG_LANG.search(q):
            dropped += 1
            continue

        if _approx_tokens(q) > cfg.filters.max_question_tokens:
            dropped += 1
            continue
        if a and _approx_tokens(a) > cfg.filters.max_answer_tokens:
            dropped += 1
            continue

        sid = _seed_id(q, a)
        store.upsert_seed(sid, q, a, topic_raw, solution_text=sol)
        trace_event(
            store,
            "ingest",
            "ingest.seed",
            seed_id=sid,
            message=f"row kept total={kept + 1}",
            payload={"question": q, "answer": a, "topic_raw": topic_raw},
        )
        kept += 1
    console.print(f"  ingested: kept {kept}, dropped {dropped}")
    trace_event(store, "ingest", "ingest.done", payload={"kept": kept, "dropped": dropped})


def normalize_topics(cfg: Config, store: Store, embedder: EmbeddingClient, llm: LLMClient) -> None:
    """Distinct topic_raw → topic_norm (cluster if many uniques)."""
    trace_event(store, "topics", "topics.normalize.start", message="distinct topic_raw → topic_norm")
    with store._conn() as c:
        rows = c.execute("SELECT DISTINCT topic_raw FROM seeds WHERE eligible = 1").fetchall()
    raws = [r["topic_raw"] for r in rows]
    n_unique = len(raws)
    console.print(f"  unique raw topics: {n_unique}")

    if n_unique <= 30:
        mapping = {r: _light_clean(r) for r in raws}
        trace_event(store, "topics", "topics.path", message=f"≤30 unique topics → light clean ({n_unique})")
    else:
        console.print("  clustering topic strings (>30 unique → normalizing)")
        trace_event(store, "topics", "topics.cluster", message=f"KMeans k≤15 on topic embeddings ({n_unique} raw)")
        vecs = np.array([embedder.embed(r) for r in raws], dtype=np.float32)
        k = min(15, max(2, int(np.sqrt(n_unique))))
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(vecs)
        labels = km.labels_
        name_for: dict[int, str] = {}
        for cid in range(k):
            members = [raws[i] for i, l in enumerate(labels) if l == cid][:25]
            trace_event(
                store,
                "topics",
                "topics.cluster.name",
                attempt=cid + 1,
                message="LLM name for cluster",
                payload={"member_count": len(members)},
            )
            name_for[cid] = _name_cluster(llm, cfg.models.judge, members)
        mapping = {raws[i]: name_for[labels[i]] for i in range(n_unique)}

    with store._conn() as c:
        for raw, norm in mapping.items():
            c.execute("UPDATE seeds SET topic_norm = ? WHERE topic_raw = ?", (norm, raw))

    counts = store.topic_counts()
    small = {t for t, n in counts.items() if n < cfg.filters.min_topic_size}
    if small:
        console.print(f"  small topics (<{cfg.filters.min_topic_size}): {sorted(small)}")
        with store._conn() as c:
            for t in small:
                c.execute("UPDATE seeds SET eligible = 0 WHERE topic_norm = ?", (t,))
        counts = store.topic_counts()
    console.print(f"  topic distribution: {dict(sorted(counts.items()))}")
    trace_event(store, "topics", "topics.normalize.done", payload={"topics": dict(sorted(counts.items()))})


def _light_clean(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s or "uncategorized"


def _name_cluster(llm: LLMClient, model: str, members: list[str]) -> str:
    user = (
        "You will be shown a cluster of topic-string examples. Return ONE short snake_case label "
        '(<= 3 words) that best describes the cluster. Output JSON: {"label": "..."}.\n\n'
        f"Members: {members}"
    )
    out = llm.chat_json(
        model=model,
        system="You are a taxonomy normalizer. Output strict JSON.",
        user=user,
        temperature=0.0,
    )
    label = str(out.get("label", "uncategorized")) if isinstance(out, dict) else "uncategorized"
    return _light_clean(label)
