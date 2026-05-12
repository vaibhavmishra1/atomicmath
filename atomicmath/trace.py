"""Structured pipeline tracing for the live dashboard (SQLite pipeline_events)."""
from __future__ import annotations

from typing import Any

from .db import Store

# Payload keys that carry full problem text / traces / large JSON blobs (larger cap per value).
_LONG_TEXT_KEYS: frozenset[str] = frozenset({
    "question",
    "answer",
    "trace",
    "solver_answer",
    "gold_answer",
    "claimed_answer",
    "parent_fingerprints",
    "candidate_question",
    "candidate_answer",
    "topic_raw",
    "verifier_answer",
    "parent_question",
    "parent_answer",
    "question_preview",
    "embedding_stem",
    "verifier_answers",
    "target_answer",
    "intended_solution_sketch",
    "forced_constraints",
    "scaffold",
})
_MAX_LONG = 14_000
_MAX_SHORT = 4_000
_MAX_LIST = 120


def _shrink_value(v: Any, key: str | None = None) -> Any:
    if isinstance(v, str):
        lim = _MAX_LONG if key in _LONG_TEXT_KEYS else _MAX_SHORT
        if len(v) > lim:
            return v[:lim] + "…"
        return v
    if isinstance(v, dict):
        return {str(k): _shrink_value(x, str(k)) for k, x in v.items()}
    if isinstance(v, list):
        tail = ["…"] if len(v) > _MAX_LIST else []
        # Use None so nested dicts/strings pick limits from their own keys (e.g. solver_answer).
        return [_shrink_value(x, None) for x in v[:_MAX_LIST]] + tail
    return v


def trace_event(
    store: Store | None,
    phase: str,
    step: str,
    *,
    seed_id: str | None = None,
    attempt: int | None = None,
    message: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Append one row to pipeline_events (no-op if store is None)."""
    if store is None:
        return
    safe_payload = _shrink_value(payload) if payload else None
    store.log_pipeline_event(
        phase, step, seed_id=seed_id, attempt=attempt, message=message, payload=safe_payload
    )
