"""Global distilled memory for mutation attempts."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .config import Config
from .db import Store


def _compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+\n", "\n", (text or "").strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _norm_key(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").lower()).strip()
    text = re.sub(r"[^a-z0-9 ,.;:_+-]", "", text)
    return text[:900]


def _score(scores: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(scores.get(key, default))
    except Exception:
        return default
    return max(0.0, min(1.0, value))


def _experience_weight(*, passed: bool, failure_kind: str | None, scores: dict[str, Any]) -> float:
    if passed:
        useful = (
            _score(scores, "mutation_quality", 0.5)
            + _score(scores, "sharpness", 0.5)
            + _score(scores, "novelty", 0.5)
            + _score(scores, "non_stitched", 0.5)
        ) / 4.0
        return round(1.0 + useful, 3)

    severity = {
        "near_paraphrase": 1.3,
        "weak_mutation": 1.2,
        "number_swap": 1.2,
        "stitched": 1.1,
        "missing_hinge": 1.0,
        "incorrect": 0.8,
        "routine_length": 0.8,
        "ambiguous": 0.8,
    }.get(failure_kind or "", 0.7)
    return round(severity, 3)


def _memory_lesson(
    cfg: Config,
    *,
    seed: Any,
    episode: Any,
    passed: bool,
    failure_kind: str | None,
    scores: dict[str, Any],
    story: str,
) -> str:
    topic = seed["topic_norm"] or seed["topic_raw"] or "unknown topic"
    kind = "SUCCESS" if passed else "FAILURE"
    score_keys = [
        "hinge_preservation",
        "mutation_quality",
        "sharpness",
        "non_stitched",
        "solution_economy",
        "novelty",
        "minhash_overlap",
    ]
    score_line = ", ".join(
        f"{key}={scores[key]:.2f}" if isinstance(scores.get(key), float) else f"{key}={scores[key]}"
        for key in score_keys
        if key in scores
    )
    lesson = f"""{kind} MEMORY
Topic: {topic}
Mutation used: {episode['mutation_used'] or 'unknown'}
Failure kind: {failure_kind or 'none'}
Scores: {score_line or 'not recorded'}

Reusable lesson:
{story.strip()}

How to use this memory:
{"Reuse this transformation pattern when the hinge is preserved but hidden in a cleaner form." if passed else "Avoid this failure pattern unless the new problem changes the solver's real bottleneck."}
"""
    return _compact(lesson, cfg.mutation.global_memory_max_lesson_chars)


def record_mutation_experience(
    cfg: Config,
    store: Store,
    *,
    seed: Any,
    episode: Any,
    passed: bool,
    failure_kind: str | None,
    scores: dict[str, Any],
    story: str,
) -> str | None:
    """Promote a judged episode story into reusable global mutation memory."""
    if not cfg.mutation.global_memory_enabled:
        return None
    if not (story or "").strip():
        return None

    kind = "success" if passed else "failure"
    lesson = _memory_lesson(
        cfg,
        seed=seed,
        episode=episode,
        passed=passed,
        failure_kind=failure_kind,
        scores=scores,
        story=story,
    )
    key = "|".join(
        [
            kind,
            failure_kind or "",
            episode["mutation_used"] or "",
            _norm_key(story),
        ]
    )
    experience_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    weight = _experience_weight(passed=passed, failure_kind=failure_kind, scores=scores)
    store.upsert_mutation_experience(
        experience_id=experience_id,
        kind=kind,
        topic_norm=seed["topic_norm"] or seed["topic_raw"] or "",
        failure_kind=failure_kind,
        mutation_used=episode["mutation_used"] or "",
        lesson=lesson,
        source_episode_id=episode["id"],
        weight_delta=weight,
    )
    store.prune_mutation_experiences(max_active=cfg.mutation.global_memory_max_active)
    return experience_id


def backfill_mutation_experiences(
    cfg: Config,
    store: Store,
    *,
    limit: int | None = None,
) -> int:
    """Create global memory rows from already judged mutation episodes."""
    scan_limit = limit or max(cfg.mutation.global_memory_max_active * 5, 1000)
    rows = store.list_mutation_stories(limit=scan_limit)
    created = 0
    for episode in rows:
        if episode["result"] not in {"accepted", "rejected"}:
            continue
        seed = store.get_seed(episode["seed_id"])
        if seed is None:
            continue
        try:
            scores = json.loads(episode["scores_json"] or "{}")
        except Exception:
            scores = {}
        experience_id = record_mutation_experience(
            cfg,
            store,
            seed=seed,
            episode=episode,
            passed=episode["result"] == "accepted",
            failure_kind=episode["failure_kind"],
            scores=scores,
            story=episode["story"],
        )
        if experience_id:
            created += 1
    return created


def experience_row_to_dict(row: Any) -> dict[str, Any]:
    try:
        source_ids = json.loads(row["source_episode_ids"] or "[]")
    except Exception:
        source_ids = []
    return {
        "id": row["id"],
        "scope": row["scope"],
        "kind": row["kind"],
        "topic_norm": row["topic_norm"],
        "failure_kind": row["failure_kind"],
        "mutation_used": row["mutation_used"],
        "lesson": row["lesson"],
        "source_episode_ids": source_ids,
        "source_count": row["source_count"],
        "weight": row["weight"],
        "active": bool(row["active"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
