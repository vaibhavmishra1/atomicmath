"""Moderate quality gate: reject clearly routine exercise-like candidates."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import Config
from .curriculum import Brief
from .llm import LLMClient
from .realizer import Candidate
from .trace import trace_event
from .db import Store


@dataclass
class QualityVerdict:
    passed: bool
    depth_score: float
    contest_score: float
    routine_score: float
    reason: str


def _clamp01(x: Any, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return max(0.0, min(1.0, v))


def _parse_verdict(raw: Any) -> QualityVerdict:
    if not isinstance(raw, dict):
        return QualityVerdict(False, 0.0, 0.0, 1.0, "quality judge returned non-object JSON")
    return QualityVerdict(
        passed=bool(raw.get("pass", False)),
        depth_score=_clamp01(raw.get("depth_score"), 0.0),
        contest_score=_clamp01(raw.get("contest_score"), 0.0),
        routine_score=_clamp01(raw.get("routine_score"), 1.0),
        reason=str(raw.get("reason", "") or "")[:1000],
    )


def judge_quality(
    llm: LLMClient,
    cfg: Config,
    brief: Brief,
    scaffold: dict[str, Any],
    candidate: Candidate,
    *,
    event_store: Store | None = None,
    synth_attempt: int | None = None,
) -> QualityVerdict:
    """Reject only clearly low-depth / direct-formula outputs.

    This is intentionally not an IMO gate. It asks whether the problem is a
    worthwhile contest-style problem for the seed profile and scaffold, while
    allowing accessible problems that have at least one real idea.
    """
    if not cfg.quality.enabled:
        return QualityVerdict(True, 1.0, 1.0, 0.0, "quality gate disabled")

    trace_event(
        event_store,
        "synthesis",
        "quality.start",
        attempt=synth_attempt,
        message="depth / routine-ness judge",
        payload={
            "model": cfg.quality.model,
            "candidate_question": candidate.question,
            "candidate_answer": candidate.answer,
            "brief": brief.__dict__,
            "scaffold": scaffold,
        },
    )
    user = (
        "Evaluate whether this generated math problem is high-quality for its seed-corpus profile.\n"
        "Do NOT judge whether it is IMO-specific. The goal is general contest-quality relevance.\n\n"
        "Reject only if it is clearly an exercise rather than a contest problem, for example:\n"
        "- solution is just one or two standard mechanical steps;\n"
        "- direct formula application with no structural insight;\n"
        "- no hidden constraint, transformation, invariant, extremal idea, smoothing, CRT split, "
        "case split, or comparable reasoning lever;\n"
        "- problem merely asks for a textbook fact or plug-in computation.\n\n"
        "Be moderate: accessible problems may pass if they require at least one meaningful idea and "
        "faithfully realize the scaffold. Do not demand extreme hardness.\n\n"
        f"BRIEF:\n{json.dumps(brief.__dict__, ensure_ascii=False)}\n\n"
        f"SCAFFOLD:\n{json.dumps(scaffold, ensure_ascii=False)[:12000]}\n\n"
        f"CANDIDATE_PROBLEM:\n{candidate.question}\n\n"
        f"CLAIMED_ANSWER:\n{candidate.answer}\n\n"
        "Return strict JSON with exactly these keys:\n"
        "{\n"
        '  "pass": true/false,\n'
        '  "depth_score": number from 0 to 1,\n'
        '  "contest_score": number from 0 to 1,\n'
        '  "routine_score": number from 0 to 1,\n'
        '  "reason": "short explanation"\n'
        "}\n"
        "Interpretation: higher depth/contest is better; higher routine_score means more exercise-like."
    )
    try:
        raw = llm.chat_json(
            model=cfg.quality.model,
            system="You are a calibrated contest-math quality judge. Return JSON only.",
            user=user,
            temperature=0.0,
        )
    except Exception as e:
        return QualityVerdict(False, 0.0, 0.0, 1.0, f"quality judge failed: {e}")

    verdict = _parse_verdict(raw)
    threshold_pass = (
        verdict.depth_score >= cfg.quality.min_depth_score
        and verdict.contest_score >= cfg.quality.min_contest_score
        and verdict.routine_score <= cfg.quality.max_routine_score
    )
    passed = threshold_pass and (verdict.passed or not cfg.quality.require_judge_pass)
    final = QualityVerdict(
        passed=passed,
        depth_score=verdict.depth_score,
        contest_score=verdict.contest_score,
        routine_score=verdict.routine_score,
        reason=verdict.reason,
    )
    trace_event(
        event_store,
        "synthesis",
        "quality.done",
        attempt=synth_attempt,
        payload=final.__dict__,
    )
    return final
