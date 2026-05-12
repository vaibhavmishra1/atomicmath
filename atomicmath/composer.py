"""Retrosynthesis composer: Brief + exemplars → Scaffold JSON (single LLM call)."""
from __future__ import annotations

import json
from typing import Any

from .config import Config
from .curriculum import Brief
from .llm import LLMClient
from .taxonomy import Taxonomy

_SYSTEM = """You are an olympiad problem author doing retrosynthesis.

Given a BRIEF (concepts + answer form) and optional EXEMPLAR scaffolds from real contests, design a new problem SCAFFOLD: start from a definite target answer and the conceptual spine, then list forced constraints that make that answer unique.

Rules:
- No difficulty labels or bands. Maximize conceptual depth and constraint tightness.
- primary_concept and secondary_concept MUST be copied exactly from the allowed lists in the user message.
- answer_form MUST be one of the allowed question forms.
- target_answer must be short (suitable for automated verification).
- forced_constraints: 3–8 short strings; each should be load-bearing for the solution.
- intended_solution_sketch: 4–10 sentences, proof-level, no fluff.

Return ONLY valid JSON with keys:
target_answer, intended_solution_sketch, forced_constraints (array of strings),
primary_concept, secondary_concept, answer_form
"""


def _coerce_scaffold(raw: dict[str, Any], brief: Brief, tax: Taxonomy) -> dict[str, Any] | None:
    try:
        ta = str(raw.get("target_answer", "")).strip()
        sk = str(raw.get("intended_solution_sketch", "")).strip()
        fc = raw.get("forced_constraints")
        pc = str(raw.get("primary_concept", "")).strip()
        sc = str(raw.get("secondary_concept", "")).strip()
        af = str(raw.get("answer_form", "")).strip()
    except Exception:
        return None
    if not ta or not sk or not isinstance(fc, list) or not fc:
        return None
    fc2 = [str(x).strip() for x in fc if str(x).strip()]
    if len(fc2) < 2:
        return None
    if pc not in tax.concepts:
        pc = brief.primary_concept if brief.primary_concept in tax.concepts else tax.concepts[0]
    if sc not in tax.concepts:
        sc = brief.secondary_concept if brief.secondary_concept in tax.concepts else tax.concepts[min(1, len(tax.concepts) - 1)]
    if af not in tax.question_forms:
        af = brief.answer_form if brief.answer_form in tax.question_forms else tax.question_forms[0]
    return {
        "target_answer": ta,
        "intended_solution_sketch": sk,
        "forced_constraints": fc2,
        "primary_concept": pc,
        "secondary_concept": sc,
        "answer_form": af,
    }


def design(
    llm: LLMClient,
    cfg: Config,
    tax: Taxonomy,
    brief: Brief,
    exemplar_scaffolds: list[dict[str, Any]],
) -> dict[str, Any]:
    model = cfg.composer.model or cfg.models.generators[0]
    concepts_sample = ", ".join(tax.concepts[:120])
    if len(tax.concepts) > 120:
        concepts_sample += ", …"
    forms = ", ".join(tax.question_forms)
    ex_block = ""
    if exemplar_scaffolds:
        parts = []
        for i, ex in enumerate(exemplar_scaffolds[:4]):
            parts.append(f"EXEMPLAR_{i + 1}:\n{json.dumps(ex, ensure_ascii=False)[:6000]}")
        ex_block = "\n\n".join(parts)
    user = (
        f"ALLOWED_CONCEPTS (exact tokens):\n{concepts_sample}\n\n"
        f"ALLOWED_ANSWER_FORMS:\n{forms}\n\n"
        f"BRIEF:\n{json.dumps(brief.__dict__, ensure_ascii=False)}\n\n"
        f"{ex_block}\n\n"
        "Design one new scaffold aligned with the brief. JSON only."
    )
    raw = llm.chat_json(
        model=model,
        system=_SYSTEM,
        user=user,
        temperature=float(cfg.composer.temperature),
    )
    if not isinstance(raw, dict):
        raise ValueError("composer returned non-object JSON")
    out = _coerce_scaffold(raw, brief, tax)
    if out is None:
        raise ValueError(f"composer scaffold invalid or incomplete: {str(raw)[:400]!r}")
    return out
