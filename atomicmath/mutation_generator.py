"""Generate single-question transformations from seed hinges and examples."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from .config import Config
from .db import Store
from .llm import LLMClient
from .mutation_hinges import ensure_hinges_for_seed
from .mutation_prompt import balanced_json_from_text, build_plan_generate_prompt


MUTATION_SYSTEM = """You are an olympiad-style math problem transformer.

You are given one solved seed problem, hinge notes, transformation examples,
and previous success/failure stories.

In a single response, first plan a nontrivial transformation and reject weak
ideas, then generate one final problem from the chosen transformation.

The new problem must be one coherent task. It must not be a number swap, surface
paraphrase, wrapper-only context, or stitched-on extension. Return JSON only."""


@dataclass
class MutationCandidate:
    episode_id: str
    seed_id: str
    new_question: str
    answer: str
    short_solution: str
    mutation_used: str
    what_got_mutated: str
    reason_for_mutation: str
    primary_hinge_preserved: str
    conceptual_delta: str
    why_problem_is_sharper: str
    why_not_stitched: str
    why_not_a_direct_sibling: str
    risk_notes: str

    def as_dict(self) -> dict[str, str]:
        return {
            "new_question": self.new_question,
            "answer": self.answer,
            "short_solution": self.short_solution,
            "mutation_used": self.mutation_used,
            "what_got_mutated": self.what_got_mutated,
            "reason_for_mutation": self.reason_for_mutation,
            "primary_hinge_preserved": self.primary_hinge_preserved,
            "conceptual_delta": self.conceptual_delta,
            "why_problem_is_sharper": self.why_problem_is_sharper,
            "why_not_stitched": self.why_not_stitched,
            "why_not_a_direct_sibling": self.why_not_a_direct_sibling,
            "risk_notes": self.risk_notes,
        }


def _coerce_plan(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    keys = [
        "core_bottleneck",
        "discarded_transformations",
        "candidate_transformations",
        "chosen_transformation",
        "conceptual_delta",
        "hinge_preserved",
        "why_this_should_be_nontrivial",
    ]
    return {k: raw.get(k) for k in keys if k in raw}


def _coerce_candidate(raw: Any, seed_id: str, episode_id: str) -> MutationCandidate:
    if not isinstance(raw, dict):
        raise ValueError("mutation generator returned non-object JSON")

    def get(key: str) -> str:
        return str(raw.get(key, "") or "").strip()

    question = get("new_question")
    answer = get("answer")
    solution = get("short_solution")
    mutation = get("mutation_used")
    if not question or not answer or not solution or not mutation:
        raise ValueError(f"mutation candidate missing required fields: {json.dumps(raw, ensure_ascii=False)[:500]}")
    return MutationCandidate(
        episode_id=episode_id,
        seed_id=seed_id,
        new_question=question,
        answer=answer,
        short_solution=solution,
        mutation_used=mutation,
        what_got_mutated=get("what_got_mutated"),
        reason_for_mutation=get("reason_for_mutation"),
        primary_hinge_preserved=get("primary_hinge_preserved"),
        conceptual_delta=get("conceptual_delta"),
        why_problem_is_sharper=get("why_problem_is_sharper"),
        why_not_stitched=get("why_not_stitched"),
        why_not_a_direct_sibling=get("why_not_a_direct_sibling"),
        risk_notes=get("risk_notes"),
    )


def generate_mutations_for_seed(
    cfg: Config,
    store: Store,
    llm: LLMClient,
    seed_id: str,
    *,
    n: int = 1,
) -> list[MutationCandidate]:
    seed = store.get_seed(seed_id)
    if seed is None:
        raise ValueError(f"unknown seed_id: {seed_id}")
    hinges = ensure_hinges_for_seed(cfg, store, llm, seed_id)
    model = cfg.mutation.generation_model or cfg.models.generators[0]
    hinge_ids = [h["id"] for h in hinges]

    candidates: list[MutationCandidate] = []
    for idx in range(max(1, int(n))):
        prompt = build_plan_generate_prompt(cfg, store, seed, hinges)
        out = llm.chat(
            model=model,
            system=MUTATION_SYSTEM,
            user=prompt,
            temperature=float(cfg.mutation.generator_temperature),
            max_tokens=2600,
            use_cache=False,
        )
        content = (out.get("content") or "").strip()
        raw = balanced_json_from_text(content)
        if raw is None:
            raise ValueError(
                f"could not parse mutation JSON from {model}: {content[:300]!r}; "
                f"finish_reason={out.get('finish_reason')!r}; usage={out.get('usage')!r}"
            )
        plan_dict = _coerce_plan(raw)
        episode_id = hashlib.sha256(
            f"{seed_id}\x1e{idx}\x1e{time.time_ns()}\x1e{json.dumps(raw, sort_keys=True, default=str)}".encode()
        ).hexdigest()[:16]
        cand = _coerce_candidate(raw, seed_id, episode_id)
        store.insert_mutation_episode(
            episode_id=episode_id,
            seed_id=seed_id,
            hinge_ids=hinge_ids,
            prompt_text=prompt,
            mutation_used=cand.mutation_used,
            new_question=cand.new_question,
            answer=cand.answer,
            short_solution=cand.short_solution,
            result="pending",
            scores={},
            plan=plan_dict,
            candidate=cand.as_dict(),
            story="",
        )
        candidates.append(cand)
    return candidates
