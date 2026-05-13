"""Judge generated mutations and write global success/failure memory."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .config import Config
from .db import Store
from .llm import LLMClient
from .minhash_util import minhash_jaccard, minhash_signature
from .mutation_memory import record_mutation_experience
from .mutation_prompt import balanced_json_from_text
from .verifier import verify_correctness


QUALITY_SYSTEM = """You judge mutation-generated contest math problems.

The candidate was generated from exactly one seed problem. Judge whether it is a
meaningful mutation: it should preserve an important hinge, change the structure
in a useful way, become sharper, and remain one coherent non-stitched problem.

Be strict about stitched problems, number swaps, bloated routine calculation,
missing original hinge, wrapper-only contexts, direct siblings of the seed, and
ambiguous statements. Return JSON only."""


@dataclass
class MutationVerdict:
    episode_id: str
    passed: bool
    failure_kind: str | None
    scores: dict[str, Any]
    story: str


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, v))


def _quality_prompt(seed: Any, hinges: list[Any], episode: Any, minhash_overlap: float) -> str:
    hinge_block = "\n\n".join(
        f"HINGE {i} ({h['label']}):\n{h['hinge_text']}" for i, h in enumerate(hinges, start=1)
    )
    try:
        plan = json.loads(episode["plan_json"] or "{}")
    except Exception:
        plan = {}
    try:
        candidate_meta = json.loads(episode["candidate_json"] or "{}")
    except Exception:
        candidate_meta = {}
    return f"""ORIGINAL QUESTION:
{seed['question']}

ORIGINAL ANSWER:
{seed['answer']}

ORIGINAL SOLUTION:
{seed['solution_text']}

HINGES:
{hinge_block}

CANDIDATE NEW QUESTION:
{episode['new_question']}

CANDIDATE ANSWER:
{episode['answer']}

CANDIDATE SHORT SOLUTION:
{episode['short_solution']}

MUTATION USED:
{episode['mutation_used']}

TRANSFORMATION PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

CANDIDATE SELF-EXPLANATION:
{json.dumps(candidate_meta, ensure_ascii=False, indent=2)}

MINHASH_OVERLAP_WITH_ORIGINAL:
{minhash_overlap:.3f}

Judge the mutation.

Reject if:
- it is only a number/sign/exponent/name swap;
- it is only a wrapper context around the same task;
- it is a direct sibling of the seed rather than a real transformation;
- it adds a downstream task after the original hinge is solved;
- the conceptual_delta is vague, cosmetic, or not visible in the solution.

Return strict JSON:
{{
  "pass": true/false,
  "hinge_preservation": number from 0 to 1,
  "mutation_quality": number from 0 to 1,
  "sharpness": number from 0 to 1,
  "non_stitched": number from 0 to 1,
  "solution_economy": number from 0 to 1,
  "novelty": number from 0 to 1,
  "failure_kind": null or one of ["incorrect", "missing_hinge", "number_swap", "stitched", "routine_length", "ambiguous", "near_paraphrase", "weak_mutation"],
  "reason": "short explanation",
  "story": "success or failure mutation story: what got mutated, why it worked or failed, and what to reuse or avoid"
}}"""


def _parse_quality(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "pass": False,
            "hinge_preservation": 0.0,
            "mutation_quality": 0.0,
            "sharpness": 0.0,
            "non_stitched": 0.0,
            "solution_economy": 0.0,
            "novelty": 0.0,
            "failure_kind": "judge_error",
            "reason": "quality judge returned non-object JSON",
            "story": "",
        }
    out = {
        "pass": bool(raw.get("pass", False)),
        "hinge_preservation": _clamp01(raw.get("hinge_preservation")),
        "mutation_quality": _clamp01(raw.get("mutation_quality")),
        "sharpness": _clamp01(raw.get("sharpness")),
        "non_stitched": _clamp01(raw.get("non_stitched")),
        "solution_economy": _clamp01(raw.get("solution_economy")),
        "novelty": _clamp01(raw.get("novelty")),
        "failure_kind": raw.get("failure_kind"),
        "reason": str(raw.get("reason", "") or "")[:1200],
        "story": str(raw.get("story", "") or "")[:3000],
    }
    if out["failure_kind"] in ("", "null", "None"):
        out["failure_kind"] = None
    return out


def _threshold_failure(cfg: Config, q: dict[str, Any]) -> str | None:
    if q.get("minhash_overlap", 0.0) > cfg.mutation.max_seed_minhash_overlap:
        return "near_paraphrase"
    if q["hinge_preservation"] < cfg.mutation.min_hinge_preservation:
        return "missing_hinge"
    if q["mutation_quality"] < cfg.mutation.min_mutation_quality:
        return "weak_mutation"
    if q["sharpness"] < cfg.mutation.min_sharpness:
        return "weak_mutation"
    if q["non_stitched"] < cfg.mutation.min_non_stitched:
        return "stitched"
    if q["solution_economy"] < cfg.mutation.min_solution_economy:
        return "routine_length"
    if q["novelty"] < cfg.mutation.min_novelty:
        return "near_paraphrase"
    return None


def judge_mutation_episode(
    cfg: Config,
    store: Store,
    llm: LLMClient,
    episode_id: str,
) -> MutationVerdict:
    episode = store.get_mutation_episode(episode_id)
    if episode is None:
        raise ValueError(f"unknown mutation episode: {episode_id}")
    seed = store.get_seed(episode["seed_id"])
    if seed is None:
        raise ValueError(f"episode {episode_id} points to missing seed {episode['seed_id']}")
    hinges = store.list_seed_hinges(seed["id"])
    if not hinges:
        raise ValueError(f"seed {seed['id']} has no stored hinges")

    orig_mh = minhash_signature(seed["question"], seed["answer"])
    cand_mh = minhash_signature(episode["new_question"] or "", episode["answer"] or "")
    mh_overlap = minhash_jaccard(orig_mh, cand_mh)

    correctness_scores: dict[str, Any] = {
        "correctness_passed": True,
        "correctness_agreements": None,
        "correctness_answers": [],
    }
    if cfg.mutation.strict_correctness:
        correctness = verify_correctness(
            llm,
            cfg,
            episode["new_question"] or "",
            episode["answer"] or "",
            generator_model=cfg.mutation.generation_model or cfg.models.generators[0],
        )
        correctness_scores = {
            "correctness_passed": correctness.passed,
            "correctness_agreements": correctness.agreements,
            "correctness_answers": correctness.answers,
        }
        if not correctness.passed:
            scores = {
                **correctness_scores,
                "minhash_overlap": mh_overlap,
                "reason": "correctness verifier did not agree with claimed answer",
            }
            story = (
                "SOURCE HINGE:\n"
                "The generated candidate was rejected before mutation scoring because its claimed answer did not pass verification.\n\n"
                "WHY IT FAILED:\n"
                "Correctness failed, so no mutation lesson should be trusted except to avoid this candidate shape."
            )
            store.update_mutation_episode(
                episode_id,
                result="rejected",
                failure_kind="incorrect",
                scores=scores,
                story=story,
            )
            record_mutation_experience(
                cfg,
                store,
                seed=seed,
                episode=episode,
                passed=False,
                failure_kind="incorrect",
                scores=scores,
                story=story,
            )
            return MutationVerdict(episode_id, False, "incorrect", scores, story)

    model = cfg.mutation.judge_model or cfg.models.judge
    prompt = _quality_prompt(seed, hinges, episode, mh_overlap)
    out = llm.chat(
        model=model,
        system=QUALITY_SYSTEM,
        user=prompt,
        temperature=float(cfg.mutation.judge_temperature),
        max_tokens=1800,
    )
    raw = balanced_json_from_text(out.get("content") or "")
    q = _parse_quality(raw)
    q["minhash_overlap"] = mh_overlap
    threshold_failure = _threshold_failure(cfg, q)
    failure_kind = threshold_failure or q.get("failure_kind")
    passed = bool(q["pass"]) and failure_kind is None
    result = "accepted" if passed else "rejected"
    scores = {
        **correctness_scores,
        "minhash_overlap": mh_overlap,
        **{k: v for k, v in q.items() if k != "story"},
    }
    story = q.get("story") or q.get("reason") or ""
    if not story:
        if passed:
            story = (
                "MUTATION USED:\n"
                f"{episode['mutation_used']}\n\n"
                "WHY IT WORKED:\n"
                "Accepted by mutation judge."
            )
        else:
            story = f"WHY IT FAILED:\n{failure_kind or 'unknown failure'}"
    store.update_mutation_episode(
        episode_id,
        result=result,
        failure_kind=None if passed else str(failure_kind or "judge_reject"),
        scores=scores,
        story=story,
    )
    record_mutation_experience(
        cfg,
        store,
        seed=seed,
        episode=episode,
        passed=passed,
        failure_kind=None if passed else str(failure_kind or "judge_reject"),
        scores=scores,
        story=story,
    )
    return MutationVerdict(
        episode_id=episode_id,
        passed=passed,
        failure_kind=None if passed else str(failure_kind or "judge_reject"),
        scores=scores,
        story=story,
    )


def judge_pending_mutation_episodes(
    cfg: Config,
    store: Store,
    llm: LLMClient,
    *,
    limit: int | None = None,
) -> list[MutationVerdict]:
    verdicts = []
    for row in store.list_pending_mutation_episodes(limit=limit):
        verdicts.append(judge_mutation_episode(cfg, store, llm, row["id"]))
    return verdicts
