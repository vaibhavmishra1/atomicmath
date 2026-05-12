"""Realizer: scaffold → contest problem JSON (best-of-K, multipart heuristic)."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .canonical import canonical_equal
from .config import Config
from .db import Store
from .llm import LLMClient
from .minhash_util import minhash_jaccard, minhash_signature
from .trace import trace_event


@dataclass
class Candidate:
    question: str
    answer: str
    topic: str
    correctness: float
    anti_distance: float

    def combined(self, cfg: Config) -> float:
        gc = cfg.generator
        w_sum = max(1e-9, (gc.correctness_weight + gc.novelty_weight))
        cw = gc.correctness_weight / w_sum
        nw = gc.novelty_weight / w_sum
        return (cw * self.correctness + nw * self.anti_distance)


_SYSTEM = """You write olympiad-style math problems.

You are given a SCAFFOLD: target answer, solution sketch, forced constraints, and concept tags.
Write ONE self-contained problem statement whose official answer equals the scaffold's target_answer (same mathematical object; formatting may differ slightly).

Hard rules:
- Single task only: one question, one thing to compute or prove at the end.
- No multipart worksheets (no labeled (a)(b)(c) parts, no "Find …; then find …").
- Keep the statement concise. Every named object should serve the single goal.
- The JSON field "topic" must match the TOPIC_LABEL given in the user message exactly.

Return only: {"question": "...", "answer": "...", "topic": "..."}. No markdown."""


def _question_looks_multipart_or_stitched(question: str) -> bool:
    q = question.strip()
    if len(q) > 2800:
        return True
    low = q.lower()
    if re.search(r"\(\s*[a-d]\s*\)\s*[\.\:]", low, re.I):
        return True
    if re.search(r"\(\s*[A-D]\s*\)\s*[\.\:]", low):
        return True
    if re.search(r"\bpart\s+[a-d]\b", low, re.I):
        return True
    if "following expression" in low or "compute the value of the following" in low:
        return True
    imperative_hits = len(re.findall(r"\b(find|compute|determine|calculate|evaluate)\b", low))
    if imperative_hits >= 3:
        return True
    if re.search(r";\s*(let|suppose|if)\s+", low, re.I):
        return True
    if low.count(" plus ") >= 2 or low.count(" and then ") >= 2:
        return True
    return False


def _spot_check_against_target(
    llm: LLMClient, model: str, question: str, claimed: str, target: str
) -> float:
    """Prefer agreement with scaffold target_answer; fall back to internal consistency."""
    if canonical_equal(claimed, target):
        return 1.0
    user = (
        f"PROBLEM:\n{question}\n\n"
        f"The intended official answer is:\n{target}\n\n"
        "Solve the problem. Return JSON: {\"answer\": \"<final answer in canonical form>\"}."
    )
    try:
        out = llm.chat_json(
            model=model,
            system="You are a math solver. Return strict JSON with the final answer only.",
            user=user,
            temperature=0.0,
        )
        ans = str(out.get("answer", "")) if isinstance(out, dict) else ""
    except Exception:
        return 0.0
    return 1.0 if canonical_equal(ans, target) else 0.0


def _extract_candidate(
    llm: LLMClient,
    cfg: Config,
    raw: dict[str, Any],
    scaffold_target: str,
    parent_anti_minhashes: list[list[int]],
    parent_anti_embeds: list[list[float]],
    embedder,
    spot_check_model: str,
    topic_label: str,
    event_store: Store | None,
    synth_attempt: int | None,
) -> Candidate | None:
    try:
        q = str(raw["question"]).strip()
        a = str(raw["answer"]).strip()
        t = str(raw["topic"]).strip()
        if not q or not a:
            return None
        if t != topic_label and topic_label:
            t = topic_label
    except Exception:
        return None
    if _question_looks_multipart_or_stitched(q):
        trace_event(
            event_store,
            "synthesis",
            "realize.candidate.rejected_multipart",
            attempt=synth_attempt,
            message="heuristic: multipart or stitched question",
            payload={"question_preview": q[:400]},
        )
        return None
    correctness = _spot_check_against_target(llm, spot_check_model, q, a, scaffold_target)
    cand_mh = minhash_signature(q, a)
    cand_emb = embedder.embed(q[:8000])
    cand_emb_arr = np.array(cand_emb, dtype=np.float32)
    max_mh_overlap = 0.0
    for mh in parent_anti_minhashes:
        max_mh_overlap = max(max_mh_overlap, minhash_jaccard(cand_mh, mh))
    max_cos = 0.0
    for pe in parent_anti_embeds:
        v = np.array(pe, dtype=np.float32)
        denom = (np.linalg.norm(cand_emb_arr) * np.linalg.norm(v))
        if denom > 0:
            max_cos = max(max_cos, float(cand_emb_arr @ v / denom))
    anti_distance = max(0.0, (1.0 - max_mh_overlap) * (1.0 - max_cos))
    return Candidate(question=q, answer=a, topic=t, correctness=correctness, anti_distance=anti_distance)


def write(
    llm: LLMClient,
    embedder,
    cfg: Config,
    scaffold: dict[str, Any],
    topic_label: str,
    parents: list[dict],
    rng: random.Random,
    event_store: Store | None = None,
    synth_attempt: int | None = None,
) -> tuple[Candidate | None, int]:
    """Best-of-K realizations. `parents` holds minhash+embedding for anti-near-parent scoring (often empty)."""
    target = str(scaffold.get("target_answer", "")).strip()
    if not target:
        return None, 0
    generator_model = rng.choice(cfg.models.generators)
    spot_check_model = cfg.models.verifiers[0]
    parent_mh = [p["minhash"] for p in parents]
    parent_emb = [p["embedding"] for p in parents]

    user = (
        f"TOPIC_LABEL (exact string for JSON topic field):\n{topic_label}\n\n"
        f"SCAFFOLD:\n{json.dumps(scaffold, ensure_ascii=False)[:12000]}\n\n"
        "Write the problem. Remember: single-task, one final answer matching the scaffold."
    )
    K = cfg.generator.candidates_per_round
    trace_event(
        event_store,
        "synthesis",
        "realize.start",
        attempt=synth_attempt,
        message=f"best-of-K · model {generator_model}",
        payload={"K": K},
    )
    candidates: list[Candidate] = []
    for ci in range(K):
        trace_event(
            event_store,
            "synthesis",
            "realize.candidate",
            attempt=synth_attempt,
            message=f"candidate {ci + 1}/{K}",
        )
        try:
            raw = llm.chat_json(
                model=generator_model,
                system=_SYSTEM,
                user=user,
                temperature=float(cfg.realizer.temperature),
            )
        except Exception:
            trace_event(event_store, "synthesis", "realize.candidate.error", attempt=synth_attempt)
            continue
        cand = _extract_candidate(
            llm,
            cfg,
            raw,
            target,
            parent_mh,
            parent_emb,
            embedder,
            spot_check_model,
            topic_label,
            event_store,
            synth_attempt,
        )
        if cand:
            candidates.append(cand)
    if not candidates:
        return None, 1
    candidates.sort(key=lambda c: c.combined(cfg), reverse=True)
    return candidates[0], 1
