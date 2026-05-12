"""Verifier: correctness consensus + novelty (MinHash / embedding)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from datasketch import MinHashLSH

from .canonical import canonical_equal
from .solver import solve_for_answer
from .config import Config
from .db import Store
from .llm import LLMClient
from .trace import trace_event
from .minhash_util import NUM_PERM, minhash_jaccard


@dataclass
class VerdictCorrectness:
    passed: bool
    agreements: int
    answers: list[str]


@dataclass
class VerdictNovelty:
    passed: bool
    failure_kind: str | None
    nearest_minhash: float
    nearest_cosine: float


# --- Correctness --------------------------------------------------------------


def _llm_answers_equivalent(
    llm: LLMClient,
    cfg: Config,
    question: str,
    solver_answer: str,
    claimed_answer: str,
) -> tuple[bool, str | None]:
    """Fallback for textual/classification answers that symbolic canonicalization cannot normalize."""
    model = cfg.gate.answer_equivalence_model
    if not cfg.gate.answer_equivalence_fallback or not model:
        return False, None
    if not solver_answer.strip() or not claimed_answer.strip():
        return False, None
    user = (
        "Determine whether two final answers to the same math problem are mathematically equivalent.\n"
        "Use the problem only for context. Ignore harmless wording differences, variable renaming, "
        "and equivalent descriptions such as 'star trees K_{1,n}' vs 'trees of diameter at most 2'.\n"
        "Do NOT solve the problem from scratch. Compare the two answers.\n\n"
        f"PROBLEM:\n{question}\n\n"
        f"ANSWER_A (solver):\n{solver_answer}\n\n"
        f"ANSWER_B (claimed):\n{claimed_answer}\n\n"
        "Return strict JSON: {\"equivalent\": true/false, \"reason\": \"short explanation\"}."
    )
    try:
        out = llm.chat_json(
            model=model,
            system="You are a strict mathematical answer-equivalence judge. Return JSON only.",
            user=user,
            temperature=0.0,
        )
    except Exception:
        return False, None
    if not isinstance(out, dict):
        return False, None
    return bool(out.get("equivalent", False)), str(out.get("reason", "") or "")[:500]


def _answers_match(
    llm: LLMClient,
    cfg: Config,
    question: str,
    solver_answer: str,
    claimed_answer: str,
) -> tuple[bool, str, str | None]:
    if canonical_equal(solver_answer, claimed_answer):
        return True, "canonical", None
    equivalent, reason = _llm_answers_equivalent(llm, cfg, question, solver_answer, claimed_answer)
    if equivalent:
        return True, "llm_equivalence", reason
    return False, "none", reason


def _select_verifier_models(pool: list[str], n: int) -> list[str]:
    """Pick up to n models from pool, preferring distinct provider prefixes when n > 1."""
    if n <= 0 or not pool:
        return []
    if n == 1:
        return [pool[0]]
    chosen: list[str] = []
    used_fams: set[str] = set()
    for m in pool:
        fam = m.split("/")[0]
        if fam not in used_fams:
            chosen.append(m)
            used_fams.add(fam)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        rest = [m for m in pool if m not in chosen]
        chosen = (chosen + rest + pool)[:n]
    return chosen


def verify_correctness(
    llm: LLMClient,
    cfg: Config,
    question: str,
    claimed_answer: str,
    generator_model: str | None,
    event_store: Store | None = None,
    synth_attempt: int | None = None,
) -> VerdictCorrectness:
    n = cfg.gate.correctness_verifier_count
    pool = [m for m in cfg.models.verifiers if m != generator_model]
    chosen = _select_verifier_models(pool, n)
    if len(chosen) < n:
        chosen = (chosen + [m for m in cfg.models.verifiers if m not in chosen] + cfg.models.verifiers)[:n]
    answers: list[str] = []
    solver_rows: list[dict[str, object]] = []
    agree = 0
    trace_event(
        event_store,
        "synthesis",
        "verify.correctness.start",
        attempt=synth_attempt,
        message=f"{len(chosen)} solver(s) vs claimed answer",
        payload={"models": chosen, "consensus_need": cfg.gate.correctness_consensus},
    )
    for m in chosen:
        try:
            ans = solve_for_answer(llm, m, question, temperature=0.0)
        except Exception:
            ans = ""
        matched, match_method, equivalence_reason = _answers_match(llm, cfg, question, ans, claimed_answer)
        answers.append(ans)
        solver_rows.append(
            {
                "model": m,
                "solver_answer": ans,
                "matches_claimed": matched,
                "match_method": match_method,
                "equivalence_reason": equivalence_reason,
            }
        )
        if matched:
            agree += 1
    passed = agree >= cfg.gate.correctness_consensus
    trace_event(
        event_store,
        "synthesis",
        "verify.correctness.done",
        attempt=synth_attempt,
        payload={
            "question": question,
            "claimed_answer": claimed_answer,
            "agreements": agree,
            "passed": passed,
            "solver_outputs": solver_rows,
        },
    )
    return VerdictCorrectness(passed=passed, agreements=agree, answers=answers)


# --- Novelty ------------------------------------------------------------------


class NoveltyIndex:
    """In-memory index of seed + accepted-output MinHash / embeddings for novelty checks."""

    def __init__(self):
        self.lsh = MinHashLSH(threshold=0.5, num_perm=NUM_PERM)
        self.minhashes: list[list[int]] = []
        self.embeddings: list[np.ndarray] = []
        self._lsh_keys: list[str] = []

    def add(self, key: str, mh: list[int], emb: list[float]) -> None:
        # LSH for top-k MinHash retrieval
        from datasketch import MinHash
        m = MinHash(num_perm=NUM_PERM, hashvalues=np.array(mh, dtype=np.uint64))
        try:
            self.lsh.insert(key, m)
        except ValueError:
            pass  # duplicate key
        self.minhashes.append(mh)
        self.embeddings.append(np.array(emb, dtype=np.float32))
        self._lsh_keys.append(key)

    def load_from_store(self, store: Store) -> None:
        # Seeds
        for r in store.list_indexed_seeds():
            self.add(
                key=f"seed:{r['id']}",
                mh=json.loads(r["minhash"]),
                emb=json.loads(r["embedding"]),
            )
        # Outputs
        for r in store.list_outputs():
            self.add(
                key=f"out:{r['id']}",
                mh=json.loads(r["minhash"]),
                emb=json.loads(r["embedding"]),
            )

    def check(self, mh: list[int], emb: list[float], cfg: Config) -> VerdictNovelty:
        # Fast embedding/MinHash thresholds
        cand_emb = np.array(emb, dtype=np.float32)
        max_cos = 0.0
        for v in self.embeddings:
            denom = np.linalg.norm(cand_emb) * np.linalg.norm(v)
            if denom > 0:
                c = float(cand_emb @ v / denom)
                if c > max_cos:
                    max_cos = c
                    if max_cos >= cfg.gate.novelty_embed_max:
                        return VerdictNovelty(False, "embedding_paraphrase", 0.0, max_cos)
        max_mh = 0.0
        for m in self.minhashes:
            j = minhash_jaccard(mh, m)
            if j > max_mh:
                max_mh = j
                if max_mh >= cfg.gate.novelty_minhash_max:
                    return VerdictNovelty(False, "minhash_near_duplicate", max_mh, max_cos)
        return VerdictNovelty(True, None, max_mh, max_cos)
