"""Bootstrap exemplar scaffolds from indexed seeds; retrieve for composer conditioning."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from rich.console import Console

from .config import Config
from .curriculum import Brief
from .db import Store
from .llm import LLMClient
from .taxonomy import Taxonomy
from .trace import trace_event

console = Console()

_BOOT_SYSTEM = """You reverse-engineer a problem-design scaffold from a finished olympiad problem.

Given question text, an optional official final answer, and a reference solution, emit JSON with keys:
target_answer (string, usually same as the given final answer),
intended_solution_sketch (4–10 sentences summarizing the proof spine),
forced_constraints (array of 3–8 short strings that a new problem would need to encode the same ideas),
primary_concept, secondary_concept (each MUST be an exact token from the allowed concept list),
answer_form (MUST be one of the allowed question forms).

No difficulty bands or labels. JSON only."""


def _bootstrap_id(seed_id: str) -> str:
    return hashlib.sha256(f"exemplar:{seed_id}".encode()).hexdigest()[:16]


def _coerce_boot(raw: dict[str, Any], tax: Taxonomy) -> dict[str, Any] | None:
    try:
        ta = str(raw.get("target_answer", "")).strip()
        sk = str(raw.get("intended_solution_sketch", "")).strip()
        fc = raw.get("forced_constraints")
        pc = str(raw.get("primary_concept", "")).strip()
        sc = str(raw.get("secondary_concept", "")).strip()
        af = str(raw.get("answer_form", "")).strip()
    except Exception:
        return None
    if not ta or not sk or not isinstance(fc, list):
        return None
    fc2 = [str(x).strip() for x in fc if str(x).strip()]
    if len(fc2) < 2:
        return None
    if pc not in tax.concepts:
        pc = tax.concepts[0]
    if sc not in tax.concepts:
        sc = tax.concepts[min(1, len(tax.concepts) - 1)]
    if af not in tax.question_forms:
        af = tax.question_forms[0]
    return {
        "target_answer": ta,
        "intended_solution_sketch": sk,
        "forced_constraints": fc2,
        "primary_concept": pc,
        "secondary_concept": sc,
        "answer_form": af,
    }


def _bootstrap_one(
    llm: LLMClient,
    cfg: Config,
    tax: Taxonomy,
    seed_id: str,
    question: str,
    answer: str,
    solution: str,
    store: Store,
) -> bool:
    model = cfg.exemplar_bootstrap.model or cfg.models.extractor
    concepts = ", ".join(tax.concepts[:120])
    if len(tax.concepts) > 120:
        concepts += ", …"
    forms = ", ".join(tax.question_forms)
    answer_block = answer[:2000] if answer.strip() else "(not provided; infer the target answer from the reference solution)"
    user = (
        f"ALLOWED_CONCEPTS:\n{concepts}\n\nALLOWED_ANSWER_FORMS:\n{forms}\n\n"
        f"QUESTION:\n{question[:8000]}\n\nFINAL_ANSWER:\n{answer_block}\n\n"
        f"REFERENCE_SOLUTION:\n{solution[:12000]}"
    )
    raw = llm.chat_json(
        model=model,
        system=_BOOT_SYSTEM,
        user=user,
        temperature=0.0,
    )
    if not isinstance(raw, dict):
        return False
    coerced = _coerce_boot(raw, tax)
    if not coerced:
        return False
    store.insert_exemplar(_bootstrap_id(seed_id), seed_id, json.dumps(coerced, ensure_ascii=False))
    return True


def ensure_pool(cfg: Config, store: Store, llm: LLMClient, tax: Taxonomy) -> None:
    """Create exemplar rows from indexed seeds until cap (for composer few-shot)."""
    n_indexed = store.count_indexed_eligible_seeds()
    cap_in = cfg.input.max_seeds
    cap_boot = cfg.exemplar_bootstrap.max_bootstrap
    target = n_indexed
    if cap_in is not None:
        target = min(target, cap_in)
    if cap_boot is not None:
        target = min(target, cap_boot)
    if target <= 0:
        return
    have = store.count_exemplars()
    if have >= target:
        console.print(f"  exemplar pool ready ({have} ≥ target {target})")
        return
    console.print(f"  bootstrapping exemplars: {have} → target {target}")
    trace_event(store, "exemplars", "exemplars.bootstrap.start", message=f"target {target}")
    rows = store.list_indexed_seeds_for_bootstrap(limit=max(target * 3, target))
    rng = random.Random(0)
    rng.shuffle(rows)
    for row in rows:
        if store.count_exemplars() >= target:
            break
        sid = row["id"]
        if store.has_exemplar_for_seed(sid):
            continue
        sol = (row["solution_text"] or "").strip()
        if len(sol) < 20:
            continue
        ok = _bootstrap_one(
            llm,
            cfg,
            tax,
            sid,
            row["question"],
            row["answer"],
            sol,
            store,
        )
        trace_event(
            store,
            "exemplars",
            "exemplars.bootstrap.seed",
            seed_id=sid,
            message="ok" if ok else "failed",
        )
    console.print(f"  exemplar pool size: {store.count_exemplars()}")
    trace_event(store, "exemplars", "exemplars.bootstrap.done", payload={"n": store.count_exemplars()})


def retrieve(store: Store, brief: Brief, k: int) -> list[dict[str, Any]]:
    rows = store.list_exemplar_rows()
    scored: list[tuple[int, dict[str, Any]]] = []
    for r in rows:
        try:
            s = json.loads(r["scaffold_json"])
        except Exception:
            continue
        score = 0
        if s.get("answer_form") == brief.answer_form:
            score += 2
        if s.get("primary_concept") == brief.primary_concept:
            score += 3
        if s.get("secondary_concept") == brief.secondary_concept:
            score += 1
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[: max(1, k)]]
