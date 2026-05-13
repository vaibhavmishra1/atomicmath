"""Extract self-contained hinge notes for single-question mutation."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .config import Config
from .db import Store
from .llm import LLMClient
from .mutation_prompt import compact_text


HINGE_SYSTEM = """You analyze solved math problems for mutation-based problem generation.

Extract 2-3 self-contained hinge notes. A hinge is a mathematical bottleneck:
the concept, trick, lemma, equality case, transformation, trap, or logic where a
strong student may still get stuck.

Do not return broad topics like "algebra", "geometry", or "inequality" unless
they are part of a precise reasoning move. The hinge notes must be useful to a
future problem author without needing to reread the original problem."""


@dataclass
class ExtractedHinge:
    label: str
    text: str


def _label_from_text(text: str, fallback: str) -> str:
    for pat in (
        r"(?im)^HINGE_NAME:\s*(.+)$",
        r"(?im)^HINGE_LABEL:\s*(.+)$",
        r"(?im)^NAME:\s*(.+)$",
    ):
        m = re.search(pat, text)
        if m:
            raw = m.group(1).strip()
            label = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
            if label:
                return label[:80]
    return fallback


def _split_hinge_blocks(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?im)^\s*(?:HINGE\s+\d+|#\s*HINGE\s+\d+)\s*[:.\-]*\s*$", text)
    blocks = [p.strip() for p in parts if p.strip()]
    if len(blocks) >= 2:
        return blocks
    parts = re.split(r"(?im)^\s*---+\s*$", text)
    blocks = [p.strip() for p in parts if p.strip()]
    if len(blocks) >= 2:
        return blocks
    return [text]


def _hinge_prompt(seed: Any, max_hinges: int) -> str:
    question = compact_text(seed["question"], 7000)
    solution = compact_text(seed["solution_text"], 9000)
    answer = compact_text(seed["answer"], 1000)
    return f"""QUESTION:
{question}

FINAL ANSWER:
{answer or "(not provided separately)"}

REFERENCE SOLUTION:
{solution}

Extract {max_hinges} hinge notes.

Each hinge must use this format:

HINGE_NAME:
Short reusable name.

WHAT_THE_PROBLEM_TESTS:
The important mathematical concept, trick, rule, lemma, or logic.

WHY_THIS_IS_NONTRIVIAL:
Why students are likely to miss it.

COMMON_WRONG_MOVE:
The tempting but incorrect approach.

HOW_THE_SOLUTION_UNLOCKS:
The general solving move.

WHAT_A_GOOD_MUTATION_SHOULD_PRESERVE:
What must remain true for a new problem to still test this hinge.

Separate hinge notes with a line containing exactly:
---"""


def extract_hinges_for_seed(
    cfg: Config,
    store: Store,
    llm: LLMClient,
    seed_id: str,
    *,
    force: bool = False,
) -> list[ExtractedHinge]:
    seed = store.get_seed(seed_id)
    if seed is None:
        raise ValueError(f"unknown seed_id: {seed_id}")
    if len((seed["solution_text"] or "").strip()) < cfg.mutation.min_solution_chars:
        raise ValueError(f"seed {seed_id} has too little solution text for hinge extraction")
    if not force:
        existing = store.list_seed_hinges(seed_id)
        if existing:
            return [ExtractedHinge(label=r["label"], text=r["hinge_text"]) for r in existing]

    model = cfg.mutation.extraction_model or cfg.models.extractor
    prompt = _hinge_prompt(seed, cfg.mutation.max_hinges_per_seed)
    out = llm.chat(
        model=model,
        system=HINGE_SYSTEM,
        user=prompt,
        temperature=0.0,
        max_tokens=2200,
    )
    content = (out.get("content") or "").strip()
    if not content:
        raise ValueError(
            f"empty hinge extraction response from {model}; "
            f"finish_reason={out.get('finish_reason')!r}; usage={out.get('usage')!r}"
        )

    blocks = _split_hinge_blocks(content)[: cfg.mutation.max_hinges_per_seed]
    hinges = []
    for idx, block in enumerate(blocks, start=1):
        label = _label_from_text(block, f"hinge_{idx}")
        hinges.append(ExtractedHinge(label=label, text=block))
    if not hinges:
        raise ValueError("hinge extractor returned no usable hinge blocks")

    store.delete_seed_hinges(seed_id)
    for idx, hinge in enumerate(hinges, start=1):
        hid = hashlib.sha256(
            f"{seed_id}\x1e{idx}\x1e{hinge.text}".encode("utf-8")
        ).hexdigest()[:16]
        store.insert_seed_hinge(
            hinge_id=hid,
            seed_id=seed_id,
            ordinal=idx,
            hinge_text=hinge.text,
            label=hinge.label,
            model=model,
            prompt_version=cfg.mutation.prompt_version,
        )
    return hinges


def ensure_hinges_for_seed(cfg: Config, store: Store, llm: LLMClient, seed_id: str) -> list[Any]:
    existing = store.list_seed_hinges(seed_id)
    if existing:
        return existing
    extract_hinges_for_seed(cfg, store, llm, seed_id, force=False)
    return store.list_seed_hinges(seed_id)
