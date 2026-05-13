"""Publish accepted single-question mutations to Hugging Face."""
from __future__ import annotations

import json
from typing import Any

from datasets import Dataset
from rich.console import Console

from .config import Config
from .db import Store

console = Console()


def accepted_mutation_records(store: Store) -> list[dict[str, Any]]:
    """Return accepted mutation episodes as dataset-ready records."""
    with store._conn() as c:
        rows = c.execute(
            """SELECT
                   m.id AS mutation_id,
                   m.seed_id AS source_seed_id,
                   m.hinge_ids,
                   m.mutation_used,
                   m.new_question,
                   m.answer,
                   m.short_solution,
                   m.scores_json,
                   m.plan_json,
                   m.candidate_json,
                   m.story,
                   m.created_at,
                   m.updated_at,
                   s.question AS source_question,
                   s.answer AS source_answer,
                   s.topic_raw AS source_topic_raw,
                   s.topic_norm AS source_topic_norm,
                   s.solution_text AS source_solution
               FROM mutation_episodes m
               JOIN seeds s ON s.id = m.seed_id
               WHERE m.result = 'accepted'
                 AND m.new_question IS NOT NULL
                 AND trim(m.new_question) != ''
               ORDER BY m.updated_at ASC""",
        ).fetchall()

    records: list[dict[str, Any]] = []
    for row in rows:
        try:
            scores = json.loads(row["scores_json"] or "{}")
        except Exception:
            scores = {}
        try:
            plan = json.loads(row["plan_json"] or "{}")
        except Exception:
            plan = {}
        try:
            candidate_meta = json.loads(row["candidate_json"] or "{}")
        except Exception:
            candidate_meta = {}
        try:
            hinge_ids = json.loads(row["hinge_ids"] or "[]")
        except Exception:
            hinge_ids = []
        records.append(
            {
                "id": row["mutation_id"],
                "question": row["new_question"],
                "answer": row["answer"] or "",
                "solution": row["short_solution"] or "",
                "mutation_used": row["mutation_used"] or "",
                "source_seed_id": row["source_seed_id"],
                "source_question": row["source_question"],
                "source_answer": row["source_answer"] or "",
                "source_solution": row["source_solution"] or "",
                "source_topic_raw": row["source_topic_raw"] or "",
                "source_topic_norm": row["source_topic_norm"] or "",
                "hinge_ids": hinge_ids,
                "scores": scores,
                "transformation_plan": plan,
                "candidate_meta": candidate_meta,
                "mutation_story": row["story"] or "",
                "created_at": float(row["created_at"] or 0.0),
                "updated_at": float(row["updated_at"] or 0.0),
            }
        )
    return records


def publish_accepted_mutations(
    cfg: Config,
    store: Store,
    *,
    dataset_id: str | None = None,
    private: bool | None = None,
    split: str = "train",
    dry_run: bool = False,
) -> str:
    repo_id = dataset_id or cfg.output.dataset
    if "/" not in repo_id:
        raise ValueError(f"dataset id must be in owner/name form, got {repo_id!r}")
    records = accepted_mutation_records(store)
    if not records:
        raise ValueError("no accepted mutation episodes to publish")

    console.print(f"  accepted mutation rows: {len(records)}")
    console.print(f"  target dataset: {repo_id} split={split}")
    if dry_run:
        console.print("  dry run: not pushing to Hugging Face")
        return repo_id

    ds = Dataset.from_list(records)
    ds.push_to_hub(
        repo_id,
        split=split,
        private=cfg.output.private if private is None else private,
    )
    return repo_id
