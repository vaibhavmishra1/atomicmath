#!/usr/bin/env python3
"""Run the atomicmath mutation pipeline on sampled MathNet seeds.

This is the companion experiment to `run_mathnet_baseline.py`. It uses the same
MathNet sampling code, but generation goes through:

seed -> hinge extraction -> memory-aware mutation prompt -> candidate -> atomic judge

Optionally it also runs the same seed-relative comparison judge used by the
baseline script, so the reported metrics can be compared directly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from rich.console import Console

from atomicmath.config import load_config
from atomicmath.db import Store
from atomicmath.llm import EmbeddingClient, LLMClient
from atomicmath.minhash_util import minhash_jaccard, minhash_signature
from atomicmath.mutation_generator import generate_mutations_for_seed
from atomicmath.mutation_hinges import extract_hinges_for_seed, ensure_hinges_for_seed
from atomicmath.mutation_quality import judge_mutation_episode

from run_mathnet_baseline import (
    SeedRow,
    _cosine,
    _threshold_failure as comparison_threshold_failure,
    judge_baseline as comparison_judge,
    sample_mathnet,
)


console = Console()


def _topic_norm(topic: str) -> str:
    base = (topic or "uncategorized").split(">", 1)[0].strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_") or "uncategorized"


def _remove_db_files(db_path: Path) -> None:
    for path in [db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")]:
        if path.exists():
            path.unlink()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not math.isnan(float(r[key]))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _prepare_store(args: argparse.Namespace, seeds: list[SeedRow], cfg: Any) -> Store:
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.storage.db_path = str(db_path)
    store = Store(db_path)
    for seed in seeds:
        store.upsert_seed(
            seed.row_id,
            seed.question,
            seed.answer,
            seed.topic,
            solution_text=seed.solution,
        )
        store.set_topic_norm(seed.row_id, _topic_norm(seed.topic))
        store.set_eligible(seed.row_id, True)
    return store


def _episode_to_candidate(episode: Any) -> dict[str, str]:
    return {
        "new_question": episode["new_question"] or "",
        "answer": episode["answer"] or "",
        "short_solution": episode["short_solution"] or "",
        "relation_to_seed": episode["mutation_used"] or "",
        "why_novel": "",
        "risk_notes": "",
    }


def _score_value(scores: dict[str, Any], key: str) -> Any:
    value = scores.get(key)
    if isinstance(value, (int, float)):
        return value
    return None


def summarize(records: list[dict[str, Any]], *, comparison: bool) -> dict[str, Any]:
    generated = [r for r in records if r.get("generation_ok")]
    atomic_judged = [r for r in records if r.get("atomic_judge_ok")]
    comparison_judged = [r for r in records if r.get("comparison_judge_ok")]
    judged = comparison_judged if comparison else atomic_judged
    accepted_key = "comparison_accepted" if comparison else "atomic_accepted"
    accepted = [r for r in records if r.get(accepted_key)]

    failure_counts: dict[str, int] = {}
    atomic_failure_counts: dict[str, int] = {}
    for row in records:
        if row.get(accepted_key):
            failure = "accepted"
        else:
            failure = str(row.get("comparison_failure_kind" if comparison else "atomic_failure_kind") or "unknown_failure")
        failure_counts[failure] = failure_counts.get(failure, 0) + 1

        if row.get("atomic_accepted"):
            atomic_failure = "accepted"
        else:
            atomic_failure = str(row.get("atomic_failure_kind") or "unknown_failure")
        atomic_failure_counts[atomic_failure] = atomic_failure_counts.get(atomic_failure, 0) + 1

    metrics: dict[str, Any] = {
        "correctness_rate": sum(1 for r in atomic_judged if r.get("correctness_passed")) / len(atomic_judged)
        if atomic_judged
        else 0.0,
        "accepted_rate": len(accepted) / len(records) if records else 0.0,
        "generation_success_rate": len(generated) / len(records) if records else 0.0,
        "judge_success_rate": len(judged) / len(records) if records else 0.0,
        "mean_minhash_overlap": _mean(atomic_judged, "minhash_overlap"),
        "mean_embedding_cosine": _mean(atomic_judged, "embedding_cosine"),
    }
    if comparison:
        metrics.update(
            {
                "mean_depth_score": _mean(comparison_judged, "depth_score"),
                "mean_contest_score": _mean(comparison_judged, "contest_score"),
                "mean_novelty_score": _mean(comparison_judged, "novelty_score"),
                "mean_seed_alignment": _mean(comparison_judged, "seed_alignment"),
                "mean_non_stitched": _mean(comparison_judged, "comparison_non_stitched"),
                "mean_solution_economy": _mean(comparison_judged, "comparison_solution_economy"),
                "mean_routine_score": _mean(comparison_judged, "routine_score"),
            }
        )

    atomic_metrics = {
        "atomic_accepted_rate": sum(1 for r in records if r.get("atomic_accepted")) / len(records) if records else 0.0,
        "mean_hinge_preservation": _mean(atomic_judged, "hinge_preservation"),
        "mean_mutation_quality": _mean(atomic_judged, "mutation_quality"),
        "mean_sharpness": _mean(atomic_judged, "sharpness"),
        "mean_atomic_novelty": _mean(atomic_judged, "atomic_novelty"),
        "mean_atomic_non_stitched": _mean(atomic_judged, "atomic_non_stitched"),
        "mean_atomic_solution_economy": _mean(atomic_judged, "atomic_solution_economy"),
    }

    return {
        "total": len(records),
        "generated": len(generated),
        "judged": len(judged),
        "accepted": len(accepted),
        "rejected": len(records) - len(accepted),
        "failure_counts": dict(sorted(failure_counts.items())),
        "metrics": metrics,
        "atomic_accepted": sum(1 for r in records if r.get("atomic_accepted")),
        "atomic_rejected": len(records) - sum(1 for r in records if r.get("atomic_accepted")),
        "atomic_failure_counts": dict(sorted(atomic_failure_counts.items())),
        "atomic_metrics": atomic_metrics,
    }


def dataset_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for row in records:
        rows.append(
            {
                "id": f"atomicmath_{row.get('seed_id', '')}",
                "source_dataset": args.dataset,
                "source_config": args.dataset_config,
                "source_split": args.split,
                "source_seed_id": row.get("seed_id", ""),
                "source_topic": row.get("seed_topic", ""),
                "source_question": row.get("seed_question", ""),
                "source_answer": row.get("seed_answer", ""),
                "source_solution": row.get("seed_solution", ""),
                "proposed_question": row.get("proposed_question", ""),
                "proposed_answer": row.get("proposed_answer", ""),
                "proposed_solution": row.get("proposed_solution", ""),
                "mutation_used": row.get("mutation_used", ""),
                "atomic_accepted": bool(row.get("atomic_accepted", False)),
                "atomic_failure_kind": row.get("atomic_failure_kind") or "",
                "comparison_accepted": bool(row.get("comparison_accepted", False)),
                "comparison_failure_kind": row.get("comparison_failure_kind") or "",
                "correctness_passed": bool(row.get("correctness_passed", False)),
                "correctness_agreements": int(row.get("correctness_agreements") or 0),
                "correctness_answers": row.get("correctness_answers") or [],
                "minhash_overlap": row.get("minhash_overlap"),
                "embedding_cosine": row.get("embedding_cosine"),
                "hinge_ids": row.get("hinge_ids") or [],
                "hinges": row.get("hinges") or [],
                "hinge_preservation": row.get("hinge_preservation"),
                "mutation_quality": row.get("mutation_quality"),
                "sharpness": row.get("sharpness"),
                "atomic_novelty": row.get("atomic_novelty"),
                "atomic_non_stitched": row.get("atomic_non_stitched"),
                "atomic_solution_economy": row.get("atomic_solution_economy"),
                "depth_score": row.get("depth_score"),
                "contest_score": row.get("contest_score"),
                "novelty_score": row.get("novelty_score"),
                "seed_alignment": row.get("seed_alignment"),
                "comparison_non_stitched": row.get("comparison_non_stitched"),
                "comparison_solution_economy": row.get("comparison_solution_economy"),
                "routine_score": row.get("routine_score"),
                "atomic_reason": row.get("atomic_reason") or "",
                "comparison_reason": row.get("comparison_reason") or "",
                "generation_model": row.get("generation_model", ""),
                "judge_model": row.get("judge_model", ""),
                "verifier_model": row.get("verifier_model", ""),
                "extractor_model": row.get("extractor_model", ""),
                "episode_id": row.get("episode_id", ""),
                "elapsed_sec": row.get("elapsed_sec"),
                "raw_record_json": json.dumps(row, ensure_ascii=False),
            }
        )
    return rows


def publish_dataset(records: list[dict[str, Any]], args: argparse.Namespace, *, dry_run: bool = False) -> str:
    if not args.dataset_id:
        raise ValueError("--dataset-id is required when --push-to-hub is set")
    if "/" not in args.dataset_id:
        raise ValueError(f"dataset id must be owner/name, got {args.dataset_id!r}")
    rows = dataset_records(records, args)
    if not rows:
        raise ValueError("no records to publish")
    console.print(f"  atomicmath rows: {len(rows)}")
    console.print(f"  target dataset: {args.dataset_id} split={args.hub_split}")
    if dry_run:
        console.print("  dry run: not pushing to Hugging Face")
        return args.dataset_id
    Dataset.from_list(rows).push_to_hub(args.dataset_id, split=args.hub_split, private=args.private)
    return args.dataset_id


def _configure_models(cfg: Any, args: argparse.Namespace) -> tuple[str, str, str, str]:
    generator = args.model or cfg.mutation.generation_model or cfg.models.generators[0]
    judge = args.judge_model or (args.model if args.model else cfg.mutation.judge_model or cfg.models.judge)
    verifier = args.verifier_model or judge
    extractor = args.extraction_model or (args.model if args.model else cfg.mutation.extraction_model or cfg.models.extractor)
    cfg.mutation.generation_model = generator
    cfg.mutation.judge_model = judge
    cfg.mutation.extraction_model = extractor
    cfg.models.verifiers = [verifier]
    cfg.gate.correctness_verifier_count = args.verifier_count
    cfg.gate.correctness_consensus = min(cfg.gate.correctness_consensus, args.verifier_count)
    return generator, judge, verifier, extractor


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if args.debug_litellm:
        import litellm

        litellm._turn_on_debug()
    generator_model, judge_model, verifier_model, extractor_model = _configure_models(cfg, args)
    console.print(
        "[bold]atomicmath models[/] "
        f"extractor={extractor_model} generator={generator_model} judge={judge_model} verifier={verifier_model}"
    )

    out_path = Path(args.out)
    summary_path = Path(args.summary_out)
    db_path = Path(args.db_path)

    if args.publish_only:
        records = _load_jsonl(out_path)
        summary = summarize(records, comparison=not args.no_comparison_judge)
        repo_id = publish_dataset(records, args, dry_run=args.dry_run_push)
        summary["hub_dataset"] = repo_id
        summary["hub_split"] = args.hub_split
        summary["hub_pushed"] = not args.dry_run_push
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    seeds = sample_mathnet(args)
    console.print(f"[bold]sampled[/] {len(seeds)} MathNet seeds")
    if args.sample_only:
        return {"sampled": len(seeds), "seed_ids": [s.row_id for s in seeds]}

    if out_path.exists() and not args.resume:
        if not args.overwrite:
            raise RuntimeError(f"{out_path} already exists; use --resume or --overwrite")
        out_path.unlink()
        if summary_path.exists():
            summary_path.unlink()
        _remove_db_files(db_path)
    elif args.overwrite:
        _remove_db_files(db_path)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    store = _prepare_store(args, seeds, cfg)
    existing = _load_jsonl(out_path) if args.resume else []
    done_ids = {str(r.get("seed_id")) for r in existing}
    records = list(existing)
    if existing:
        console.print(f"[bold]resume[/] loaded {len(existing)} existing rows")

    llm = LLMClient(cache_dir=cfg.storage.cache_dir)
    embedder = None if args.no_embeddings else EmbeddingClient(cfg.storage.cache_dir, cfg.models.embedder)

    seed_by_id = {seed.row_id: seed for seed in seeds}
    for idx, seed in enumerate(seeds, start=1):
        if seed.row_id in done_ids:
            continue
        console.print(f"[bold]atomicmath seed {idx}/{len(seeds)}[/] {seed.row_id}")
        started = time.time()
        row: dict[str, Any] = {
            "seed_id": seed.row_id,
            "seed_topic": seed.topic,
            "seed_question": seed.question,
            "seed_answer": seed.answer,
            "seed_solution": seed.solution,
            "generation_model": generator_model,
            "judge_model": judge_model,
            "verifier_model": verifier_model,
            "extractor_model": extractor_model,
            "generation_ok": False,
            "atomic_judge_ok": False,
            "comparison_judge_ok": False,
            "atomic_accepted": False,
            "comparison_accepted": False,
            "atomic_failure_kind": None,
            "comparison_failure_kind": None,
            "error": None,
        }
        try:
            console.print("  extracting hinges")
            hinges = extract_hinges_for_seed(cfg, store, llm, seed.row_id, force=args.force_hinges)
            stored_hinges = ensure_hinges_for_seed(cfg, store, llm, seed.row_id)
            row["hinge_ids"] = [h["id"] for h in stored_hinges]
            row["hinges"] = [
                {"id": h["id"], "label": h["label"], "text": h["hinge_text"]}
                for h in stored_hinges
            ]
            console.print(f"  hinges: {len(hinges)}")

            console.print("  generating mutation")
            candidates = generate_mutations_for_seed(cfg, store, llm, seed.row_id, n=1)
            if not candidates:
                raise RuntimeError("mutation generator produced no candidates")
            candidate = candidates[0]
            row["generation_ok"] = True
            row["episode_id"] = candidate.episode_id
            row["proposed_question"] = candidate.new_question
            row["proposed_answer"] = candidate.answer
            row["proposed_solution"] = candidate.short_solution
            row["mutation_used"] = candidate.mutation_used
            console.print("  generated")

            console.print("  judging atomic mutation")
            verdict = judge_mutation_episode(cfg, store, llm, candidate.episode_id)
            episode = store.get_mutation_episode(candidate.episode_id)
            if episode is None:
                raise RuntimeError(f"missing mutation episode after judge: {candidate.episode_id}")
            try:
                scores = json.loads(episode["scores_json"] or "{}")
            except Exception:
                scores = {}
            row["atomic_judge_ok"] = True
            row["atomic_accepted"] = verdict.passed
            row["atomic_failure_kind"] = verdict.failure_kind
            row["correctness_passed"] = bool(scores.get("correctness_passed", False))
            row["correctness_agreements"] = scores.get("correctness_agreements") or 0
            row["correctness_answers"] = scores.get("correctness_answers") or []
            row["minhash_overlap"] = scores.get("minhash_overlap")
            row["hinge_preservation"] = _score_value(scores, "hinge_preservation")
            row["mutation_quality"] = _score_value(scores, "mutation_quality")
            row["sharpness"] = _score_value(scores, "sharpness")
            row["atomic_non_stitched"] = _score_value(scores, "non_stitched")
            row["atomic_solution_economy"] = _score_value(scores, "solution_economy")
            row["atomic_novelty"] = _score_value(scores, "novelty")
            row["atomic_reason"] = scores.get("reason") or verdict.story

            if row.get("minhash_overlap") is None:
                seed_mh = minhash_signature(seed.question, seed.answer)
                cand_mh = minhash_signature(candidate.new_question, candidate.answer)
                row["minhash_overlap"] = minhash_jaccard(seed_mh, cand_mh)

            embedding_cosine = None
            if embedder is not None:
                seed_emb = embedder.embed(seed.question[:8000])
                cand_emb = embedder.embed(candidate.new_question[:8000])
                embedding_cosine = _cosine(seed_emb, cand_emb)
            row["embedding_cosine"] = embedding_cosine

            if not args.no_comparison_judge:
                console.print("  judging comparison metrics")
                comp = comparison_judge(
                    cfg,
                    llm,
                    seed,
                    _episode_to_candidate(episode),
                    correctness_passed=bool(row["correctness_passed"]),
                    minhash_overlap=float(row["minhash_overlap"] or 0.0),
                    embedding_cosine=embedding_cosine,
                    model=judge_model,
                )
                comp_failure = comparison_threshold_failure(
                    cfg,
                    comp,
                    correctness_passed=bool(row["correctness_passed"]),
                    minhash_overlap=float(row["minhash_overlap"] or 0.0),
                    embedding_cosine=embedding_cosine,
                    max_minhash_overlap=cfg.mutation.max_seed_minhash_overlap,
                    max_embedding_cosine=cfg.gate.novelty_embed_max,
                )
                row["comparison_judge_ok"] = True
                row["depth_score"] = comp.get("depth_score")
                row["contest_score"] = comp.get("contest_score")
                row["novelty_score"] = comp.get("novelty_score")
                row["seed_alignment"] = comp.get("seed_alignment")
                row["comparison_non_stitched"] = comp.get("non_stitched")
                row["comparison_solution_economy"] = comp.get("solution_economy")
                row["routine_score"] = comp.get("routine_score")
                row["comparison_reason"] = comp.get("reason") or ""
                row["comparison_accepted"] = bool(comp.get("pass")) and comp_failure is None
                row["comparison_failure_kind"] = None if row["comparison_accepted"] else str(
                    comp_failure or comp.get("failure_kind") or "judge_reject"
                )

            console.print(
                f"  atomic={'[green]accepted[/]' if row['atomic_accepted'] else '[yellow]rejected[/]'} "
                f"{row['atomic_failure_kind'] or ''} "
                f"comparison={'[green]accepted[/]' if row.get('comparison_accepted') else '[yellow]rejected[/]'} "
                f"{row.get('comparison_failure_kind') or ''}"
            )
        except Exception as exc:
            row["error"] = str(exc)
            row["atomic_failure_kind"] = row.get("atomic_failure_kind") or "error"
            row["comparison_failure_kind"] = row.get("comparison_failure_kind") or "error"
            console.print(f"  [red]error[/] {exc}")
        finally:
            row["elapsed_sec"] = round(time.time() - started, 3)
            _append_jsonl(out_path, row)
            records.append(row)
            summary = summarize(records, comparison=not args.no_comparison_judge)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    summary = summarize(records, comparison=not args.no_comparison_judge)
    if args.push_to_hub:
        repo_id = publish_dataset(records, args, dry_run=args.dry_run_push)
        summary["hub_dataset"] = repo_id
        summary["hub_split"] = args.hub_split
        summary["hub_pushed"] = not args.dry_run_push
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run atomicmath mutation pipeline on MathNet.")
    p.add_argument("--config", default="examples/config.example.yaml")
    p.add_argument("--dataset", default="ShadenA/MathNet")
    p.add_argument("--dataset-config", default="United_States")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--scan-limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topic-contains", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--verifier-model", default=None)
    p.add_argument("--extraction-model", default=None)
    p.add_argument("--verifier-count", type=int, default=1)
    p.add_argument("--force-hinges", action="store_true")
    p.add_argument("--no-comparison-judge", action="store_true")
    p.add_argument("--no-embeddings", action="store_true")
    p.add_argument("--debug-litellm", action="store_true")
    p.add_argument("--sample-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--db-path", default="out/atomicmath/mathnet_atomicmath_100.db")
    p.add_argument("--out", default="out/atomicmath/mathnet_atomicmath_100.jsonl")
    p.add_argument("--summary-out", default="out/atomicmath/mathnet_atomicmath_100_summary.json")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--publish-only", action="store_true")
    p.add_argument("--dataset-id", default=None)
    p.add_argument("--hub-split", default="train")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run-push", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    console.print_json(data=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
