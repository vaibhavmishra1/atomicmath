#!/usr/bin/env python3
"""Direct-generation baseline for MathNet seeds.

This script samples solved MathNet problems, asks a model to directly generate a
new problem from each seed, and evaluates the outputs with a seed-relative judge.
It intentionally does not use hinge extraction or global mutation memory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset
from rich.console import Console

from atomicmath.config import Config, load_config
from atomicmath.llm import EmbeddingClient, LLMClient
from atomicmath.minhash_util import minhash_jaccard, minhash_signature
from atomicmath.mutation_prompt import balanced_json_from_text, compact_text
from atomicmath.verifier import verify_correctness


console = Console()


BASELINE_SYSTEM = """You write contest-style math problems.

You are given one solved seed problem. Generate one new problem in the same broad
mathematical domain and at comparable difficulty, but do not paraphrase the seed.

Return JSON only."""


BASELINE_JUDGE_SYSTEM = """You are a strict contest-math generation evaluator.

You judge whether a generated problem is a good direct baseline output from a
seed problem. Score mathematical quality, novelty, and whether the problem is a
coherent single contest task. Return JSON only."""


@dataclass
class SeedRow:
    row_id: str
    question: str
    answer: str
    solution: str
    topic: str
    problem_type: str
    country: str


def _cell_str(row: dict[str, Any], key: str) -> str:
    v = row.get(key)
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return "\n\n".join(str(x).strip() for x in v if x is not None and str(x).strip())
    return str(v).strip()


def _topic(row: dict[str, Any]) -> str:
    topics = row.get("topics_flat")
    if isinstance(topics, list) and topics:
        return str(topics[0]).strip()
    return _cell_str(row, "topics_flat") or "uncategorized"


def _clean_question(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text or "").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def _is_englishish(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return ascii_chars / max(1, len(text)) >= 0.80


def _eligible_mathnet_row(row: dict[str, Any]) -> SeedRow | None:
    question = _clean_question(_cell_str(row, "problem_markdown"))
    solution = _cell_str(row, "solutions_markdown")
    answer = _cell_str(row, "final_answer")
    if not question or not solution:
        return None
    if not _is_englishish(question):
        return None
    if row.get("images"):
        figure_words = re.search(
            r"\b(as shown|in the figure|in the diagram|see the figure|the drawing)\b",
            question,
            re.I,
        )
        if figure_words:
            return None
    problem_type = _cell_str(row, "problem_type")
    if problem_type and problem_type not in {"final answer only", "proof and answer"}:
        return None
    return SeedRow(
        row_id=_cell_str(row, "id") or str(hash(question)),
        question=question,
        answer=answer,
        solution=solution,
        topic=_topic(row),
        problem_type=problem_type,
        country=_cell_str(row, "country"),
    )


def sample_mathnet(args: argparse.Namespace) -> list[SeedRow]:
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    indices = list(range(len(ds)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    rows: list[SeedRow] = []
    scanned = 0
    for idx in indices:
        if args.scan_limit and scanned >= args.scan_limit:
            break
        scanned += 1
        row = _eligible_mathnet_row(dict(ds[int(idx)]))
        if row is None:
            continue
        if args.topic_contains and args.topic_contains.lower() not in row.topic.lower():
            continue
        rows.append(row)
        if len(rows) >= args.limit:
            break
    if len(rows) < args.limit:
        raise RuntimeError(f"only found {len(rows)} eligible rows; requested {args.limit}")
    return rows


def _baseline_prompt(seed: SeedRow) -> str:
    return f"""Generate one new contest-style math problem from this seed.

The task is a direct baseline: use only the seed question and solution. Do not use
hinge notes, mutation memory, or any hidden pipeline information.

SEED TOPIC:
{seed.topic}

SEED QUESTION:
{compact_text(seed.question, 7000)}

SEED ANSWER:
{compact_text(seed.answer, 1000) or "(not separately provided)"}

SEED SOLUTION:
{compact_text(seed.solution, 9000)}

Requirements:
- Same broad topic and comparable difficulty.
- One self-contained final task.
- No multipart worksheet.
- Do not paraphrase the seed.
- Do not merely change constants, names, signs, or exponents.
- Difficulty should come from mathematical reasoning, not long arithmetic.
- Provide a short correct solution.

Return strict JSON with exactly these keys:
{{
  "new_question": "...",
  "answer": "...",
  "short_solution": "...",
  "relation_to_seed": "...",
  "why_novel": "...",
  "risk_notes": "..."
}}"""


def generate_baseline(
    cfg: Config,
    llm: LLMClient,
    seed: SeedRow,
    *,
    model: str,
    temperature: float,
) -> tuple[dict[str, Any] | None, str | None]:
    out = llm.chat(
        model=model,
        system=BASELINE_SYSTEM,
        user=_baseline_prompt(seed),
        temperature=temperature,
        max_tokens=2400,
        use_cache=False,
    )
    content = (out.get("content") or "").strip()
    raw = balanced_json_from_text(content)
    if not isinstance(raw, dict):
        return None, (
            f"could not parse generation JSON: {content[:300]!r}; "
            f"finish_reason={out.get('finish_reason')!r}; usage={out.get('usage')!r}"
        )
    for key in ("new_question", "answer", "short_solution"):
        if not str(raw.get(key, "") or "").strip():
            return None, f"generation missing required key: {key}"
    return raw, None


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0:
        return 0.0
    return float(va @ vb / denom)


def _judge_prompt(
    seed: SeedRow,
    candidate: dict[str, Any],
    *,
    correctness_passed: bool,
    minhash_overlap: float,
    embedding_cosine: float | None,
) -> str:
    emb = "not computed" if embedding_cosine is None else f"{embedding_cosine:.3f}"
    return f"""Evaluate this generated math problem against its seed.

SEED TOPIC:
{seed.topic}

SEED QUESTION:
{seed.question}

SEED ANSWER:
{seed.answer}

SEED SOLUTION:
{compact_text(seed.solution, 8000)}

GENERATED QUESTION:
{candidate["new_question"]}

GENERATED ANSWER:
{candidate["answer"]}

GENERATED SHORT SOLUTION:
{candidate["short_solution"]}

GENERATOR SELF-EXPLANATION:
relation_to_seed: {candidate.get("relation_to_seed", "")}
why_novel: {candidate.get("why_novel", "")}
risk_notes: {candidate.get("risk_notes", "")}

AUTOMATIC SIGNALS:
correctness_passed: {correctness_passed}
minhash_overlap_with_seed: {minhash_overlap:.3f}
embedding_cosine_with_seed: {emb}

Score the candidate.

Reject if it is:
- mathematically incorrect or answer/solution are unreliable;
- a near paraphrase or direct sibling of the seed;
- only a number/sign/name/context swap;
- a routine exercise with no contest-style idea;
- stitched together from unrelated tasks;
- ambiguous or underspecified.

Return strict JSON:
{{
  "pass": true/false,
  "depth_score": number from 0 to 1,
  "contest_score": number from 0 to 1,
  "novelty_score": number from 0 to 1,
  "seed_alignment": number from 0 to 1,
  "non_stitched": number from 0 to 1,
  "solution_economy": number from 0 to 1,
  "routine_score": number from 0 to 1,
  "failure_kind": null or one of ["incorrect", "near_paraphrase", "weak_quality", "routine", "stitched", "ambiguous", "off_distribution", "judge_reject"],
  "reason": "short explanation"
}}"""


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, v))


def _parse_judge(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "pass": False,
            "depth_score": 0.0,
            "contest_score": 0.0,
            "novelty_score": 0.0,
            "seed_alignment": 0.0,
            "non_stitched": 0.0,
            "solution_economy": 0.0,
            "routine_score": 1.0,
            "failure_kind": "judge_reject",
            "reason": "judge returned non-object JSON",
        }
    out = {
        "pass": bool(raw.get("pass", False)),
        "depth_score": _clamp01(raw.get("depth_score")),
        "contest_score": _clamp01(raw.get("contest_score")),
        "novelty_score": _clamp01(raw.get("novelty_score")),
        "seed_alignment": _clamp01(raw.get("seed_alignment")),
        "non_stitched": _clamp01(raw.get("non_stitched")),
        "solution_economy": _clamp01(raw.get("solution_economy")),
        "routine_score": _clamp01(raw.get("routine_score"), 1.0),
        "failure_kind": raw.get("failure_kind"),
        "reason": str(raw.get("reason", "") or "")[:1200],
    }
    if out["failure_kind"] in ("", "null", "None"):
        out["failure_kind"] = None
    return out


def judge_baseline(
    cfg: Config,
    llm: LLMClient,
    seed: SeedRow,
    candidate: dict[str, Any],
    *,
    correctness_passed: bool,
    minhash_overlap: float,
    embedding_cosine: float | None,
    model: str,
) -> dict[str, Any]:
    out = llm.chat(
        model=model,
        system=BASELINE_JUDGE_SYSTEM,
        user=_judge_prompt(
            seed,
            candidate,
            correctness_passed=correctness_passed,
            minhash_overlap=minhash_overlap,
            embedding_cosine=embedding_cosine,
        ),
        temperature=0.0,
        max_tokens=1600,
    )
    raw = balanced_json_from_text(out.get("content") or "")
    return _parse_judge(raw)


def _threshold_failure(
    cfg: Config,
    judge: dict[str, Any],
    *,
    correctness_passed: bool,
    minhash_overlap: float,
    embedding_cosine: float | None,
    max_minhash_overlap: float,
    max_embedding_cosine: float,
) -> str | None:
    if not correctness_passed:
        return "incorrect"
    if minhash_overlap > max_minhash_overlap:
        return "near_paraphrase"
    if embedding_cosine is not None and embedding_cosine > max_embedding_cosine:
        return "near_paraphrase"
    if judge["depth_score"] < cfg.quality.min_depth_score:
        return "weak_quality"
    if judge["contest_score"] < cfg.quality.min_contest_score:
        return "weak_quality"
    if judge["novelty_score"] < cfg.mutation.min_novelty:
        return "near_paraphrase"
    if judge["non_stitched"] < cfg.mutation.min_non_stitched:
        return "stitched"
    if judge["solution_economy"] < cfg.mutation.min_solution_economy:
        return "routine"
    if judge["routine_score"] > cfg.quality.max_routine_score:
        return "routine"
    return None


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not math.isnan(float(r[key]))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [r for r in records if r.get("generation_ok")]
    judged = [r for r in records if r.get("judge_ok")]
    accepted = [r for r in records if r.get("accepted")]
    failure_counts: dict[str, int] = {}
    for r in records:
        if r.get("accepted"):
            failure = "accepted"
        else:
            failure = str(r.get("failure_kind") or "unknown_failure")
        failure_counts[failure] = failure_counts.get(failure, 0) + 1
    metrics = {
        "correctness_rate": sum(1 for r in judged if r.get("correctness_passed")) / len(judged) if judged else 0.0,
        "accepted_rate": len(accepted) / len(records) if records else 0.0,
        "generation_success_rate": len(attempted) / len(records) if records else 0.0,
        "judge_success_rate": len(judged) / len(records) if records else 0.0,
        "mean_minhash_overlap": _mean(judged, "minhash_overlap"),
        "mean_embedding_cosine": _mean(judged, "embedding_cosine"),
        "mean_depth_score": _mean(judged, "depth_score"),
        "mean_contest_score": _mean(judged, "contest_score"),
        "mean_novelty_score": _mean(judged, "novelty_score"),
        "mean_seed_alignment": _mean(judged, "seed_alignment"),
        "mean_non_stitched": _mean(judged, "non_stitched"),
        "mean_solution_economy": _mean(judged, "solution_economy"),
        "mean_routine_score": _mean(judged, "routine_score"),
    }
    return {
        "total": len(records),
        "generated": len(attempted),
        "judged": len(judged),
        "accepted": len(accepted),
        "rejected": len(records) - len(accepted),
        "failure_counts": dict(sorted(failure_counts.items())),
        "metrics": metrics,
    }


def baseline_dataset_records(
    records: list[dict[str, Any]],
    *,
    source_dataset: str,
    source_config: str,
    source_split: str,
) -> list[dict[str, Any]]:
    """Flatten JSONL experiment rows into a Hub-friendly comparison dataset."""
    rows: list[dict[str, Any]] = []
    for record in records:
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
        rows.append(
            {
                "id": f"baseline_{record.get('seed_id', '')}",
                "source_dataset": source_dataset,
                "source_config": source_config,
                "source_split": source_split,
                "source_seed_id": record.get("seed_id", ""),
                "source_topic": record.get("seed_topic", ""),
                "source_question": record.get("seed_question", ""),
                "source_answer": record.get("seed_answer", ""),
                "source_solution": record.get("seed_solution", ""),
                "proposed_question": candidate.get("new_question", ""),
                "proposed_answer": candidate.get("answer", ""),
                "proposed_solution": candidate.get("short_solution", ""),
                "relation_to_seed": candidate.get("relation_to_seed", ""),
                "why_novel": candidate.get("why_novel", ""),
                "risk_notes": candidate.get("risk_notes", ""),
                "generation_model": record.get("generation_model", ""),
                "judge_model": record.get("judge_model", ""),
                "verifier_model": record.get("verifier_model", ""),
                "generation_ok": bool(record.get("generation_ok", False)),
                "judge_ok": bool(record.get("judge_ok", False)),
                "accepted": bool(record.get("accepted", False)),
                "failure_kind": record.get("failure_kind") or "",
                "error": record.get("error") or "",
                "correctness_passed": bool(record.get("correctness_passed", False)),
                "correctness_agreements": int(record.get("correctness_agreements") or 0),
                "correctness_answers": record.get("correctness_answers") or [],
                "minhash_overlap": record.get("minhash_overlap"),
                "embedding_cosine": record.get("embedding_cosine"),
                "depth_score": record.get("depth_score"),
                "contest_score": record.get("contest_score"),
                "novelty_score": record.get("novelty_score"),
                "seed_alignment": record.get("seed_alignment"),
                "non_stitched": record.get("non_stitched"),
                "solution_economy": record.get("solution_economy"),
                "routine_score": record.get("routine_score"),
                "judge_reason": record.get("reason") or "",
                "elapsed_sec": record.get("elapsed_sec"),
                "raw_candidate_json": json.dumps(candidate, ensure_ascii=False),
                "raw_record_json": json.dumps(record, ensure_ascii=False),
            }
        )
    return rows


def publish_baseline_dataset(
    records: list[dict[str, Any]],
    *,
    dataset_id: str,
    split: str,
    private: bool,
    source_dataset: str,
    source_config: str,
    source_split: str,
    dry_run: bool = False,
) -> str:
    if "/" not in dataset_id:
        raise ValueError(f"dataset id must be in owner/name form, got {dataset_id!r}")
    rows = baseline_dataset_records(
        records,
        source_dataset=source_dataset,
        source_config=source_config,
        source_split=source_split,
    )
    if not rows:
        raise ValueError("no baseline records to publish")

    accepted = sum(1 for r in rows if r["accepted"])
    console.print(f"  baseline rows: {len(rows)} accepted={accepted}")
    console.print(f"  target dataset: {dataset_id} split={split}")
    if dry_run:
        console.print("  dry run: not pushing to Hugging Face")
        return dataset_id

    ds = Dataset.from_list(rows)
    ds.push_to_hub(dataset_id, split=split, private=private)
    return dataset_id


def _load_existing(path: Path) -> list[dict[str, Any]]:
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if args.debug_litellm:
        import litellm

        litellm._turn_on_debug()
    if args.verifier_count is not None:
        cfg.gate.correctness_verifier_count = args.verifier_count
        cfg.gate.correctness_consensus = min(cfg.gate.correctness_consensus, args.verifier_count)
    model = args.model or cfg.mutation.generation_model or cfg.models.generators[0]
    judge_model = args.judge_model or (args.model if args.model else cfg.mutation.judge_model or cfg.models.judge)
    verifier_model = args.verifier_model or judge_model
    cfg.models.verifiers = [verifier_model]
    console.print(
        f"[bold]baseline model[/] generator={model} judge={judge_model} verifier={verifier_model}"
    )
    max_minhash_overlap = args.max_minhash_overlap
    if max_minhash_overlap is None:
        max_minhash_overlap = cfg.mutation.max_seed_minhash_overlap
    max_embedding_cosine = args.max_embedding_cosine or cfg.gate.novelty_embed_max
    out_path = Path(args.out)
    summary_path = Path(args.summary_out)

    if args.publish_only:
        if not args.push_to_hub:
            raise ValueError("--publish-only requires --push-to-hub")
        if not args.dataset_id:
            raise ValueError("--dataset-id is required when --push-to-hub is set")
        records = _load_existing(out_path)
        if not records:
            raise ValueError(f"no baseline records found at {out_path}")
        summary = summarize(records)
        repo_id = publish_baseline_dataset(
            records,
            dataset_id=args.dataset_id,
            split=args.hub_split,
            private=args.private,
            source_dataset=args.dataset,
            source_config=args.dataset_config,
            source_split=args.split,
            dry_run=args.dry_run_push,
        )
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
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; export it before running the baseline")
    if out_path.exists() and not args.resume:
        if not args.overwrite:
            raise RuntimeError(f"{out_path} already exists; use --resume or --overwrite")
        out_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

    existing = _load_existing(out_path) if args.resume else []
    done_ids = {str(r.get("seed_id")) for r in existing}
    if existing:
        console.print(f"[bold]resume[/] loaded {len(existing)} existing rows from {out_path}")

    llm = LLMClient(cache_dir=cfg.storage.cache_dir)
    embedder = None if args.no_embeddings else EmbeddingClient(cfg.storage.cache_dir, cfg.models.embedder)
    records = list(existing)

    for idx, seed in enumerate(seeds, start=1):
        if seed.row_id in done_ids:
            continue
        console.print(f"[bold]baseline seed {idx}/{len(seeds)}[/] {seed.row_id}")
        started = time.time()
        record: dict[str, Any] = {
            "seed_id": seed.row_id,
            "seed_topic": seed.topic,
            "seed_question": seed.question,
            "seed_answer": seed.answer,
            "seed_solution": seed.solution,
            "generation_model": model,
            "judge_model": judge_model,
            "verifier_model": verifier_model,
            "generation_ok": False,
            "judge_ok": False,
            "accepted": False,
            "failure_kind": None,
            "error": None,
        }
        should_write_record = True
        try:
            candidate, error = generate_baseline(
                cfg,
                llm,
                seed,
                model=model,
                temperature=args.temperature,
            )
            if error:
                record["failure_kind"] = "generation_error"
                record["error"] = error
                console.print(f"  [red]generation error[/] {error}")
                continue
            assert candidate is not None
            record["generation_ok"] = True
            record["candidate"] = candidate
            console.print("  generated")

            correctness = verify_correctness(
                llm,
                cfg,
                candidate["new_question"],
                candidate["answer"],
                generator_model=model,
            )
            record["correctness_passed"] = correctness.passed
            record["correctness_agreements"] = correctness.agreements
            record["correctness_answers"] = correctness.answers

            seed_mh = minhash_signature(seed.question, seed.answer)
            cand_mh = minhash_signature(candidate["new_question"], candidate["answer"])
            minhash_overlap = minhash_jaccard(seed_mh, cand_mh)
            record["minhash_overlap"] = minhash_overlap

            embedding_cosine = None
            if embedder is not None:
                seed_emb = embedder.embed(seed.question[:8000])
                cand_emb = embedder.embed(candidate["new_question"][:8000])
                embedding_cosine = _cosine(seed_emb, cand_emb)
            record["embedding_cosine"] = embedding_cosine

            judge = judge_baseline(
                cfg,
                llm,
                seed,
                candidate,
                correctness_passed=correctness.passed,
                minhash_overlap=minhash_overlap,
                embedding_cosine=embedding_cosine,
                model=judge_model,
            )
            record["judge_ok"] = True
            record.update({k: v for k, v in judge.items() if k != "pass"})
            threshold_failure = _threshold_failure(
                cfg,
                judge,
                correctness_passed=correctness.passed,
                minhash_overlap=minhash_overlap,
                embedding_cosine=embedding_cosine,
                max_minhash_overlap=float(max_minhash_overlap),
                max_embedding_cosine=float(max_embedding_cosine),
            )
            failure_kind = threshold_failure or judge.get("failure_kind")
            accepted = bool(judge["pass"]) and failure_kind is None
            record["accepted"] = accepted
            record["failure_kind"] = None if accepted else str(failure_kind or "judge_reject")
            console.print(
                f"  {'[green]accepted[/]' if accepted else '[yellow]rejected[/]'} "
                f"correct={correctness.passed} mh={minhash_overlap:.3f} "
                f"emb={embedding_cosine if embedding_cosine is not None else 'NA'} "
                f"failure={record['failure_kind']}"
            )
        except Exception as exc:
            record["failure_kind"] = "error"
            record["error"] = str(exc)
            console.print(f"  [red]error[/] {exc}")
        except BaseException as exc:
            record["failure_kind"] = "interrupted"
            record["error"] = f"{exc.__class__.__name__}: {exc}"
            console.print(f"  [red]interrupted[/] {record['error']}")
            raise
        finally:
            record["elapsed_sec"] = round(time.time() - started, 3)
            if (
                should_write_record
                and not record.get("generation_ok")
                and not record.get("failure_kind")
                and not record.get("error")
            ):
                record["failure_kind"] = "incomplete_generation"
                record["error"] = "generation call ended before a candidate or explicit error was recorded"
            if should_write_record:
                _append_jsonl(out_path, record)
                records.append(record)
                summary = summarize(records)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    summary = summarize(records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.push_to_hub:
        dataset_id = args.dataset_id
        if not dataset_id:
            raise ValueError("--dataset-id is required when --push-to-hub is set")
        repo_id = publish_baseline_dataset(
            records,
            dataset_id=dataset_id,
            split=args.hub_split,
            private=args.private,
            source_dataset=args.dataset,
            source_config=args.dataset_config,
            source_split=args.split,
            dry_run=args.dry_run_push,
        )
        summary["hub_dataset"] = repo_id
        summary["hub_split"] = args.hub_split
        summary["hub_pushed"] = not args.dry_run_push
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run direct-generation baseline on MathNet.")
    p.add_argument("--config", default="examples/config.example.yaml")
    p.add_argument("--dataset", default="ShadenA/MathNet")
    p.add_argument("--dataset-config", default="United_States")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--scan-limit", type=int, default=0, help="Max shuffled rows to scan; 0 scans until limit found.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topic-contains", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--verifier-model", default=None, help="Correctness solver model; defaults to --judge-model.")
    p.add_argument("--debug-litellm", action="store_true", help="Enable verbose LiteLLM request diagnostics.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--verifier-count", type=int, default=1)
    p.add_argument("--max-minhash-overlap", type=float, default=None)
    p.add_argument("--max-embedding-cosine", type=float, default=None)
    p.add_argument("--no-embeddings", action="store_true")
    p.add_argument("--sample-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Replace existing --out/--summary-out files before running.")
    p.add_argument("--out", default="out/baseline/mathnet_direct_baseline_100.jsonl")
    p.add_argument("--summary-out", default="out/baseline/mathnet_direct_baseline_100_summary.json")
    p.add_argument("--push-to-hub", action="store_true", help="Upload JSONL results as a Hugging Face dataset.")
    p.add_argument("--publish-only", action="store_true", help="Upload an existing --out JSONL without running generation.")
    p.add_argument("--dataset-id", default=None, help="HF dataset id for baseline upload, e.g. vibhuiitj/mathnet-baseline.")
    p.add_argument("--hub-split", default="train")
    p.add_argument("--private", action="store_true", help="Create/update the Hub dataset as private.")
    p.add_argument("--dry-run-push", action="store_true", help="Prepare upload rows but do not push to Hugging Face.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    console.print_json(data=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
