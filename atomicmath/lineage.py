"""Lineage-first math problem synthesis.

This module intentionally avoids hinge extraction. The unit of work is a solved
problem, and the system grows a lineage by repeatedly asking for nontrivial
transformations, refining them, judging them, then carrying the selected problem
forward as the next iteration.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset, load_dataset
from rich.console import Console

from .config import Config
from .llm import LLMClient

console = Console()
EventSink = Callable[[dict[str, Any]], None]


@dataclass
class SeedRecord:
    index: int
    source_id: str
    source_iteration: int
    question: str
    answer: str
    memory: str
    raw: dict[str, Any]


@dataclass
class Candidate:
    label: str
    question: str
    answer: str
    solution: str
    transformation: str
    why_new: str
    risk: str
    raw_text: str
    scores: dict[str, float] = field(default_factory=dict)
    correct: bool = False
    decision: str = "reject"
    failure_kind: str = ""
    judge_notes: str = ""

    def to_artifact(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "question": self.question,
            "answer": self.answer,
            "solution": self.solution,
            "transformation": self.transformation,
            "why_new": self.why_new,
            "risk": self.risk,
            "scores": self.scores,
            "correct": self.correct,
            "decision": self.decision,
            "failure_kind": self.failure_kind,
            "judge_notes": self.judge_notes,
        }


@dataclass
class LineageResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    local_path: str
    summary_path: str
    hub_dataset: str | None = None
    hub_pushed: bool = False
    current_iteration: int = 1


def _stable_id(*parts: Any, size: int = 16) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:size]


def _trim(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 120].rstrip() + "\n...[trimmed]...\n" + text[-100:].lstrip()


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n\n".join(_field_text(item) for item in value if _field_text(item)).strip()
    if isinstance(value, dict):
        for key in ("solution", "answer", "text", "content", "markdown"):
            if key in value:
                return _field_text(value.get(key))
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value).strip()


def _dataset_kwargs(cfg: Config) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"path": cfg.input.dataset, "split": cfg.input.split}
    if cfg.input.config_name:
        kwargs["name"] = cfg.input.config_name
    return kwargs


def _parse_iteration(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = re.search(r"-?\d+", str(value))
    return int(match.group(0)) if match else None


def _extract_memory(row: dict[str, Any], memory_field: str) -> str:
    value = row.get(memory_field)
    if value is None and memory_field != "memory":
        value = row.get("memory")
    if value is None:
        value = row.get("memory_json")
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value).strip()


def _is_parent_candidate(row: dict[str, Any]) -> bool:
    role = str(row.get("role") or "").strip().lower()
    if role == "rejected":
        return False
    accepted = row.get("accepted")
    if isinstance(accepted, bool):
        return accepted
    if accepted is not None and str(accepted).strip().lower() in {"false", "0", "no", "rejected"}:
        return False
    return True


def load_seed_records(
    cfg: Config,
    *,
    num_seeds: int | None = None,
    seed: int | None = None,
) -> tuple[list[SeedRecord], int, int]:
    """Load rows from the latest available iteration and infer the next one.

    If the configured iteration column exists, only rows from the current max
    iteration are used as parents and generated rows get max+1. If the column is
    absent, the dataset is treated as raw seed data and generated rows get
    iteration 1.
    """

    ds = load_dataset(**_dataset_kwargs(cfg))
    q_field = cfg.input.question_field
    a_field = cfg.input.answer_field
    iteration_field = cfg.input.iteration_field
    memory_field = cfg.input.memory_field
    missing = [name for name in (q_field, a_field) if name not in ds.column_names]
    if missing:
        raise ValueError(
            f"input dataset is missing required column(s): {missing}. "
            f"Available columns: {list(ds.column_names)}"
        )

    has_iteration = bool(iteration_field and iteration_field in ds.column_names)
    source_iteration = 0
    current_iteration = 1
    if has_iteration:
        parsed_iterations = [
            parsed
            for row in ds
            if _is_parent_candidate(dict(row))
            and (parsed := _parse_iteration(row.get(iteration_field))) is not None
        ]
        if parsed_iterations:
            source_iteration = max(parsed_iterations)
            current_iteration = source_iteration + 1

    valid: list[int] = []
    for idx, row in enumerate(ds):
        question = _field_text(row.get(q_field))
        answer = _field_text(row.get(a_field))
        row_iteration = _parse_iteration(row.get(iteration_field)) if has_iteration else None
        if (
            question
            and answer
            and _is_parent_candidate(dict(row))
            and (not has_iteration or row_iteration == source_iteration)
        ):
            valid.append(idx)

    rng = random.Random(cfg.input.seed if seed is None else seed)
    rng.shuffle(valid)
    limit = cfg.input.max_seeds if num_seeds is None else num_seeds
    chosen = valid[:limit]

    records: list[SeedRecord] = []
    for source_position, idx in enumerate(chosen):
        row = dict(ds[int(idx)])
        source_id = str(row.get("id") or row.get("row_id") or row.get("problem_id") or idx)
        records.append(
            SeedRecord(
                index=source_position,
                source_id=source_id,
                source_iteration=source_iteration,
                question=_field_text(row.get(q_field)),
                answer=_field_text(row.get(a_field)),
                memory=_extract_memory(row, memory_field),
                raw=row,
            )
        )
    return records, source_iteration, current_iteration


def _memory_block(title: str, items: list[str]) -> str:
    if not items:
        return f"{title}: none yet"
    body = "\n".join(f"- {item}" for item in items)
    return f"{title}:\n{body}"


def _generation_prompt(
    cfg: Config,
    *,
    seed: SeedRecord,
    current_question: str,
    current_answer: str,
    iteration: int,
    local_memory: list[str],
    global_memory: list[str],
) -> str:
    parent_memory = seed.memory.strip() or "No dataset memory is attached to this parent row yet."
    return f"""You are growing a dataset of contest-style mathematics problems one iteration at a time.

The parent row is from iteration {seed.source_iteration}. You are creating
exactly one next-generation problem for iteration {iteration}. The goal is not
to paraphrase the parent. The goal is to create a new solved problem by changing
the mathematical pressure of the parent problem: make the
solver notice a sharper condition, a hidden obstruction, a tight equality case,
a representation switch, or a small conceptual trap. Avoid adding long routine
calculation. Avoid stitching an unrelated second task onto the problem.

Return exactly {cfg.lineage.candidates_per_iteration} candidates in this loose
section format. Do not use JSON.

=== CANDIDATE 1 ===
Transformation:
Problem:
Answer:
Solution:
Why It Is New:
Failure Risk:

Continue with CANDIDATE 2, etc.

Parent problem:
{_trim(current_question, cfg.lineage.max_question_chars)}

Parent answer:
{_trim(current_answer, cfg.lineage.max_answer_chars)}

Dataset memory stored on the parent row:
{_trim(parent_memory, cfg.lineage.max_answer_chars)}

{_memory_block("Memory from earlier rows in this run", global_memory[-cfg.lineage.max_memory_items:])}
"""


def _refiner_prompt(cfg: Config, *, seed: SeedRecord, iteration: int, candidate_text: str) -> str:
    return f"""Refine the candidate math problems below.

For each candidate, preserve the central transformation but remove artificial
complexity, fix any correctness issue you see, and make the solution concise.
Do not invent a separate downstream task. Do not turn it into a longer problem
just to look harder.

Return the same number of blocks in this format, and do not use JSON:

=== REFINED 1 ===
Transformation:
Problem:
Answer:
Solution:
Why It Is New:
Failure Risk:

Parent problem:
{_trim(seed.question, cfg.lineage.max_question_chars)}

Target iteration: {iteration}

Candidates to refine:
{candidate_text}
"""


def _judge_prompt(cfg: Config, *, seed: SeedRecord, current_question: str, candidates: list[Candidate]) -> str:
    blocks = _format_candidate_blocks(candidates)
    return f"""Judge the refined candidate problems against the seed/current problem.

You are not rewarding surface novelty. Reward candidates that are correct,
contest-like, compact, and genuinely transform the mathematical idea. Penalize
near paraphrases, routine parameter changes, stitched-on extra tasks, and
solutions that are mostly computation.

Return one block per candidate. Do not use JSON.

=== JUDGEMENT 1 ===
Correct: yes/no
Novelty: number from 0 to 1
Depth: number from 0 to 1
Seed Alignment: number from 0 to 1
Non-Stitched: number from 0 to 1
Solution Economy: number from 0 to 1
Decision: accept/reject
Failure Kind: accepted/incorrect/near_paraphrase/routine/stitched/weak_quality
Notes:

Parent problem:
{_trim(seed.question, cfg.lineage.max_question_chars)}

Parent lineage problem:
{_trim(current_question, cfg.lineage.max_question_chars)}

Candidates:
{blocks}
"""


def _normalized_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _block_name_aliases(names: tuple[str, ...]) -> list[str]:
    aliases: set[str] = set()
    normalized = {_normalized_label(name) for name in names}
    for name in normalized:
        aliases.add(name)
    if "judgement" in normalized or "judgment" in normalized:
        aliases.update({"judgement", "judgment", "judge", "review", "evaluation", "candidate"})
    if "candidate" in normalized:
        aliases.update({"candidate", "option", "proposal", "problem"})
    if "refined" in normalized:
        aliases.update({"refined", "candidate", "option", "proposal", "problem"})
    return sorted(aliases, key=len, reverse=True)


def _split_numbered_blocks(text: str, names: tuple[str, ...]) -> list[tuple[str, str]]:
    aliases = "|".join(re.escape(alias).replace(r"\ ", r"\s+") for alias in _block_name_aliases(names))
    marker_patterns = [
        rf"(?im)^\s*={{2,}}\s*(?:{aliases})\s*(?:#|no\.?|number)?\s*([A-Za-z0-9_.-]+)\s*={{2,}}\s*$",
        rf"(?im)^\s*#{{1,6}}\s*(?:{aliases})\s*(?:#|no\.?|number)?\s*([A-Za-z0-9_.-]+)\s*:?\s*$",
        rf"(?im)^\s*\*{{0,2}}\s*(?:{aliases})\s*(?:#|no\.?|number)?\s*([A-Za-z0-9_.-]+)\s*\*{{0,2}}\s*:?\s*$",
        rf"(?im)^\s*(?:{aliases})\s*(?:#|no\.?|number)?\s*([A-Za-z0-9_.-]+)\s*[:.)-]?\s*$",
    ]
    matches: list[re.Match[str]] = []
    seen_starts: set[int] = set()
    for pattern in marker_patterns:
        for match in re.finditer(pattern, text or ""):
            if match.start() in seen_starts:
                continue
            seen_starts.add(match.start())
            matches.append(match)
    matches.sort(key=lambda match: match.start())

    # Common loose form:
    # 1. Correct: yes
    # ...
    # 2. Correct: no
    # ...
    if not matches:
        generic = list(re.finditer(r"(?im)^\s*(\d+)\s*[.)]\s+(?=[A-Za-z])", text or ""))
        if len(generic) > 1:
            matches = generic

    blocks: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((match.group(1), text[start:end].strip()))
    return blocks


_FIELD_ALIASES = {
    "transformation": "transformation",
    "mutation": "transformation",
    "change": "transformation",
    "what changed": "transformation",
    "problem": "problem",
    "question": "problem",
    "new problem": "problem",
    "proposed problem": "problem",
    "generated problem": "problem",
    "answer": "answer",
    "final answer": "answer",
    "result": "answer",
    "solution": "solution",
    "proof": "solution",
    "full solution": "solution",
    "why it is new": "why it is new",
    "why new": "why it is new",
    "why this is new": "why it is new",
    "why sharper": "why it is new",
    "reason": "notes",
    "failure risk": "failure risk",
    "risk": "failure risk",
    "risks": "failure risk",
    "correct": "correct",
    "correctness": "correct",
    "mathematical correctness": "correct",
    "novelty": "novelty",
    "novelty score": "novelty",
    "depth": "depth",
    "depth score": "depth",
    "quality": "depth",
    "seed alignment": "seed alignment",
    "alignment": "seed alignment",
    "non stitched": "non stitched",
    "nonstitched": "non stitched",
    "not stitched": "non stitched",
    "solution economy": "solution economy",
    "economy": "solution economy",
    "decision": "decision",
    "verdict": "decision",
    "accepted": "decision",
    "failure kind": "failure kind",
    "failure reason": "failure kind",
    "rejection reason": "failure kind",
    "notes": "notes",
    "judge notes": "notes",
    "explanation": "notes",
}

_META_HEADINGS = {
    "parent problem",
    "parent answer",
    "parent lineage problem",
    "dataset memory",
    "dataset memory stored on the parent row",
    "memory from earlier rows in this run",
    "target iteration",
    "candidates",
    "candidates to refine",
}


def _canonical_field_name(raw: str) -> str | None:
    key = _normalized_label(raw)
    return _FIELD_ALIASES.get(key)


def _is_meta_heading(raw: str) -> bool:
    return _normalized_label(raw) in _META_HEADINGS


def _parse_heading_line(line: str) -> tuple[str | None, str]:
    clean = line.strip()
    clean = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", clean)
    clean = clean.replace("**", "").replace("__", "").strip()
    if not clean:
        return None, ""

    # Heading with delimiter and optional inline value: "Problem: ...",
    # "Novelty - 0.7", "Verdict — accept".
    match = re.match(r"^(.{1,60}?)(?:\s*:|\s+[-–—])\s*(.*)$", clean)
    if match:
        raw_heading = match.group(1).strip()
        if _is_meta_heading(raw_heading):
            return "__META__", ""
        canonical = _canonical_field_name(raw_heading)
        if canonical:
            return canonical, match.group(2).strip()

    # Heading alone on its own line: "Solution"
    raw_heading = clean.rstrip(":").strip()
    if _is_meta_heading(raw_heading):
        return "__META__", ""
    canonical = _canonical_field_name(raw_heading)
    if canonical and len(raw_heading) <= 60:
        return canonical, ""
    return None, ""


def _fields(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current_key: str | None = None
    for line in (block or "").splitlines():
        key, inline_value = _parse_heading_line(line)
        if key == "__META__":
            break
        if key:
            if key in out:
                current_key = None
                continue
            out[key] = inline_value
            current_key = key
            continue
        if current_key is not None:
            out[current_key] = (out[current_key] + "\n" + line).strip()
    return {key: value.strip() for key, value in out.items()}


def _parse_candidates(text: str, names: tuple[str, ...] = ("CANDIDATE", "REFINED")) -> list[Candidate]:
    blocks = _split_numbered_blocks(text, names)
    if not blocks:
        blocks = [("1", text or "")]
    candidates: list[Candidate] = []
    for label, block in blocks:
        fields = _fields(block)
        question = fields.get("problem") or fields.get("question") or ""
        answer = fields.get("answer") or ""
        solution = fields.get("solution") or ""
        if not question.strip() or not answer.strip():
            continue
        candidates.append(
            Candidate(
                label=str(label),
                question=question.strip(),
                answer=answer.strip(),
                solution=solution.strip(),
                transformation=(fields.get("transformation") or "").strip(),
                why_new=(fields.get("why it is new") or fields.get("why new") or "").strip(),
                risk=(fields.get("failure risk") or fields.get("risk") or "").strip(),
                raw_text=block,
            )
        )
    return candidates


def _format_candidate_blocks(candidates: list[Candidate]) -> str:
    parts: list[str] = []
    for i, cand in enumerate(candidates, start=1):
        parts.append(
            f"""=== CANDIDATE {i} ===
Transformation:
{cand.transformation}

Problem:
{cand.question}

Answer:
{cand.answer}

Solution:
{cand.solution}

Why It Is New:
{cand.why_new}

Failure Risk:
{cand.risk}
"""
        )
    return "\n".join(parts)


def _as_float(value: str, default: float = 0.0) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value or "")
    if not match:
        return default
    return max(0.0, min(1.0, float(match.group(0))))


def _as_bool(value: str) -> bool:
    return str(value or "").strip().lower().startswith(("yes", "true", "1", "correct"))


def _apply_judgements(candidates: list[Candidate], text: str) -> None:
    blocks = _split_numbered_blocks(text, ("JUDGEMENT", "JUDGMENT"))
    if not blocks:
        blocks = _split_numbered_blocks(text, ("CANDIDATE",))
    if not blocks:
        blocks = _split_numbered_blocks(text, ("REVIEW", "EVALUATION"))
    if not blocks and candidates:
        fields = _fields(text or "")
        if len(candidates) == 1 and fields:
            blocks = [("1", text or "")]
        else:
            note = "Judge did not return parseable judgement blocks."
            if (text or "").strip():
                note += " Raw judge response: " + _trim(text, 900)
            for cand in candidates:
                cand.correct = False
                cand.decision = "reject"
                cand.failure_kind = "judge_unparseable" if (text or "").strip() else "judge_empty"
                cand.judge_notes = note
            return
    for idx, cand in enumerate(candidates):
        if idx >= len(blocks):
            cand.failure_kind = "judge_missing"
            cand.judge_notes = "No judgement block returned for this candidate."
            if (text or "").strip():
                cand.judge_notes += " Raw judge response: " + _trim(text, 900)
            continue
        _, block = blocks[idx]
        fields = _fields(block)
        cand.correct = _as_bool(fields.get("correct", ""))
        cand.scores = {
            "novelty": _as_float(fields.get("novelty", "")),
            "depth": _as_float(fields.get("depth", "")),
            "seed_alignment": _as_float(fields.get("seed alignment", "")),
            "non_stitched": _as_float(fields.get("non stitched", "")),
            "solution_economy": _as_float(fields.get("solution economy", "")),
        }
        decision = (fields.get("decision") or "").strip().lower()
        if not decision:
            decision = "accept" if cand.correct else "reject"
        cand.decision = decision
        cand.failure_kind = (fields.get("failure kind") or "").strip().lower()
        cand.judge_notes = (fields.get("notes") or "").strip()


def _candidate_passes(cfg: Config, cand: Candidate) -> bool:
    if not cand.correct and cfg.lineage.min_correctness >= 1.0:
        return False
    return (
        cand.scores.get("novelty", 0.0) >= cfg.lineage.min_novelty
        and cand.scores.get("depth", 0.0) >= cfg.lineage.min_depth
        and cand.scores.get("non_stitched", 0.0) >= cfg.lineage.min_non_stitched
        and cand.scores.get("solution_economy", 0.0) >= cfg.lineage.min_solution_economy
        and cand.decision != "reject"
    )


def _candidate_score(cand: Candidate) -> float:
    s = cand.scores
    return (
        0.28 * s.get("depth", 0.0)
        + 0.24 * s.get("novelty", 0.0)
        + 0.18 * s.get("seed_alignment", 0.0)
        + 0.16 * s.get("non_stitched", 0.0)
        + 0.14 * s.get("solution_economy", 0.0)
        + (0.10 if cand.correct else -0.30)
    )


def _select_candidate(cfg: Config, candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    passing = [cand for cand in candidates if _candidate_passes(cfg, cand)]
    if passing:
        return max(passing, key=_candidate_score)
    if cfg.lineage.continue_on_rejected:
        return max(candidates, key=_candidate_score)
    return None


def _seed_row(cfg: Config, seed: SeedRecord, lineage_id: str) -> dict[str, Any]:
    row_id = _stable_id("seed", cfg.input.dataset, cfg.input.split, seed.source_id)
    return {
        "id": row_id,
        "lineage_id": lineage_id,
        "seed_id": seed.source_id,
        "parent_id": "",
        "source_iteration": seed.source_iteration,
        "iteration": seed.source_iteration,
        "role": "source",
        "question": seed.question,
        "answer": seed.answer,
        "solution": "",
        "accepted": True,
        "failure_kind": "",
        "memory": seed.memory,
        "memory_json": json.dumps({"source_memory": seed.memory}, ensure_ascii=False, default=str),
        "scores_json": "{}",
        "artifacts_json": json.dumps({"source_record": seed.raw}, ensure_ascii=False, default=str),
        "created_at": int(time.time()),
    }


def _build_memory(seed: SeedRecord, selected: Candidate, artifacts: dict[str, Any], *, accepted: bool) -> str:
    prior = seed.memory.strip()
    rejected = [
        f"{cand.get('failure_kind') or 'rejected'}: "
        f"{cand.get('transformation') or str(cand.get('question') or '')[:160]}"
        for cand in artifacts.get("candidates", [])
        if not cand.get("correct") or cand.get("decision") == "reject"
    ][:3]
    scores = ", ".join(f"{k}={v:.2f}" for k, v in selected.scores.items())
    parts = []
    if prior:
        parts.append("Previous dataset memory:\n" + prior)
    parts.append(
        "Latest synthesis memory:\n"
        f"- Parent iteration: {seed.source_iteration}\n"
        f"- New status: {'accepted' if accepted else 'rejected'}\n"
        f"- Transformation used: {selected.transformation or 'unspecified'}\n"
        f"- Why it changed the problem: {selected.why_new or 'unspecified'}\n"
        f"- Judge notes: {selected.judge_notes or 'none'}\n"
        f"- Scores: {scores or '{}'}"
    )
    if rejected:
        parts.append("Failed directions to avoid next time:\n- " + "\n- ".join(rejected))
    return "\n\n".join(parts).strip()


def _generated_row(
    cfg: Config,
    *,
    seed: SeedRecord,
    lineage_id: str,
    parent_id: str,
    iteration: int,
    selected: Candidate,
    artifacts: dict[str, Any],
    role: str = "generated",
) -> dict[str, Any]:
    row_id = _stable_id(role, lineage_id, iteration, selected.question, selected.answer)
    accepted = _candidate_passes(cfg, selected)
    memory = _build_memory(seed, selected, artifacts, accepted=accepted)
    memory_json = {
        "parent_memory": seed.memory,
        "status": "accepted" if accepted else "rejected",
        "transformation": selected.transformation,
        "why_new": selected.why_new,
        "risk": selected.risk,
        "judge_notes": selected.judge_notes,
        "scores": selected.scores,
        "candidate_summaries": artifacts.get("candidates", []),
    }
    return {
        "id": row_id,
        "lineage_id": lineage_id,
        "seed_id": seed.source_id,
        "parent_id": parent_id,
        "source_iteration": seed.source_iteration,
        "iteration": iteration,
        "role": role,
        "question": selected.question,
        "answer": selected.answer,
        "solution": selected.solution,
        cfg.input.question_field: selected.question,
        cfg.input.answer_field: selected.answer,
        cfg.input.iteration_field: iteration,
        cfg.input.memory_field: memory,
        "memory": memory,
        "memory_json": json.dumps(memory_json, ensure_ascii=False, default=str),
        "source_question": seed.question,
        "source_answer": seed.answer,
        "source_id": seed.source_id,
        "accepted": accepted,
        "failure_kind": selected.failure_kind,
        "scores_json": json.dumps(selected.scores, ensure_ascii=False, default=str),
        "artifacts_json": json.dumps(artifacts, ensure_ascii=False, default=str),
        "created_at": int(time.time()),
    }


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _safe_dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            out[key] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            out[key] = value
    return out


def _align_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({key for row in records for key in row})
    return [{key: row.get(key) for key in keys} for row in records]


def _same_dataset(a: str | None, b: str | None) -> bool:
    return bool(a and b and a.strip().lower() == b.strip().lower())


def publish_rows(cfg: Config, rows: list[dict[str, Any]], *, dataset_id: str | None = None) -> str:
    repo_id = dataset_id or cfg.output.dataset
    if not repo_id:
        raise ValueError("No output dataset id configured.")

    if _same_dataset(repo_id, cfg.input.dataset) and cfg.output.append_if_same_dataset:
        records = [_safe_dataset_row(row) for row in rows if row.get("role") in {"generated", "rejected"}]
        if not records:
            raise ValueError("No generated or rejected records to append to the input dataset.")
        base = load_dataset(**_dataset_kwargs(cfg))
        base_records = [_safe_dataset_row(dict(row)) for row in base]
        generated: list[dict[str, Any]] = []
        for row in records:
            row = dict(row)
            row[cfg.input.question_field] = row.get("question")
            row[cfg.input.answer_field] = row.get("answer")
            generated.append(row)
        records = base_records + generated
    else:
        records = [_safe_dataset_row(row) for row in rows]

    if not records:
        raise ValueError("No records to publish.")

    ds = Dataset.from_list(_align_records(records))
    ds.push_to_hub(repo_id, split=cfg.output.split, private=cfg.output.private)
    return repo_id


def run_lineage(
    cfg: Config,
    *,
    num_seeds: int | None = None,
    seed: int | None = None,
    push_to_hub: bool | None = None,
    dataset_id: str | None = None,
    event_sink: EventSink | None = None,
) -> LineageResult:
    def emit(kind: str, **payload: Any) -> None:
        if event_sink is None:
            return
        event = {"type": kind, "time": int(time.time()), **payload}
        event_sink(event)

    if num_seeds is not None:
        cfg.input.max_seeds = num_seeds

    seeds, source_iteration, current_iteration = load_seed_records(cfg, num_seeds=num_seeds, seed=seed)
    emit(
        "run_started",
        input_dataset=cfg.input.dataset,
        input_split=cfg.input.split,
        output_dataset=dataset_id or cfg.output.dataset,
        seed_count=len(seeds),
        source_iteration=source_iteration,
        current_iteration=current_iteration,
    )
    rows: list[dict[str, Any]] = []
    global_memory: list[str] = []
    llm = LLMClient(cfg.storage.cache_dir)
    summary: dict[str, Any] = {
        "input_dataset": cfg.input.dataset,
        "input_split": cfg.input.split,
        "output_dataset": dataset_id or cfg.output.dataset,
        "seeds": len(seeds),
        "source_iteration": source_iteration,
        "current_iteration": current_iteration,
        "source_rows": 0,
        "generated_rows": 0,
        "accepted_steps": 0,
        "rejected_steps": 0,
        "errors": [],
        "model_calls": 0,
    }

    for seed_idx, seed_record in enumerate(seeds, start=1):
        lineage_id = _stable_id("lineage", cfg.input.dataset, cfg.input.split, seed_record.source_id)
        seed_row = _seed_row(cfg, seed_record, lineage_id)
        rows.append(seed_row)
        summary["source_rows"] += 1
        emit(
            "seed_started",
            seed_id=seed_record.source_id,
            seed_index=seed_idx,
            seed_count=len(seeds),
            lineage_id=lineage_id,
            row=seed_row,
        )

        parent_id = seed_row["id"]
        current_question = seed_record.question
        current_answer = seed_record.answer
        console.print(f"lineage seed {seed_idx}/{len(seeds)} {seed_record.source_id}")

        if cfg.runtime.dry_run:
            emit("seed_done", seed_id=seed_record.source_id, status="dry_run")
            continue

        iteration = current_iteration
        console.print(f"  target iteration {iteration}: generate")
        emit(
            "stage_started",
            seed_id=seed_record.source_id,
            iteration=iteration,
            stage="generate",
        )
        try:
            gen_prompt = _generation_prompt(
                cfg,
                seed=seed_record,
                current_question=current_question,
                current_answer=current_answer,
                iteration=iteration,
                local_memory=[],
                global_memory=global_memory,
            )
            gen_out = llm.chat(
                model=str(cfg.models.generator),
                system="You synthesize original contest mathematics problems.",
                user=gen_prompt,
                temperature=cfg.lineage.generator_temperature,
                max_tokens=5000,
                use_cache=False,
            )
            raw_generation = gen_out.get("content") or ""
            candidates = _parse_candidates(raw_generation, ("CANDIDATE",))
            if not candidates:
                raise RuntimeError("generator returned no parseable candidates")
            emit(
                "stage_done",
                seed_id=seed_record.source_id,
                iteration=iteration,
                stage="generate",
                candidate_count=len(candidates),
            )

            console.print(f"  target iteration {iteration}: refine {len(candidates)}")
            emit(
                "stage_started",
                seed_id=seed_record.source_id,
                iteration=iteration,
                stage="refine",
                candidate_count=len(candidates),
            )
            ref_prompt = _refiner_prompt(
                cfg,
                seed=seed_record,
                iteration=iteration,
                candidate_text=_format_candidate_blocks(candidates),
            )
            ref_out = llm.chat(
                model=str(cfg.models.refiner),
                system="You refine math problems for correctness, compactness, and conceptual sharpness.",
                user=ref_prompt,
                temperature=cfg.lineage.refiner_temperature,
                max_tokens=5000,
                use_cache=False,
            )
            raw_refinement = ref_out.get("content") or ""
            refined = _parse_candidates(raw_refinement, ("REFINED",))
            if len(refined) == len(candidates):
                candidates = refined
            emit(
                "stage_done",
                seed_id=seed_record.source_id,
                iteration=iteration,
                stage="refine",
                candidate_count=len(candidates),
            )

            console.print(f"  target iteration {iteration}: judge")
            emit(
                "stage_started",
                seed_id=seed_record.source_id,
                iteration=iteration,
                stage="judge",
                candidate_count=len(candidates),
            )
            judge_prompt = _judge_prompt(
                cfg,
                seed=seed_record,
                current_question=current_question,
                candidates=candidates,
            )
            judge_out = llm.chat(
                model=str(cfg.models.judge),
                system="You are a strict contest math problem judge.",
                user=judge_prompt,
                temperature=cfg.lineage.judge_temperature,
                max_tokens=4500,
                use_cache=False,
            )
            raw_judgement = judge_out.get("content") or ""
            _apply_judgements(candidates, raw_judgement)
            selected = _select_candidate(cfg, candidates)
            emit(
                "stage_done",
                seed_id=seed_record.source_id,
                iteration=iteration,
                stage="judge",
                candidates=[cand.to_artifact() for cand in candidates],
                raw_response=raw_judgement,
            )

            artifacts = {
                "source_iteration": seed_record.source_iteration,
                "iteration": iteration,
                "raw_generation_response": raw_generation,
                "raw_refinement_response": raw_refinement,
                "raw_judgement_response": raw_judgement,
                "candidates": [cand.to_artifact() for cand in candidates],
                "parent_memory": seed_record.memory,
                "global_memory_used": list(global_memory[-cfg.lineage.max_memory_items:]),
            }

            if selected is None:
                summary["rejected_steps"] += 1
                best = max(candidates, key=_candidate_score)
                rows.append(
                    rejected_row := _generated_row(
                        cfg,
                        seed=seed_record,
                        lineage_id=lineage_id,
                        parent_id=parent_id,
                        iteration=iteration,
                        selected=best,
                        artifacts=artifacts,
                        role="rejected",
                    )
                )
                emit(
                    "iteration_rejected",
                    seed_id=seed_record.source_id,
                    iteration=iteration,
                    row=rejected_row,
                    artifacts=artifacts,
                )
                lesson = (
                    f"Rejected at parent {seed_record.source_id} for iteration {iteration}: "
                    f"{best.failure_kind or 'weak_candidate'}; {best.judge_notes[:240]}"
                )
                global_memory.append(lesson)
                console.print(f"  rejected: {best.failure_kind or 'no candidate passed'}")
                emit("seed_done", seed_id=seed_record.source_id, status="rejected")
                continue

            row = _generated_row(
                cfg,
                seed=seed_record,
                lineage_id=lineage_id,
                parent_id=parent_id,
                iteration=iteration,
                selected=selected,
                artifacts=artifacts,
            )
            rows.append(row)
            summary["generated_rows"] += 1
            summary["accepted_steps"] += int(row["accepted"])

            lesson = (
                f"Accepted parent {seed_record.source_id} for iteration {iteration}: "
                f"{selected.transformation[:220]} "
                f"(depth={selected.scores.get('depth', 0):.2f}, novelty={selected.scores.get('novelty', 0):.2f})."
            )
            global_memory.append(lesson)
            console.print(f"  accepted {row['id']}")
            emit(
                "iteration_accepted",
                seed_id=seed_record.source_id,
                iteration=iteration,
                row=row,
                artifacts=artifacts,
            )
            emit("seed_done", seed_id=seed_record.source_id, status="done")
        except Exception as exc:
            message = f"{seed_record.source_id} iteration {iteration}: {exc}"
            summary["errors"].append(message)
            console.print(f"  error: {message}")
            emit(
                "seed_error",
                seed_id=seed_record.source_id,
                iteration=iteration,
                error=str(exc),
            )


    summary["model_calls"] = llm.call_count
    summary["total_rows"] = len(rows)
    summary["global_memory"] = global_memory[-cfg.lineage.max_memory_items:]

    _write_jsonl(cfg.output.local_path, rows)
    _write_json(cfg.output.summary_path, summary)

    hub_pushed = False
    hub_dataset = None
    should_push = cfg.output.push_to_hub if push_to_hub is None else push_to_hub
    if should_push:
        emit("upload_started", dataset=dataset_id or cfg.output.dataset)
        hub_dataset = publish_rows(cfg, rows, dataset_id=dataset_id)
        hub_pushed = True
        summary["hub_dataset"] = hub_dataset
        summary["hub_pushed"] = True
        _write_json(cfg.output.summary_path, summary)
        emit("upload_done", dataset=hub_dataset)

    emit("run_done", summary=summary)

    return LineageResult(
        rows=rows,
        summary=summary,
        local_path=cfg.output.local_path,
        summary_path=cfg.output.summary_path,
        hub_dataset=hub_dataset,
        hub_pushed=hub_pushed,
        current_iteration=current_iteration,
    )
