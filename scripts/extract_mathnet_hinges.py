#!/usr/bin/env python3
"""Rough hinge extraction probe for MathNet.

Example:
    python3 scripts/extract_mathnet_hinges.py --dry-run --limit 50
    python3 scripts/extract_mathnet_hinges.py --limit 50 --model openai/gpt-5-mini
    python3 scripts/extract_mathnet_hinges.py --domain Algebra --subdomain "Prealgebra / Basic Algebra"

The script intentionally stays outside the main atomicmath pipeline. It samples
one MathNet domain/subdomain slice, then asks an OpenAI-compatible LiteLLM model
to extract the nontrivial mathematical hinge from each question-solution pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATASET = "ShadenA/MathNet"
DEFAULT_CONFIG = "United_States"
DEFAULT_SPLIT = "train"
DEFAULT_MODEL = "openai/gpt-5-mini"

QUESTION_FIELD = "problem_markdown"
SOLUTION_FIELD = "solutions_markdown"
ANSWER_FIELD = "final_answer"
TOPIC_FIELD = "topics_flat"


@dataclass
class ProblemRow:
    row_id: str
    question: str
    answer: str
    solution: str
    topics_flat: list[str]
    domain: str
    subdomain: str
    matched_topic: str
    problem_type: str
    country: str


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _cell_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _cell_list(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _solution_text(row: dict[str, Any]) -> str:
    parts = _cell_list(row, SOLUTION_FIELD)
    return "\n\n".join(parts)


def _topic_groups(topics: list[str]) -> list[tuple[str, str, str]]:
    """Return (domain, subdomain, full_topic) candidates from MathNet topics_flat."""
    groups: list[tuple[str, str, str]] = []
    for topic in topics:
        pieces = [p.strip() for p in topic.split(">") if p.strip()]
        if not pieces:
            continue
        domain = pieces[0]
        subdomain = pieces[1] if len(pieces) >= 2 else ""
        groups.append((domain, subdomain, topic))
    return groups


def _stable_id(question: str, answer: str) -> str:
    return hashlib.sha256(f"{question}\x1e{answer}".encode()).hexdigest()[:16]


def _looks_english(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return ascii_chars / max(1, len(text)) >= 0.80


def _iter_dataset(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.config_name:
        return load_dataset(args.dataset, args.config_name, split=args.split, streaming=True)
    return load_dataset(args.dataset, split=args.split, streaming=True)


def _passes_basic_filters(row: dict[str, Any], args: argparse.Namespace) -> bool:
    question = _cell_str(row, QUESTION_FIELD)
    solution = _solution_text(row)
    if not question or not solution:
        return False
    if args.english_only and not _looks_english(question):
        return False
    if args.skip_images and row.get("images"):
        return False
    if args.problem_type:
        allowed = {_norm_space(x).lower() for x in args.problem_type}
        got = _norm_space(_cell_str(row, "problem_type")).lower()
        if got and got not in allowed:
            return False
    return True


def _choose_group(args: argparse.Namespace) -> tuple[str, str]:
    if args.domain is not None and args.subdomain is not None:
        return args.domain, args.subdomain

    counts: Counter[tuple[str, str]] = Counter()
    scanned = 0
    kept = 0
    for row in _iter_dataset(args):
        scanned += 1
        if args.scan_limit and scanned > args.scan_limit:
            break
        if not _passes_basic_filters(row, args):
            continue
        topics = _cell_list(row, TOPIC_FIELD)
        row_groups = _topic_groups(topics)
        if args.domain is not None:
            row_groups = [g for g in row_groups if g[0] == args.domain]
        if args.subdomain is not None:
            row_groups = [g for g in row_groups if g[1] == args.subdomain]
        for domain, subdomain, _ in row_groups:
            counts[(domain, subdomain)] += 1
        kept += 1

    if not counts:
        raise RuntimeError(
            f"no usable topic groups found after scanning {scanned} rows "
            f"({kept} passed basic filters)"
        )
    (domain, subdomain), count = counts.most_common(1)[0]
    print(f"selected group: domain={domain!r}, subdomain={subdomain!r}, count_in_scan={count}")
    return domain, subdomain


def _collect_rows(args: argparse.Namespace, domain: str, subdomain: str) -> list[ProblemRow]:
    out: list[ProblemRow] = []
    scanned = 0
    for row in _iter_dataset(args):
        scanned += 1
        if args.scan_limit and scanned > args.scan_limit:
            break
        if not _passes_basic_filters(row, args):
            continue

        topics = _cell_list(row, TOPIC_FIELD)
        match = None
        for d, s, full_topic in _topic_groups(topics):
            if d == domain and s == subdomain:
                match = (d, s, full_topic)
                break
        if match is None:
            continue

        question = _cell_str(row, QUESTION_FIELD)
        answer = _cell_str(row, ANSWER_FIELD)
        solution = _solution_text(row)
        row_id = _cell_str(row, "id") or _stable_id(question, answer)
        out.append(
            ProblemRow(
                row_id=row_id,
                question=question,
                answer=answer,
                solution=solution,
                topics_flat=topics,
                domain=match[0],
                subdomain=match[1],
                matched_topic=match[2],
                problem_type=_cell_str(row, "problem_type"),
                country=_cell_str(row, "country"),
            )
        )
        if len(out) >= args.limit:
            break

    if len(out) < args.limit:
        print(f"warning: collected only {len(out)} rows for {domain!r} > {subdomain!r}", file=sys.stderr)
    return out


HINGE_SYSTEM = """You are a mathematical problem analyst.

Extract the actual mathematical hinge of a problem: the most nontrivial move a
student must discover for the solution to work. Do not return broad topics like
"algebra" or "number theory" unless they are part of a precise technique.

Focus on:
- the major concept being tested
- the hidden or non-obvious student move
- minimal conditions that make the move valid
- secondary hinges that could be recombined later
- how the hinge could be pushed closer to the edge of solvability

Return compact structured text. JSON is optional; clarity is more important than
machine-strict formatting for this prototype."""


def _hinge_prompt(row: ProblemRow, max_question_chars: int, max_solution_chars: int) -> str:
    question = row.question[:max_question_chars]
    solution = row.solution[:max_solution_chars]
    return f"""DOMAIN: {row.domain}
SUBDOMAIN: {row.subdomain}
MATHNET_TOPICS: {row.topics_flat}
PROBLEM_TYPE: {row.problem_type}

QUESTION:
{question}

FINAL_ANSWER:
{row.answer or "(not provided separately)"}

REFERENCE_SOLUTION:
{solution}

Return a concise hinge note in this loose format:

HINGE_LABEL: short_snake_case_label
MAJOR_CONCEPT: specific concept being tested, not just the broad topic
PRIMARY_HINGE: one precise sentence describing the unlocking move
WHY_NONTRIVIAL: why a student may miss this move
MINIMAL_CONDITIONS:
- conditions required for this hinge to be valid
SOLUTION_SKELETON:
- 3-7 minimal reasoning steps
SECONDARY_HINGES:
- label: how it supports the primary hinge
CAN_PUSH:
- ways to make this hinge sharper without clutter
DO_NOT_CROSS:
- changes that would make the problem ambiguous, routine, or unsolvable
ANTI_PATTERNS:
- boring or invalid mutations to avoid
DIFFICULTY_SIGNALS:
- features that control difficulty here
CONFIDENCE: low/medium/high"""


def _load_litellm():
    try:
        import litellm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LiteLLM is not installed in this Python environment. Run `python -m pip install litellm`, "
            "then retry with the same `python` executable."
        ) from exc
    litellm.drop_params = True
    return litellm


def _balanced_json_from_text(content: str) -> Any | None:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if lines and lines[-1].startswith("```") else "\n".join(lines[1:])
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == opener:
                depth += 1
            elif text[end] == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : end + 1])
    return None


def _cache_path(cache_dir: str, model: str, system: str, user: str, kwargs: dict[str, Any]) -> Path:
    payload = {"model": model, "system": system, "user": user, "kwargs": kwargs}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return Path(cache_dir) / "hinge_llm" / key[:2] / f"{key}.json"


def _message_value(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return str(value).strip()


def _extract_response_text(resp: Any) -> tuple[str, dict[str, Any]]:
    choice = resp.choices[0]
    message = getattr(choice, "message", None)
    fields: list[Any] = []
    if message is not None:
        fields.extend(
            [
                _message_value(message, "content"),
                _message_value(message, "reasoning_content"),
                _message_value(message, "text"),
            ]
        )
    fields.append(getattr(choice, "text", None))
    for field in fields:
        text = _stringify_content(field)
        if text:
            return text, {}
    debug = {
        "finish_reason": getattr(choice, "finish_reason", None),
        "message": repr(message)[:1200],
        "choice": repr(choice)[:1200],
        "usage": repr(getattr(resp, "usage", None))[:500],
    }
    return "", debug


def _completion_kwargs(model: str, max_tokens: int, variant: int) -> dict[str, Any]:
    lower = model.lower()
    is_reasoning_family = "gpt-5" in lower or "/o" in lower or lower.startswith("o")
    if variant == 0 and is_reasoning_family:
        return {"max_completion_tokens": max_tokens}
    if variant == 1:
        return {"max_tokens": max_tokens}
    if variant == 2:
        return {}
    return {"temperature": 0.0, "max_tokens": max_tokens}


def _extract_hinge_text(litellm: Any, args: argparse.Namespace, prompt: str) -> str:
    """Call the model in loose text mode and store the raw hinge note.

    This intentionally bypasses the repo LLMClient because we want to inspect
    empty-content failures while prototyping with different OpenAI/LiteLLM model
    families.
    """
    messages = [
        {"role": "system", "content": HINGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    last_debug: dict[str, Any] = {}
    for variant in range(3):
        kwargs = _completion_kwargs(args.model, args.max_tokens, variant)
        cache_file = _cache_path(args.cache_dir, args.model, HINGE_SYSTEM, prompt, kwargs)
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            content = str(cached.get("content", "")).strip()
            if content:
                return content
        resp = litellm.completion(model=args.model, messages=messages, **kwargs)
        content, debug = _extract_response_text(resp)
        if content:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"content": content, "kwargs": kwargs, "_t": time.time()}, ensure_ascii=False),
                encoding="utf-8",
            )
            return content
        last_debug = {"kwargs": kwargs, **debug}
    raise ValueError(f"empty model response after loose completion calls; debug={last_debug}")


def _extract_hinges(args: argparse.Namespace, rows: list[ProblemRow]) -> list[dict[str, Any]]:
    litellm = _load_litellm()
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] extracting hinge for row_id={row.row_id}")
        prompt = _hinge_prompt(row, args.max_question_chars, args.max_solution_chars)
        try:
            hinge_text = _extract_hinge_text(litellm, args, prompt)
            hinge = _balanced_json_from_text(hinge_text)
            error = None
        except Exception as exc:
            hinge_text = ""
            hinge = None
            error = str(exc)
            print(f"  error: {error}", file=sys.stderr)
        records.append(
            {
                "row": asdict(row),
                "hinge": hinge,
                "hinge_text": hinge_text,
                "error": error,
                "model": args.model,
                "extracted_at": time.time(),
            }
        )
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_summary(path: Path, records: list[dict[str, Any]], domain: str, subdomain: str) -> None:
    labels: Counter[str] = Counter()
    secondary: Counter[str] = Counter()
    for rec in records:
        hinge = rec.get("hinge")
        hinge_text = str(rec.get("hinge_text", "") or "")
        label = ""
        if isinstance(hinge, dict):
            label = str(hinge.get("hinge_label", "")).strip()
        if not label:
            m = re.search(r"(?im)^HINGE_LABEL:\s*([a-zA-Z0-9_ -]+)", hinge_text)
            label = _norm_space(m.group(1)).replace(" ", "_").lower() if m else ""
        if label:
            labels[label] += 1
        if isinstance(hinge, dict):
            for item in hinge.get("secondary_hinges", []) or []:
                if isinstance(item, dict):
                    sec = str(item.get("label", "")).strip()
                    if sec:
                        secondary[sec] += 1

    summary = {
        "domain": domain,
        "subdomain": subdomain,
        "n_records": len(records),
        "n_errors": sum(1 for rec in records if rec.get("error")),
        "primary_hinge_labels": labels.most_common(),
        "secondary_hinge_labels": secondary.most_common(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract loose hinge notes from a MathNet domain/subdomain slice.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--scan-limit", type=int, default=5000)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--subdomain", default=None)
    parser.add_argument("--problem-type", action="append", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--out", default="./out/hinges/mathnet_hinges.jsonl")
    parser.add_argument("--summary-out", default="./out/hinges/mathnet_hinges_summary.json")
    parser.add_argument("--dry-run", action="store_true", help="Select rows and write metadata without LLM calls.")
    parser.add_argument("--english-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-question-chars", type=int, default=6000)
    parser.add_argument("--max-solution-chars", type=int, default=9000)
    parser.add_argument("--max-tokens", type=int, default=1400)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if (args.domain is None) != (args.subdomain is None):
        print("note: only one of --domain/--subdomain was supplied; auto-selection will complete the pair")
    return args


def main() -> int:
    args = parse_args()
    domain, subdomain = _choose_group(args)
    rows = _collect_rows(args, domain, subdomain)
    if not rows:
        raise RuntimeError(f"no rows collected for {domain!r} > {subdomain!r}")

    print(f"collected {len(rows)} rows from {domain!r} > {subdomain!r}")
    print("sample row:")
    print(json.dumps(asdict(rows[0]) | {"solution": rows[0].solution[:500] + "..."}, indent=2, ensure_ascii=False))

    if args.dry_run:
        records = [
            {
                "row": asdict(row),
                "hinge": None,
                "hinge_text": "",
                "error": None,
                "dry_run": True,
                "model": args.model,
                "extracted_at": None,
            }
            for row in rows
        ]
    else:
        records = _extract_hinges(args, rows)

    out_path = Path(args.out)
    summary_path = Path(args.summary_out)
    _write_jsonl(out_path, records)
    _write_summary(summary_path, records, domain, subdomain)
    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
