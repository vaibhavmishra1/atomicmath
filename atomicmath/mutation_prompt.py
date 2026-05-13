"""Prompt assembly for single-question transformation synthesis."""
from __future__ import annotations

import json
import re
from typing import Any

from .config import Config
from .db import Store


TRANSFORMATION_EXAMPLES = """These examples are not categories to choose from. They teach what a good transformation means.

WEAK EXAMPLE: number swap
Cow/barn version:
Original: A cow tied to a 10-foot rope grazes near a barn.
Weak transformation: Change the rope to 12 feet and ask the same area question.
Why weak: The same problem is being re-skinned.
Math analogue: Changing constants, signs, or exponents while preserving the exact solution path is usually too close.

GOOD EXAMPLE: boundary stress
Cow/barn example:
Original: A cow tied to the middle of a barn wall grazes a clean semicircle.
Transformation: Tie the cow exactly at a corner, where the accessible region changes because the boundary matters.
Math interpretation:
Move a condition to a boundary where an equality case, degeneracy, repeated root, tangent case, parity threshold, or extremal condition becomes decisive.
Why good: The original idea survives, but the point where the solver gets stuck changes.

GOOD EXAMPLE: representation shift
Cow/barn example:
Original: A cow's grazing area is described directly by a rope length.
Transformation: Describe the same constraint through a gate path or fence shadow, so the solver must translate the setup before using the same idea.
Math interpretation:
Keep the same core logic but express it through another representation: polynomial roots instead of inequalities, vectors instead of coordinates, geometry instead of algebra, recurrence instead of closed form, or graph condition instead of number condition.
Why good: The solver must discover the old structure instead of seeing it immediately.

GOOD EXAMPLE: reverse the condition
Cow/barn example:
Original: The rope length is fixed and the grazing area is computed.
Transformation: The grazing area is fixed and the rope length or anchor position must be determined.
Math interpretation:
Invert the role of known and unknown quantities while preserving the same reasoning engine. For example, instead of proving a maximum, ask for the condition under which equality is possible.
Why good: The hidden condition becomes necessary rather than decorative.

GOOD EXAMPLE: move the trap
Cow/barn example:
Original: The tempting mistake is to use the full circular grazing area.
Transformation: Make the tempting mistake a different one: the cow can reach around one corner but not another.
Math interpretation:
Preserve the kind of student mistake, but change where the trap appears. The new problem should punish the same false shortcut without copying the old surface form.
Why good: The same logical danger appears in a new place.

WEAK EXAMPLE: stitched extension
Cow/barn example:
Original: A cow tied to a barn grazes an area.
Weak transformation: First solve the original grazing area, then add a separate question about painting the barn.
Why weak: The second part begins after the main hinge is already over.
Math analogue: Do not add a downstream theorem, extra final task, or long verification after the original idea is solved.
"""


def balanced_json_from_text(content: str) -> Any | None:
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
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            ch = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : end + 1])
                    except Exception:
                        break
    return None


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _format_hinges(hinges: list[Any]) -> str:
    if not hinges:
        return "(No hinge notes available yet.)"
    parts = []
    for i, h in enumerate(hinges, start=1):
        if isinstance(h, dict):
            label = h.get("label") or h.get("id") or f"hinge_{i}"
            text = h.get("hinge_text") or h.get("text") or ""
        else:
            label = h["label"] or h["id"] or f"hinge_{i}"
            text = h["hinge_text"]
        parts.append(f"HINGE {i} ({label}):\n{text}")
    return "\n\n".join(parts)


def _format_memories(rows: list[Any], empty: str) -> str:
    if not rows:
        return empty
    parts = []
    for i, row in enumerate(rows, start=1):
        get = row.get if isinstance(row, dict) else row.__getitem__
        kind = get("kind") or "memory"
        mutation = get("mutation_used") or "unknown mutation"
        failure = get("failure_kind") or "none"
        topic = get("topic_norm") or "global"
        weight = get("weight") or 0
        lesson = get("lesson") or ""
        parts.append(
            f"MEMORY {i} (kind={kind}; mutation={mutation}; failure={failure}; topic={topic}; weight={weight}):\n{lesson}"
        )
    return "\n\n".join(parts)


def _load_memory_blocks(
    cfg: Config,
    store: Store,
    seed: Any,
    *,
    success_memories: list[Any] | None = None,
    failure_memories: list[Any] | None = None,
) -> tuple[list[Any], list[Any]]:
    if not cfg.mutation.global_memory_enabled:
        return [], []
    topic = (seed["topic_norm"] or seed["topic_raw"] or "") if cfg.mutation.global_memory_prioritize_topic else None
    if success_memories is None:
        success_memories = (
            store.list_mutation_experiences(
                kind="success",
                topic_norm=topic,
                limit=cfg.mutation.global_success_memory_limit,
                prioritize_topic=cfg.mutation.global_memory_prioritize_topic,
            )
            if cfg.mutation.global_success_memory_limit > 0
            else []
        )
    if failure_memories is None:
        failure_memories = (
            store.list_mutation_experiences(
                kind="failure",
                topic_norm=topic,
                limit=cfg.mutation.global_failure_memory_limit,
                prioritize_topic=cfg.mutation.global_memory_prioritize_topic,
            )
            if cfg.mutation.global_failure_memory_limit > 0
            else []
        )
    return success_memories, failure_memories


def build_plan_generate_prompt(
    cfg: Config,
    store: Store,
    seed: Any,
    hinges: list[Any],
    *,
    success_memories: list[Any] | None = None,
    failure_memories: list[Any] | None = None,
) -> str:
    success_memories, failure_memories = _load_memory_blocks(
        cfg,
        store,
        seed,
        success_memories=success_memories,
        failure_memories=failure_memories,
    )
    question = compact_text(seed["question"], 7000)
    solution = compact_text(seed["solution_text"], 9000)
    answer = compact_text(seed["answer"], 1000)
    return f"""You are transforming ONE solved contest-style math problem into a new problem.

This is a single API call. You must do both jobs:
1. Think like a transformation planner: diagnose the bottleneck, reject weak ideas, choose a strong direction.
2. Think like a problem writer: produce one final self-contained problem with answer and short solution.

You are not choosing from fixed mutation types. You may transform the problem however you want, but the new problem must preserve one important mathematical bottleneck from the seed.

QUESTION:
{question}

FINAL ANSWER:
{answer or "(not provided separately)"}

SOLUTION:
{solution}

HINGES:
{_format_hinges(hinges)}

TRANSFORMATION EXAMPLES:
{TRANSFORMATION_EXAMPLES}

GLOBAL SUCCESS MEMORY:
{_format_memories(success_memories, "(No global success memory stored yet.)")}

GLOBAL FAILURE MEMORY:
{_format_memories(failure_memories, "(No global failure memory stored yet.)")}

TASK:
Generate one new math problem by first choosing a nontrivial transformation and then writing the final problem.

Rules:
- Treat global memory as reusable guidance, not as constraints to copy.
- Reject boring transformations before choosing the final direction.
- Do not merely change numbers, signs, exponent size, names, or setting.
- Do not use a wrapper context unless it changes how the hinge is discovered.
- Do not stitch a second task after the original hinge is solved.
- Prefer transformations that make the hinge appear in a less obvious form.
- Prefer one clean conceptual delta over many small edits.
- The new problem must have one final task.
- The preserved hinge must be necessary in the short solution.
- The solution should stay short; difficulty should come from logic, not bookkeeping.

Return strict JSON with exactly these keys:
{{
  "core_bottleneck": "...",
  "discarded_transformations": [
    {{
      "idea": "...",
      "why_rejected": "too close / routine / stitched / long calculation / loses hinge"
    }}
  ],
  "candidate_transformations": [
    {{
      "idea": "...",
      "conceptual_delta": "...",
      "hinge_preserved": "...",
      "why_not_trivial": "...",
      "risk": "...",
      "ambition_score": 0.0
    }}
  ],
  "chosen_transformation": "...",
  "conceptual_delta": "...",
  "hinge_preserved": "...",
  "why_this_should_be_nontrivial": "...",
  "new_question": "...",
  "answer": "...",
  "short_solution": "...",
  "mutation_used": "short free-form name for the chosen transformation",
  "what_got_mutated": "...",
  "reason_for_mutation": "...",
  "primary_hinge_preserved": "...",
  "why_problem_is_sharper": "...",
  "why_not_stitched": "...",
  "why_not_a_direct_sibling": "...",
  "risk_notes": "..."
}}"""
