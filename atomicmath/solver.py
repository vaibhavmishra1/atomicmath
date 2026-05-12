"""LLM single-shot solver used by correctness verification."""
from __future__ import annotations

from .llm import LLMClient

_SOLVE_SYSTEM = """You are a math solver. Solve the given problem and return ONLY the final answer in the simplest canonical form.

Output JSON: {"answer": "<the final answer>"}.

Do not include explanation, working, or commentary."""


def solve_for_answer(llm: LLMClient, model: str, question: str, *, temperature: float = 0.7) -> str:
    user = f"PROBLEM:\n{question}\n\nReturn JSON: {{\"answer\": \"...\"}}."
    out = llm.chat_json(model=model, system=_SOLVE_SYSTEM, user=user, temperature=temperature)
    if isinstance(out, dict) and "answer" in out:
        return str(out["answer"])
    return str(out)
