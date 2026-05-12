"""MinHash signatures over canonicalized SymPy-extractable expressions in question+answer."""
from __future__ import annotations

import re

import sympy
from datasketch import MinHash
from sympy.parsing.sympy_parser import parse_expr

NUM_PERM = 128

# Greedy regex for math-ish expressions inside text.
_EXPR_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\s*[\+\-\*/\^=<>][^\s,;]+|\$[^$]+\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]|[0-9]+\s*[\+\-\*/\^=<>][^\s,;]+")


def _extract_expressions(text: str) -> list[str]:
    found = _EXPR_RE.findall(text or "")
    out = []
    for raw in found:
        s = raw.strip("$\\()[]")
        try:
            e = sympy.simplify(parse_expr(s, evaluate=True))
            out.append(str(e))
        except Exception:
            out.append(re.sub(r"\s+", "", s))
    return out


def minhash_signature(question: str, answer: str) -> list[int]:
    expressions = _extract_expressions(f"{question} {answer}")
    mh = MinHash(num_perm=NUM_PERM)
    if not expressions:
        # fall back to character n-grams of the question to avoid empty signatures
        text = (question or "")
        for i in range(max(0, len(text) - 4)):
            mh.update(text[i:i + 5].encode())
    else:
        for e in expressions:
            mh.update(e.encode())
    return list(int(x) for x in mh.hashvalues)


def minhash_jaccard(a: list[int], b: list[int]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    eq = sum(1 for x, y in zip(a, b) if x == y)
    return eq / len(a)
