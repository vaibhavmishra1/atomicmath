"""Answer canonicalization. math-verify → SymPy → numeric → string fallback."""
from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import parse_expr

# math-verify (HuggingFace) handles LaTeX, sets, intervals, equations, units.
# We try it first; if it returns False we fall through to our existing logic
# in case the input format isn't math-verify-friendly.
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify  # type: ignore
    _MV_AVAILABLE = True
except Exception:
    _MV_AVAILABLE = False


def _strip_text_braces(s: str) -> str:
    """Strip \\text{X} → X (math-verify doesn't extract from \\text{} but we know what
    the user means)."""
    return re.sub(r"\\text\{([^}]+)\}", r"\1", s or "")


def _mv_equal(a: str, b: str) -> bool | None:
    """Try math-verify; return True/False if it succeeds, None if it errors."""
    if not _MV_AVAILABLE:
        return None
    try:
        ga = _mv_parse(_strip_text_braces(a))
        gb = _mv_parse(_strip_text_braces(b))
        # math-verify's verify() is intentionally asymmetric; try both orders.
        return bool(_mv_verify(ga, gb) or _mv_verify(gb, ga))
    except Exception:
        return None


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).rstrip(".,;")


def _try_sympy(s: str) -> sympy.Expr | None:
    try:
        return sympy.simplify(parse_expr(s, evaluate=True))
    except Exception:
        return None


def _try_numeric(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except Exception:
        return None


def _normalize_list_string(s: str) -> str:
    # "1, 2, 3" / "{1,2,3}" / "[1, 2, 3]" → sorted comma-joined
    cleaned = re.sub(r"[\[\]{}()]", "", s)
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) <= 1:
        return _strip(s)
    return ", ".join(sorted(parts))


_BOXED_TOKEN = r"\boxed{"


def _boxed_inners(s: str) -> list[str]:
    """Inner bodies of \\\\boxed{...} with correct brace nesting (e.g. MATH solutions)."""
    if not s:
        return []
    frags: list[str] = []
    i = 0
    n = len(s)
    tok = _BOXED_TOKEN
    lt = len(tok)
    while True:
        j = s.find(tok, i)
        if j < 0:
            break
        start = j + lt
        depth = 1
        k = start
        found = False
        while k < n:
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
                if depth == 0:
                    frags.append(s[start:k])
                    i = k + 1
                    found = True
                    break
            k += 1
        if not found:
            i = j + lt
    return frags


def _answer_variants(s: str) -> list[str]:
    """Strings to treat as candidate answers (full text plus \\\\boxed{...} inners, \\\\text stripped)."""
    if s is None:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        t = x.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)

    add(s)
    for inner in _boxed_inners(s):
        add(inner)
        add(_strip_text_braces(inner))
    return out


def _canonical_equal_core(a: str, b: str) -> bool:
    """Pairwise equality without unwrapping \\\\boxed (see canonical_equal)."""
    mv = _mv_equal(a, b)
    if mv is True:
        return True
    if _strip(a) == _strip(b):
        return True
    sa, sb = _try_sympy(a), _try_sympy(b)
    if sa is not None and sb is not None:
        try:
            if sympy.simplify(sa - sb) == 0:
                return True
        except Exception:
            pass
    na, nb = _try_numeric(a), _try_numeric(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= 1e-9 * max(1.0, abs(na))
    return _normalize_list_string(a).lower() == _normalize_list_string(b).lower()


def canonical_equal(a: str, b: str) -> bool:
    """Return True iff a and b represent the same mathematical answer.

    Also compares across MATH-style ``\\\\boxed{...}`` wrappers: a short model answer like
    ``Evelyn`` is matched against a long gold string whose final answer lives in ``\\\\boxed{...}``.

    Per pair, order of attempts:
      0) math-verify (LaTeX-aware: handles \\frac, \\sqrt, sets, intervals, units, ...)
      1) Quick string-strip equality
      2) SymPy parse + simplify(a-b) == 0
      3) Numeric within tolerance
      4) String compare on whitespace/punct-stripped, list-sorted form
    """
    if a is None or b is None:
        return False
    for va in _answer_variants(a):
        for vb in _answer_variants(b):
            if _canonical_equal_core(va, vb):
                return True
    return False


def canonical_form(s: str) -> str:
    """Return a stable canonical-form string for a single answer."""
    if s is None:
        return ""
    sa = _try_sympy(s)
    if sa is not None:
        return str(sa)
    na = _try_numeric(s)
    if na is not None:
        return repr(na)
    return _normalize_list_string(s).lower()
