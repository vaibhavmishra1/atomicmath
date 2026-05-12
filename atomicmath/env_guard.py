"""Fail fast when config references providers whose API keys are missing from the environment."""
from __future__ import annotations

import os

from .config import Config


def _model_strings(cfg: Config) -> list[str]:
    out = [
        cfg.models.embedder,
        cfg.models.extractor,
        cfg.models.judge,
        *cfg.models.generators,
        *cfg.models.verifiers,
        cfg.gate.answer_equivalence_model,
        cfg.quality.model,
    ]
    return [m for m in out if isinstance(m, str) and m and "/" in m]


def check_config_api_env(cfg: Config) -> None:
    """Raise RuntimeError with a clear checklist if required env vars are absent."""
    names = _model_strings(cfg)
    missing: list[str] = []

    def _any(prefix: str) -> bool:
        return any(n.startswith(prefix) for n in names)

    if _any("openai/") and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY — required for openai/… models (embedder, GPT, etc.)")
    if _any("anthropic/") and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY — required for anthropic/… models")
    if _any("gemini/") and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        missing.append("GEMINI_API_KEY or GOOGLE_API_KEY — required for gemini/… models")
    if _any("vertex_ai/") or _any("vertex_ai_beta/"):
        if not (os.environ.get("VERTEXAI_PROJECT") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")):
            missing.append("Vertex AI — set VERTEXAI_* / GOOGLE_APPLICATION_CREDENTIALS per your setup")

    if missing:
        raise RuntimeError(
            "Missing API credentials for your config models. Export these before `atomicmath run`:\n\n"
            + "\n".join(f"  • {m}" for m in missing)
            + "\n\nSee https://docs.litellm.ai/docs/set_keys and your provider’s dashboard."
        )
