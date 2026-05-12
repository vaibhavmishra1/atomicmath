"""LLM client with content-hash caching. Wraps litellm for multi-provider access."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import litellm
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# Some recent models (gpt-5 family, o-series) reject `temperature` and
# `max_tokens` and require `max_completion_tokens` / `reasoning_effort`.
# Letting litellm silently drop unsupported params is the cleanest path —
# we keep our own temperature semantics but don't break on stricter providers.
litellm.drop_params = True


def _non_retryable_provider_error(exc: BaseException) -> bool:
    """Auth / missing-key style failures should not spin on exponential backoff."""
    mod = getattr(exc.__class__, "__module__", "") or ""
    name = exc.__class__.__name__
    if "litellm" in mod and name == "AuthenticationError":
        return True
    s = str(exc).lower()
    if "missing" in s and "api key" in s:
        return True
    if "authentication" in s or "unauthorized" in s or "invalid api key" in s or "incorrect api key" in s:
        return True
    if "api key" in s and ("invalid" in s or "not provided" in s or "required" in s or "must be set" in s):
        return True
    return False


def _should_retry_api_call(exc: BaseException) -> bool:
    return not _non_retryable_provider_error(exc)


def _cache_key(model: str, messages: list[dict], **kwargs: Any) -> str:
    payload = {"model": model, "messages": messages, "kwargs": kwargs}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


class LLMClient:
    """Provider-agnostic LLM caller with on-disk caching keyed on (model, messages, kwargs)."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir) / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key[:2]}/{key}.json"

    def _read_cache(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def _write_cache(self, key: str, value: dict) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # `default=str` handles litellm's nested wrapper objects in `usage`.
        p.write_text(json.dumps(value, default=str))

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception(_should_retry_api_call),
        reraise=True,
    )
    def _raw_call(self, model: str, messages: list[dict], **kwargs: Any) -> dict:
        resp = litellm.completion(model=model, messages=messages, **kwargs)
        # Normalize to a small dict we control
        return {
            "content": resp.choices[0].message.content,
            "finish_reason": getattr(resp.choices[0], "finish_reason", None),
            "usage": getattr(resp, "usage", {}).__dict__ if getattr(resp, "usage", None) else {},
            "_t": time.time(),
        }

    def chat(
        self,
        model: str,
        system: str | None,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        use_cache: bool | None = None,
    ) -> dict:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        # Default: cache only deterministic calls. Stochastic generations (temperature > 0)
        # must not be cached or they'll collapse to identical outputs.
        if use_cache is None:
            use_cache = temperature == 0.0

        key = _cache_key(model, messages, **kwargs)
        if use_cache:
            cached = self._read_cache(key)
            if cached is not None:
                return cached
        value = self._raw_call(model, messages, **kwargs)
        self._call_count += 1
        if use_cache:
            self._write_cache(key, value)
        return value

    def chat_json(
        self,
        model: str,
        system: str | None,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        use_cache: bool | None = None,
    ) -> Any:
        """Chat call expecting JSON output. Parses and returns the dict/list.

        Defensive against models that don't honor `response_format` (e.g., gpt-5):
        strips fences, then if direct parse fails, extracts the first balanced
        {...} or [...] block from the content."""
        out = self.chat(
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            use_cache=use_cache,
        )
        content = (out.get("content") or "").strip()
        # Strip fences
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
        try:
            return json.loads(content)
        except Exception:
            pass
        # Extract the first balanced JSON object or array
        for opener, closer in (("{", "}"), ("[", "]")):
            i = content.find(opener)
            if i < 0:
                continue
            depth = 0
            for j in range(i, len(content)):
                if content[j] == opener:
                    depth += 1
                elif content[j] == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(content[i:j + 1])
                        except Exception:
                            break
        raise ValueError(f"could not parse JSON from model {model}: {content[:200]!r}")


class EmbeddingClient:
    """Embedding client (also litellm-backed)."""

    def __init__(self, cache_dir: str | Path, model: str):
        self.cache_dir = Path(cache_dir) / "emb"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model

    def _cache_path(self, h: str) -> Path:
        return self.cache_dir / f"{h[:2]}/{h}.json"

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=15),
        retry=retry_if_exception(_should_retry_api_call),
        reraise=True,
    )
    def _raw_embed(self, text: str) -> list[float]:
        resp = litellm.embedding(model=self.model, input=[text])
        vec = resp["data"][0]["embedding"]
        return list(vec)

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()
        p = self._cache_path(h)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        vec = self._raw_embed(text)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(vec))
        return vec
