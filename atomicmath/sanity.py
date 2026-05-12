"""Post-index hook (placeholder for optional quality gates)."""

from __future__ import annotations

from .config import Config
from .db import Store


def post_index_sanity(cfg: Config, store: Store) -> tuple[bool, str]:
    _ = cfg, store
    return True, "skipped (no post-index gate)"
