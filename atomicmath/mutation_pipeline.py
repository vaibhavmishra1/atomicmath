"""End-to-end helpers for the single-question mutation path."""
from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console

from .config import Config
from .db import Store
from .ingest import ingest_dataset
from .llm import LLMClient
from .mutation_generator import MutationCandidate, generate_mutations_for_seed
from .mutation_hinges import extract_hinges_for_seed, ensure_hinges_for_seed
from .mutation_quality import MutationVerdict, judge_mutation_episode

console = Console()


@dataclass
class ProbeResult:
    seeds_seen: int = 0
    hinges_extracted: int = 0
    generated: int = 0
    accepted: int = 0
    rejected: int = 0
    errors: list[str] = field(default_factory=list)


def ensure_mutation_seed_pool(cfg: Config, store: Store, *, needed: int, ingest: bool = True) -> None:
    rows = store.list_mutation_seed_rows(limit=needed)
    if len(rows) >= needed or not ingest:
        return
    ingest_dataset(cfg, store)


def run_mutation_probe(
    cfg: Config,
    *,
    limit: int = 10,
    n: int = 1,
    force_hinges: bool = False,
    judge: bool = True,
    ingest: bool = True,
) -> ProbeResult:
    store = Store(cfg.storage.db_path)
    ensure_mutation_seed_pool(cfg, store, needed=limit, ingest=ingest)
    rows = store.list_mutation_seed_rows(limit=limit)
    llm = LLMClient(cache_dir=cfg.storage.cache_dir)
    result = ProbeResult(seeds_seen=len(rows))
    if not rows:
        result.errors.append("no eligible seeds with solution text available")
        return result

    for si, seed in enumerate(rows, start=1):
        seed_id = seed["id"]
        console.print(f"[bold]mutation seed {si}/{len(rows)}[/] {seed_id}")
        try:
            before = store.count_seed_hinges(seed_id)
            if force_hinges or before == 0:
                extract_hinges_for_seed(cfg, store, llm, seed_id, force=force_hinges)
            hinges = ensure_hinges_for_seed(cfg, store, llm, seed_id)
            result.hinges_extracted += len(hinges)
            console.print(f"  hinges: {len(hinges)}")

            candidates: list[MutationCandidate] = generate_mutations_for_seed(cfg, store, llm, seed_id, n=n)
            result.generated += len(candidates)
            console.print(f"  generated: {len(candidates)}")

            if judge:
                for cand in candidates:
                    verdict: MutationVerdict = judge_mutation_episode(cfg, store, llm, cand.episode_id)
                    if verdict.passed:
                        result.accepted += 1
                        console.print(f"  [green]accepted[/] {cand.episode_id}")
                    else:
                        result.rejected += 1
                        console.print(f"  [yellow]rejected[/] {cand.episode_id}: {verdict.failure_kind}")
        except Exception as exc:
            msg = f"{seed_id}: {exc}"
            result.errors.append(msg)
            console.print(f"  [red]error[/] {msg}")
    return result
