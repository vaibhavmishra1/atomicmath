"""Orchestrator: ingest → index → post-index hook → retrosynthesis loop → publish."""
from __future__ import annotations

import hashlib
import json
import random
import time

from rich.console import Console

from . import composer, curriculum, exemplars, quality, realizer
from .config import Config
from .db import Store
from .env_guard import check_config_api_env
from .index_pass import index_seeds
from .ingest import ingest_dataset, normalize_topics
from .llm import EmbeddingClient, LLMClient
from .minhash_util import minhash_signature
from .sanity import post_index_sanity
from .taxonomy import Taxonomy
from .trace import trace_event
from .verifier import NoveltyIndex, verify_correctness

console = Console()


def run(cfg: Config, tax: Taxonomy) -> None:
    store = Store(cfg.storage.db_path)
    store.clear_pipeline_events()
    trace_event(
        store,
        "run",
        "run.start",
        message="pipeline started",
        payload={
            "db_path": cfg.storage.db_path,
            "target_count": cfg.runtime.target_count,
            "max_attempts": cfg.runtime.max_attempts,
            "dry_run": cfg.runtime.dry_run,
        },
    )
    check_config_api_env(cfg)
    llm = LLMClient(cache_dir=cfg.storage.cache_dir)
    embedder = EmbeddingClient(cache_dir=cfg.storage.cache_dir, model=cfg.models.embedder)

    console.rule("[bold]1. Ingest")
    ingest_dataset(cfg, store)

    console.rule("[bold]2. Topic normalization")
    normalize_topics(cfg, store, embedder, llm)

    console.rule("[bold]3. Index pass")
    index_seeds(cfg, store, embedder)

    console.rule("[bold]4. Post-index")
    trace_event(store, "sanity", "sanity.check.start", message="post-index hook")
    ok, reason = post_index_sanity(cfg, store)
    console.print(f"  {reason}")
    trace_event(store, "sanity", "sanity.check.done", message=reason, payload={"passed": ok})

    if cfg.runtime.dry_run:
        console.print("[yellow]Dry run — stopping before synthesis.[/]")
        trace_event(store, "run", "run.stop", message="dry run — no synthesis")
        return

    console.rule("[bold]5. Exemplar pool")
    exemplars.ensure_pool(cfg, store, llm, tax)

    console.rule("[bold]6. Synthesis loop (composer → realizer)")
    trace_event(
        store,
        "synthesis",
        "synthesis.loop.start",
        message=f"target {cfg.runtime.target_count} accepts",
    )
    novelty_index = NoveltyIndex()
    novelty_index.load_from_store(store)
    rng = random.Random(int(time.time()))

    target = cfg.runtime.target_count
    max_attempts = cfg.runtime.max_attempts
    attempts = 0
    while store.output_count() < target and attempts < max_attempts:
        attempts += 1
        try:
            _composer_round(cfg, store, llm, embedder, tax, novelty_index, rng, attempts)
        except Exception as e:
            console.print(f"  [red]round failed[/]: {e}")
            store.log_round(time.time(), None, None, "exception", str(e), None, None)
            trace_event(
                store,
                "synthesis",
                "synthesis.exception",
                attempt=attempts,
                message=str(e),
            )
        if attempts % 25 == 0:
            console.print(f"  attempts={attempts} accepts={store.output_count()} target={target}")

    console.print(f"[bold green]synthesis done[/]: {store.output_count()} accepts after {attempts} attempts")
    trace_event(
        store,
        "synthesis",
        "synthesis.loop.done",
        message=f"{store.output_count()} accepts in {attempts} attempts",
    )

    console.rule("[bold]7. Publish")
    from .publish import publish_dataset

    publish_dataset(cfg, store)
    trace_event(store, "run", "run.complete", message="publish finished")


def _composer_round(
    cfg: Config,
    store: Store,
    llm: LLMClient,
    embedder: EmbeddingClient,
    tax: Taxonomy,
    novelty_index: NoveltyIndex,
    rng: random.Random,
    synth_idx: int,
) -> None:
    trace_event(
        store,
        "synthesis",
        "synthesis.attempt",
        attempt=synth_idx,
        message=f"round {synth_idx} · outputs={store.output_count()}/{cfg.runtime.target_count}",
    )
    brief = curriculum.sample_brief(cfg, store, tax, rng)
    brief_json = json.dumps(brief.__dict__, sort_keys=True)
    brief_id = hashlib.sha256(brief_json.encode()).hexdigest()[:16]
    ex = exemplars.retrieve(store, brief, k=4)
    trace_event(
        store,
        "synthesis",
        "synthesis.brief",
        attempt=synth_idx,
        message="sampled brief",
        payload={"brief": brief.__dict__, "n_exemplars": len(ex)},
    )
    try:
        scaffold = composer.design(llm, cfg, tax, brief, ex)
    except Exception as e:
        store.log_round(time.time(), brief.topic, None, "reject", "composer", None, {"error": str(e)})
        trace_event(store, "synthesis", "synthesis.reject", attempt=synth_idx, message="composer", payload={"error": str(e)})
        return

    scaffold_json = json.dumps(scaffold, sort_keys=True)
    scaffold_id = hashlib.sha256(f"{brief_id}\x1e{scaffold_json}".encode()).hexdigest()[:16]
    store.insert_brief(brief_id, json.dumps(brief.__dict__, ensure_ascii=False))
    store.insert_scaffold(scaffold_id, brief_id, scaffold_json)

    candidate, refinement_rounds = realizer.write(
        llm,
        embedder,
        cfg,
        scaffold,
        topic_label=brief.topic,
        parents=[],
        rng=rng,
        event_store=store,
        synth_attempt=synth_idx,
    )
    if candidate is None:
        store.log_round(
            time.time(),
            brief.topic,
            None,
            "reject",
            "realizer_no_survivor",
            None,
            {"brief_id": brief_id, "scaffold_id": scaffold_id},
        )
        trace_event(
            store,
            "synthesis",
            "synthesis.reject",
            attempt=synth_idx,
            message="realizer_no_survivor",
            payload={"brief_id": brief_id, "scaffold_id": scaffold_id},
        )
        return

    correctness = verify_correctness(
        llm,
        cfg,
        candidate.question,
        candidate.answer,
        generator_model=None,
        event_store=store,
        synth_attempt=synth_idx,
    )
    if not correctness.passed:
        store.log_round(
            time.time(),
            brief.topic,
            None,
            "reject",
            "correctness",
            None,
            {"agree": correctness.agreements, "brief_id": brief_id},
        )
        trace_event(
            store,
            "synthesis",
            "synthesis.reject",
            attempt=synth_idx,
            message="correctness",
            payload={
                "agreements": correctness.agreements,
                "candidate_question": candidate.question,
                "candidate_answer": candidate.answer,
                "verifier_answers": correctness.answers,
            },
        )
        return

    quality_verdict = quality.judge_quality(
        llm,
        cfg,
        brief,
        scaffold,
        candidate,
        event_store=store,
        synth_attempt=synth_idx,
    )
    if not quality_verdict.passed:
        store.log_round(
            time.time(),
            brief.topic,
            None,
            "reject",
            "quality_routine",
            None,
            {
                "brief_id": brief_id,
                "scaffold_id": scaffold_id,
                "quality": quality_verdict.__dict__,
            },
        )
        trace_event(
            store,
            "synthesis",
            "synthesis.reject",
            attempt=synth_idx,
            message="quality_routine",
            payload={
                "candidate_question": candidate.question,
                "candidate_answer": candidate.answer,
                "quality": quality_verdict.__dict__,
            },
        )
        return

    trace_event(store, "synthesis", "verify.novelty.start", attempt=synth_idx, message="MinHash / embedding gates")
    cand_mh = minhash_signature(candidate.question, candidate.answer)
    cand_emb = embedder.embed(candidate.question[:8000])
    novelty = novelty_index.check(cand_mh, cand_emb, cfg)
    trace_event(
        store,
        "synthesis",
        "verify.novelty.done",
        attempt=synth_idx,
        payload={
            "passed": novelty.passed,
            "failure_kind": novelty.failure_kind,
            "nearest_minhash": novelty.nearest_minhash,
            "nearest_cosine": novelty.nearest_cosine,
        },
    )
    if not novelty.passed:
        store.log_round(
            time.time(),
            brief.topic,
            None,
            "reject",
            novelty.failure_kind or "novelty",
            None,
            {"brief_id": brief_id},
        )
        trace_event(
            store,
            "synthesis",
            "synthesis.reject",
            attempt=synth_idx,
            message=novelty.failure_kind or "novelty",
        )
        return

    out_id = hashlib.sha256(f"{candidate.question}\x1e{candidate.answer}".encode()).hexdigest()[:16]
    audit = {
        "path": "composer_realizer",
        "brief": brief.__dict__,
        "brief_id": brief_id,
        "scaffold_id": scaffold_id,
        "scaffold": scaffold,
        "spot_check_correctness": candidate.correctness,
        "anti_distance": candidate.anti_distance,
        "novelty_max_minhash": novelty.nearest_minhash,
        "novelty_max_cosine": novelty.nearest_cosine,
        "verifier_agreements": correctness.agreements,
        "verifier_answers": correctness.answers,
        "quality": quality_verdict.__dict__,
        "parent_fingerprints": [],
    }
    clean_accept = refinement_rounds == 1
    kwargs_out = dict(
        id=out_id,
        question=candidate.question,
        answer=candidate.answer,
        topic=candidate.topic or brief.topic,
        parent_seed_ids=json.dumps([]),
        parent_fingerprints=json.dumps([]),
        brief_id=brief_id,
        scaffold_id=scaffold_id,
        embedding=json.dumps(cand_emb),
        minhash=json.dumps(cand_mh),
        audit_json=json.dumps(audit),
        accepted_at=time.time(),
        clean_accept=int(clean_accept),
        refinement_rounds=refinement_rounds,
    )
    store.insert_output(**kwargs_out)
    novelty_index.add(f"out:{out_id}", cand_mh, cand_emb)
    store.bump_coverage(brief.primary_concept, brief.secondary_concept)
    store.log_round(time.time(), brief.topic, None, "accept", None, out_id, None)
    trace_event(
        store,
        "synthesis",
        "synthesis.accept",
        attempt=synth_idx,
        message=f"output {out_id}",
        payload={
            "topic": brief.topic,
            "brief_id": brief_id,
            "scaffold_id": scaffold_id,
            "output_id": out_id,
            "candidate_question": candidate.question,
            "candidate_answer": candidate.answer,
        },
    )
