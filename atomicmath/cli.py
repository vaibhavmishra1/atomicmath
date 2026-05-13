"""CLI entry point: `atomicmath run --config config.yaml`."""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.traceback import install as rich_traceback_install

from .config import load_config
from .pipeline import run as pipeline_run
from .taxonomy import load_taxonomy

console = Console()


def main(argv: list[str] | None = None) -> int:
    rich_traceback_install(show_locals=False)
    p = argparse.ArgumentParser(prog="atomicmath", description="Stochastic synthesis of novel math problems.")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="Run the pipeline end-to-end.")
    p_run.add_argument("--config", required=True, help="Path to YAML config.")
    p_run.add_argument("--dry-run", action="store_true", help="Stop before synthesis (ingest, normalize, index only).")
    p_run.add_argument("--target-count", type=int, default=None, help="Override target output count.")
    p_run.add_argument(
        "--dashboard",
        type=int,
        metavar="PORT",
        default=None,
        help="Serve the live trace UI on http://127.0.0.1:PORT/ while the pipeline runs (same db as config).",
    )

    p_watch = sub.add_parser("watch", help="Serve only the live trace UI (read-only; pipeline runs elsewhere).")
    p_watch.add_argument("--config", required=True, help="Path to YAML config (for storage.db_path).")
    p_watch.add_argument("--host", default="127.0.0.1", help="Bind address.")
    p_watch.add_argument("--port", type=int, default=8765, help="TCP port.")

    p_mut = sub.add_parser("mutate", help="Single-question mutation pipeline.")
    mut_sub = p_mut.add_subparsers(dest="mutate_cmd", required=True)

    p_m_extract = mut_sub.add_parser("extract-hinges", help="Extract mutation hinge notes for one or more seeds.")
    p_m_extract.add_argument("--config", required=True)
    p_m_extract.add_argument("--seed-id", default=None)
    p_m_extract.add_argument("--limit", type=int, default=1)
    p_m_extract.add_argument("--force", action="store_true")
    p_m_extract.add_argument("--no-ingest", action="store_true")

    p_m_prompt = mut_sub.add_parser("build-prompt", help="Print the full mutation prompt for one seed.")
    p_m_prompt.add_argument("--config", required=True)
    p_m_prompt.add_argument("--seed-id", required=True)

    p_m_gen = mut_sub.add_parser("generate", help="Generate mutation candidates for one seed.")
    p_m_gen.add_argument("--config", required=True)
    p_m_gen.add_argument("--seed-id", required=True)
    p_m_gen.add_argument("--n", type=int, default=1)
    p_m_gen.add_argument("--judge", action="store_true")

    p_m_judge = mut_sub.add_parser("judge", help="Judge one pending mutation episode.")
    p_m_judge.add_argument("--config", required=True)
    p_m_judge.add_argument("--episode-id", required=True)

    p_m_probe = mut_sub.add_parser("probe", help="Run extract → generate → judge on a small seed batch.")
    p_m_probe.add_argument("--config", required=True)
    p_m_probe.add_argument("--limit", type=int, default=10)
    p_m_probe.add_argument("--n", type=int, default=1)
    p_m_probe.add_argument("--force-hinges", action="store_true")
    p_m_probe.add_argument("--no-judge", action="store_true")
    p_m_probe.add_argument("--no-ingest", action="store_true")

    p_m_publish = mut_sub.add_parser("publish", help="Publish accepted mutation episodes to Hugging Face.")
    p_m_publish.add_argument("--config", required=True)
    p_m_publish.add_argument("--dataset", default=None, help="HF dataset id, e.g. vibhuiitj/math500-output.")
    p_m_publish.add_argument("--split", default="train")
    p_m_publish.add_argument("--private", action="store_true", help="Publish as private, overriding config.")
    p_m_publish.add_argument("--dry-run", action="store_true")

    p_m_memory = mut_sub.add_parser("memory", help="Inspect distilled global mutation memory.")
    p_m_memory.add_argument("--config", required=True)
    p_m_memory.add_argument("--kind", choices=["all", "success", "failure"], default="all")
    p_m_memory.add_argument("--topic", default=None, help="Prioritize memories from this normalized topic.")
    p_m_memory.add_argument("--limit", type=int, default=20)
    p_m_memory.add_argument("--backfill", action="store_true", help="Distill existing judged episodes into memory first.")
    p_m_memory.add_argument("--json", action="store_true")

    args = p.parse_args(argv)
    if args.cmd == "run":
        cfg = load_config(args.config)
        if args.dry_run:
            cfg.runtime.dry_run = True
        if args.target_count is not None:
            cfg.runtime.target_count = args.target_count
        tax = load_taxonomy(cfg.taxonomy.path)
        console.print(f"[bold]atomicmath[/] starting: input={cfg.input.dataset}, output={cfg.output.dataset}")
        console.print(f"  taxonomy: {tax.n_concepts()} concepts, {len(tax.topics)} topics, {len(tax.bond_types)} bond types")
        if args.dashboard is not None:
            import threading

            import uvicorn

            from .watch import build_app

            port = int(args.dashboard)
            app = build_app(cfg.storage.db_path)

            def _serve() -> None:
                uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

            threading.Thread(target=_serve, daemon=True).start()
            console.print(f"[bold green]trace dashboard[/] http://127.0.0.1:{port}/")
        pipeline_run(cfg, tax)
        return 0
    if args.cmd == "watch":
        from .watch import serve_forever

        serve_forever(args.config, host=args.host, port=args.port)
        return 0
    if args.cmd == "mutate":
        cfg = load_config(args.config)
        from .db import Store
        from .llm import LLMClient
        from .mutation_generator import generate_mutations_for_seed
        from .mutation_hinges import ensure_hinges_for_seed, extract_hinges_for_seed
        from .mutation_memory import backfill_mutation_experiences, experience_row_to_dict
        from .mutation_pipeline import ensure_mutation_seed_pool, run_mutation_probe
        from .mutation_prompt import build_plan_generate_prompt
        from .mutation_publish import publish_accepted_mutations
        from .mutation_quality import judge_mutation_episode

        store = Store(cfg.storage.db_path)
        if args.mutate_cmd == "extract-hinges":
            if not args.no_ingest:
                ensure_mutation_seed_pool(cfg, store, needed=args.limit, ingest=True)
            seed_ids = [args.seed_id] if args.seed_id else [r["id"] for r in store.list_mutation_seed_rows(args.limit)]
            llm = LLMClient(cache_dir=cfg.storage.cache_dir)
            for seed_id in seed_ids:
                hinges = extract_hinges_for_seed(cfg, store, llm, seed_id, force=args.force)
                console.rule(f"seed {seed_id}")
                for i, h in enumerate(hinges, start=1):
                    console.print(f"[bold]HINGE {i}: {h.label}[/]\n{h.text}\n")
            return 0
        if args.mutate_cmd == "build-prompt":
            seed = store.get_seed(args.seed_id)
            if seed is None:
                console.print(f"[red]unknown seed_id[/]: {args.seed_id}")
                return 1
            llm = LLMClient(cache_dir=cfg.storage.cache_dir)
            hinges = ensure_hinges_for_seed(cfg, store, llm, args.seed_id)
            console.print(build_plan_generate_prompt(cfg, store, seed, hinges))
            return 0
        if args.mutate_cmd == "generate":
            llm = LLMClient(cache_dir=cfg.storage.cache_dir)
            cands = generate_mutations_for_seed(cfg, store, llm, args.seed_id, n=args.n)
            for cand in cands:
                console.rule(f"episode {cand.episode_id}")
                console.print(f"[bold]mutation[/]: {cand.mutation_used}")
                console.print(cand.new_question)
                console.print(f"[bold]answer[/]: {cand.answer}")
                if args.judge:
                    verdict = judge_mutation_episode(cfg, store, llm, cand.episode_id)
                    console.print(f"[bold]judge[/]: {'accepted' if verdict.passed else 'rejected'} {verdict.failure_kind or ''}")
            return 0
        if args.mutate_cmd == "judge":
            llm = LLMClient(cache_dir=cfg.storage.cache_dir)
            verdict = judge_mutation_episode(cfg, store, llm, args.episode_id)
            console.print_json(data=verdict.__dict__)
            return 0
        if args.mutate_cmd == "probe":
            result = run_mutation_probe(
                cfg,
                limit=args.limit,
                n=args.n,
                force_hinges=args.force_hinges,
                judge=not args.no_judge,
                ingest=not args.no_ingest,
            )
            console.print_json(data=result.__dict__)
            return 0
        if args.mutate_cmd == "publish":
            repo_id = publish_accepted_mutations(
                cfg,
                store,
                dataset_id=args.dataset,
                private=True if args.private else None,
                split=args.split,
                dry_run=args.dry_run,
            )
            console.print(f"[bold green]published[/] https://huggingface.co/datasets/{repo_id}")
            return 0
        if args.mutate_cmd == "memory":
            backfilled = 0
            if args.backfill:
                backfilled = backfill_mutation_experiences(cfg, store, limit=None)
            rows = store.list_mutation_experiences(
                kind=args.kind,
                topic_norm=args.topic,
                limit=args.limit,
                prioritize_topic=bool(args.topic),
            )
            data = [experience_row_to_dict(row) for row in rows]
            if args.json:
                console.print_json(data={"backfilled": backfilled, "memories": data})
            else:
                console.print(
                    f"[bold]mutation memory[/] active={store.count_mutation_experiences(active_only=True)} "
                    f"shown={len(data)} backfilled={backfilled}"
                )
                for item in data:
                    console.rule(f"{item['kind']} {item['id']} weight={item['weight']}")
                    console.print(f"[bold]topic[/]: {item['topic_norm'] or 'global'}")
                    console.print(f"[bold]mutation[/]: {item['mutation_used'] or 'unknown'}")
                    if item["failure_kind"]:
                        console.print(f"[bold]failure[/]: {item['failure_kind']}")
                    console.print(item["lesson"])
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
