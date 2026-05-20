"""CLI entry point for atomicmath lineage synthesis."""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.traceback import install as rich_traceback_install

from .config import load_config
from .lineage import run_lineage
from .webui import serve as serve_web

console = Console()


def main(argv: list[str] | None = None) -> int:
    rich_traceback_install(show_locals=False)

    parser = argparse.ArgumentParser(
        prog="atomicmath",
        description="Iterative synthesis of new math problems from solved seed problems.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run the lineage pipeline end-to-end.")
    p_run.add_argument("--config", required=True, help="Path to YAML config.")
    p_run.add_argument("--num-seeds", type=int, default=None, help="Override input.max_seeds.")
    p_run.add_argument("--seed", type=int, default=None, help="Override deterministic sampling seed.")
    p_run.add_argument("--dry-run", action="store_true", help="Load and sample seeds but skip LLM generation.")
    p_run.add_argument("--push-to-hub", action="store_true", help="Upload generated rows to Hugging Face.")
    p_run.add_argument("--dataset-id", default=None, help="Override output.dataset for upload.")

    p_web = sub.add_parser("web", help="Launch the local web UI.")
    p_web.add_argument("--config", default="examples/config.example.yaml", help="Path to default YAML config.")
    p_web.add_argument("--host", default="127.0.0.1", help="Bind address.")
    p_web.add_argument("--port", type=int, default=8765, help="TCP port.")
    p_web.add_argument("--run-dir", default="./out/web_runs", help="Directory for web run state and outputs.")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        cfg = load_config(args.config)
        if args.dry_run:
            cfg.runtime.dry_run = True
        if args.dataset_id:
            cfg.output.dataset = args.dataset_id

        console.print(
            "[bold]atomicmath[/] lineage run "
            f"input={cfg.input.dataset}:{cfg.input.split} "
            f"output={cfg.output.dataset}"
        )
        result = run_lineage(
            cfg,
            num_seeds=args.num_seeds,
            seed=args.seed,
            push_to_hub=args.push_to_hub or None,
            dataset_id=args.dataset_id,
        )
        console.print_json(data=result.summary)
        console.print(f"[bold green]wrote[/] {result.local_path}")
        console.print(f"[bold green]summary[/] {result.summary_path}")
        if result.hub_pushed:
            console.print(f"[bold green]published[/] https://huggingface.co/datasets/{result.hub_dataset}")
        return 0
    if args.cmd == "web":
        serve_web(args.config, host=args.host, port=args.port, run_dir=args.run_dir)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
