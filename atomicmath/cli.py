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
    return 1


if __name__ == "__main__":
    sys.exit(main())
