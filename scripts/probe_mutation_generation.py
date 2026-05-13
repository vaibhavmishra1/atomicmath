#!/usr/bin/env python3
"""Run the single-question mutation probe.

Example:
    python3 scripts/probe_mutation_generation.py --config examples/config.example.yaml --limit 10 --n 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atomicmath.config import load_config
from atomicmath.mutation_pipeline import run_mutation_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe single-question mutation generation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--n", type=int, default=1, help="Candidates per seed.")
    parser.add_argument("--force-hinges", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--no-ingest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    result = run_mutation_probe(
        cfg,
        limit=args.limit,
        n=args.n,
        force_hinges=args.force_hinges,
        judge=not args.no_judge,
        ingest=not args.no_ingest,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
