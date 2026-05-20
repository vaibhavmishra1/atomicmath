#!/usr/bin/env python3
"""Launch the local atomicmath web UI."""
import sys

from atomicmath.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["web", *sys.argv[1:]]))
