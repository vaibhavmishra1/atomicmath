#!/usr/bin/env python3
"""Convenience wrapper for `python3 -m atomicmath.cli run ...`."""
import sys

from atomicmath.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
