#!/usr/bin/env python3

"""Wrapper script for `santa_packing._tools.train_l2o_bc`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return the repository root (directory containing `pyproject.toml`)."""
    here = Path(__file__).resolve()
    for cand in (here.parent, *here.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    return here.parent


def main() -> int:
    """Execute `python -m santa_packing._tools.train_l2o_bc` in the repo root."""
    root = _repo_root()
    cmd = [sys.executable, "-m", "santa_packing._tools.train_l2o_bc", *sys.argv[1:]]
    return int(subprocess.call(cmd, cwd=str(root)))


if __name__ == "__main__":
    raise SystemExit(main())
