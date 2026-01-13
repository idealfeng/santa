#!/usr/bin/env python3

"""Build the optional `santa_packing.fastcollide` C++ extension in-place."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Invoke setuptools build_ext --inplace for the extension."""
    root = Path(__file__).resolve().parents[2]
    setup = root / "scripts" / "build" / "setup_fastcollide.py"
    if not setup.is_file():
        raise SystemExit(f"Missing setup script: {setup}")

    cmd = [sys.executable, str(setup), "build_ext", "--inplace"]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
