#!/usr/bin/env python3

"""CLI to score a `submission.csv` locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from santa_packing.scoring import score_submission


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, score the submission, and print JSON to stdout."""
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="Score a Santa 2025 submission.csv")
    ap.add_argument("submission", type=Path, help="Path to submission.csv")
    ap.add_argument("--nmax", type=int, default=None, help="Max puzzle n to score")
    ap.add_argument(
        "--no-overlap",
        action="store_true",
        help="Skip overlap checks (faster, but unsafe)",
    )
    ap.add_argument(
        "--overlap-mode",
        type=str,
        default="kaggle",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap predicate used when overlap checks are enabled (strict/kaggle allow touching; conservative counts touching).",
    )
    ap.add_argument(
        "--no-require-complete",
        action="store_true",
        help="Do not fail if puzzles 1..nmax are missing",
    )
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = ap.parse_args(argv)

    result = score_submission(
        args.submission,
        nmax=args.nmax,
        check_overlap=not args.no_overlap,
        overlap_mode=args.overlap_mode,
        require_complete=not args.no_require_complete,
    )

    data = result.to_json()
    if args.pretty:
        print(json.dumps(data, indent=2))
    else:
        print(json.dumps(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
