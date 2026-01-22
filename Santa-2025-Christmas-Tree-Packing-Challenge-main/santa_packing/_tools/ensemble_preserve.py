#!/usr/bin/env python3

"""Ensemble multiple submissions per puzzle n, preserving original string values.

Why:
- Some strong Kaggle submissions rely on very fine-grained coordinates.
- Rewriting/quantizing values (even slightly) can introduce overlaps and cause
  Kaggle submission errors.

This tool selects the best candidate per puzzle `n` (minimizing `s_n`) while
writing the chosen `(x,y,deg)` strings exactly as they appear in the source CSV.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from santa_packing.geom_np import packing_score
from santa_packing.scoring import score_submission
from santa_packing.tree_data import TREE_POINTS


@dataclass(frozen=True)
class PuzzleStr:
    # item_id -> (x_str, y_str, deg_str) including 's' prefix
    rows: dict[int, tuple[str, str, str]]

    def to_poses(self, n: int) -> np.ndarray:
        poses = np.zeros((n, 3), dtype=float)
        for i in range(n):
            x, y, deg = self.rows[i]
            poses[i, 0] = float(x.lstrip("s"))
            poses[i, 1] = float(y.lstrip("s"))
            poses[i, 2] = float(deg.lstrip("s"))
        return poses


def _load_submission_strings(path: Path, *, nmax: int) -> dict[int, PuzzleStr]:
    puzzles: dict[int, dict[int, tuple[str, str, str]]] = {n: {} for n in range(1, nmax + 1)}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if (r.fieldnames or []) != ["id", "x", "y", "deg"]:
            raise ValueError(f"Unexpected header in {path}: {r.fieldnames}")
        for row in r:
            gid_s, iid_s = str(row["id"]).split("_", 1)
            n = int(gid_s)
            if n < 1 or n > nmax:
                continue
            i = int(iid_s)
            puzzles[n][i] = (str(row["x"]), str(row["y"]), str(row["deg"]))

    out: dict[int, PuzzleStr] = {}
    for n in range(1, nmax + 1):
        rows = puzzles[n]
        if len(rows) != n or any(i not in rows for i in range(n)):
            raise ValueError(f"{path} missing/invalid puzzle {n}: got {len(rows)} rows")
        out[n] = PuzzleStr(rows=dict(rows))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-n ensemble preserving original x/y/deg strings (no quantization).")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv")
    ap.add_argument("--nmax", type=int, default=200)
    ap.add_argument(
        "--pin",
        action="append",
        default=[],
        help="Pin a puzzle to a specific source CSV, e.g. --pin 104=path/to/submission.csv (repeatable).",
    )
    ap.add_argument("inputs", nargs="+", type=Path, help="Input submission.csv files")
    ns = ap.parse_args(argv)

    nmax = int(ns.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")
    inputs = [Path(p).resolve() for p in ns.inputs]
    if len(inputs) < 1:
        raise SystemExit("Need at least one input CSV")

    points = np.array(TREE_POINTS, dtype=float)

    pinned: dict[int, Path] = {}
    for raw in list(ns.pin or []):
        s = str(raw)
        if "=" in s:
            left, right = s.split("=", 1)
        elif ":" in s:
            left, right = s.split(":", 1)
        else:
            raise SystemExit(f"Invalid --pin {raw!r}; expected N=PATH")
        n = int(left.strip())
        if n < 1 or n > nmax:
            raise SystemExit(f"--pin puzzle id out of range: {n} (nmax={nmax})")
        p = Path(right.strip()).resolve()
        if not p.is_file():
            raise SystemExit(f"--pin file not found: {p}")
        pinned[n] = p

    parsed = []
    for p in inputs:
        if not p.is_file():
            raise SystemExit(f"Input not found: {p}")
        parsed.append((p, _load_submission_strings(p, nmax=nmax)))

    pinned_parsed: dict[int, tuple[Path, PuzzleStr]] = {}
    for n, p in pinned.items():
        pinned_parsed[n] = (p, _load_submission_strings(p, nmax=nmax)[n])

    chosen_by_n: dict[int, tuple[Path, PuzzleStr, float]] = {}
    for n in range(1, nmax + 1):
        if n in pinned_parsed:
            p, ps = pinned_parsed[n]
            poses = ps.to_poses(n)
            s = float(packing_score(points, poses))
            chosen_by_n[n] = (p, ps, s)
            continue
        best = None
        for p, puzzles in parsed:
            ps = puzzles[n]
            poses = ps.to_poses(n)
            s = float(packing_score(points, poses))
            if best is None or s < best[2] - 1e-15:
                best = (p, ps, s)
        assert best is not None
        chosen_by_n[n] = best

    out_path = Path(ns.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "deg"])
        for n in range(1, nmax + 1):
            src, ps, _ = chosen_by_n[n]
            _ = src  # keep for debug if needed
            for i in range(n):
                x, y, deg = ps.rows[i]
                w.writerow([f"{n:03d}_{i}", x, y, deg])

    res = score_submission(out_path, nmax=nmax, check_overlap=False, require_complete=True)
    print(f"wrote: {out_path}")
    print(f"score(no-overlap-check): {res.score:.12f}")
    if pinned_parsed:
        pins = ", ".join(f"{n:03d}->{p.name}" for n, (p, _) in sorted(pinned_parsed.items()))
        print(f"pinned: {pins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
