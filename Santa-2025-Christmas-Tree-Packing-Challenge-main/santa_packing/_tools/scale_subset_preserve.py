#!/usr/bin/env python3

"""Scale selected puzzles inside a submission.csv while preserving other rows verbatim.

This is a pragmatic way to add a tiny bit of "slack" when Kaggle flags an overlap
that is hard to reproduce locally due to numeric tolerances.

Unlike `scale_puzzle.py`, this tool:
- only modifies specified puzzle ids
- copies all other `x/y/deg` strings exactly as-is (avoids re-serialization)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from santa_packing.submission_format import fit_xy_in_bounds, format_submission_value


@dataclass(frozen=True)
class PuzzleStr:
    rows: dict[int, tuple[str, str, str]]  # item_id -> (x_str, y_str, deg_str)

    def to_poses(self, n: int) -> np.ndarray:
        poses = np.zeros((n, 3), dtype=float)
        for i in range(n):
            x, y, deg = self.rows[i]
            poses[i, 0] = float(str(x).lstrip("sS"))
            poses[i, 1] = float(str(y).lstrip("sS"))
            poses[i, 2] = float(str(deg).lstrip("sS"))
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
    ap = argparse.ArgumentParser(description="Scale selected puzzles about centroid (preserving other rows).")
    ap.add_argument("--base", type=Path, required=True, help="Input submission.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n in the CSV (default 200).")
    ap.add_argument(
        "--puzzles",
        type=str,
        required=True,
        help="Comma-separated puzzle ids to scale (e.g. '104,030').",
    )
    ap.add_argument("--scale", type=float, required=True, help="Scale factor (>1 expands).")
    ns = ap.parse_args(argv)

    nmax = int(ns.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")

    scale = float(ns.scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise SystemExit("--scale must be a positive finite float")

    puzzles_set = {int(x) for x in str(ns.puzzles).split(",") if x.strip()}
    if not puzzles_set:
        raise SystemExit("--puzzles must be non-empty")
    if any(n < 1 or n > nmax for n in puzzles_set):
        raise SystemExit(f"--puzzles must be within [1,{nmax}]")

    base = Path(ns.base).resolve()
    data = _load_submission_strings(base, nmax=nmax)

    out = Path(ns.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "deg"])
        for n in range(1, nmax + 1):
            ps = data[n]
            if n not in puzzles_set:
                for i in range(n):
                    x, y, deg = ps.rows[i]
                    w.writerow([f"{n:03d}_{i}", x, y, deg])
                continue

            poses = ps.to_poses(n)
            center = np.mean(poses[:, 0:2], axis=0)
            poses[:, 0:2] = center[None, :] + (poses[:, 0:2] - center[None, :]) * scale
            poses = fit_xy_in_bounds(poses)
            for i in range(n):
                _, _, deg_str = ps.rows[i]
                w.writerow(
                    [
                        f"{n:03d}_{i}",
                        format_submission_value(float(poses[i, 0])),
                        format_submission_value(float(poses[i, 1])),
                        str(deg_str),
                    ]
                )

    print(f"wrote: {out}")
    print(f"scaled puzzles: {sorted(puzzles_set)} scale={scale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

