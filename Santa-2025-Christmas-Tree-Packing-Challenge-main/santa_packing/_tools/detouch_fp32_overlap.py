#!/usr/bin/env python3

"""Make a submission robust to fp32 overlap checks (Kaggle-style numeric quirks).

Some Kaggle evaluators / reference implementations effectively reduce numeric
precision (e.g., via float32 or different geometry predicates). Highly packed
solutions that are valid in float64 can become (tiny) overlaps under fp32.

This tool:
- loads a submission.csv
- detects overlaps using `first_overlap_pair` on fp32 points/poses
- for any overlapping puzzle, applies a tiny "push apart" translation on the
  colliding pair (or optional centroid scaling) until fp32 overlap disappears
- rewrites ONLY the adjusted puzzles; all other rows are copied verbatim
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from santa_packing.scoring import first_overlap_pair
from santa_packing.submission_format import fit_xy_in_bounds, format_submission_value
from santa_packing.tree_data import TREE_POINTS


@dataclass(frozen=True)
class PuzzleStr:
    # item_id -> (x_str, y_str, deg_str)
    rows: dict[int, tuple[str, str, str]]

    def to_poses_xy(self, n: int) -> np.ndarray:
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


def _scale_about_centroid_xy(poses: np.ndarray, *, scale: float) -> np.ndarray:
    poses = np.array(poses, dtype=float, copy=True)
    if poses.shape[0] == 0:
        return poses
    center = np.mean(poses[:, 0:2], axis=0)
    poses[:, 0:2] = center[None, :] + (poses[:, 0:2] - center[None, :]) * float(scale)
    return poses


def _has_fp32_overlap(points_f32: np.ndarray, poses_f64: np.ndarray) -> tuple[int, int] | None:
    poses_f32 = np.array(poses_f64, dtype=np.float32, copy=False)
    return first_overlap_pair(points_f32, poses_f32, eps=0.0, mode="kaggle")


def _push_apart_xy(
    poses: np.ndarray,
    *,
    i: int,
    j: int,
    step: float,
    iteration: int,
) -> np.ndarray:
    poses = np.array(poses, dtype=float, copy=True)
    ci = poses[i, 0:2]
    cj = poses[j, 0:2]
    v = ci - cj
    norm = float(np.hypot(float(v[0]), float(v[1])))
    if norm < 1e-12:
        # Deterministic pseudo-random direction.
        a = float(((i + 1) * 104729 + (j + 1) * 10007 + iteration * 7919) % 628318) / 100000.0
        v = np.array([np.cos(a), np.sin(a)], dtype=float)
        norm = 1.0
    delta = float(step) * (1.0 + 0.01 * float(iteration))
    d = (v / norm) * (0.5 * delta)
    poses[i, 0:2] = ci + d
    poses[j, 0:2] = cj - d
    return poses


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Detouch only puzzles that overlap under fp32 checks.")
    ap.add_argument("input", type=Path, help="Input submission.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n to process (default 200).")
    ap.add_argument(
        "--method",
        type=str,
        default="push",
        choices=["push", "scale"],
        help="Detouch method: push (pairwise translation) or scale (centroid scaling).",
    )
    ap.add_argument(
        "--step",
        type=float,
        default=1e-5,
        help="For method=push: base translation step. For method=scale: multiplicative scale step.",
    )
    ap.add_argument("--max-iters", type=int, default=50, help="Max scale attempts per puzzle.")
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated puzzle ids to adjust (e.g. '104,030'). Empty=auto-detect.",
    )
    ns = ap.parse_args(argv)

    nmax = int(ns.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")
    if ns.step <= 0.0:
        raise SystemExit("--step must be > 0")
    if ns.max_iters <= 0:
        raise SystemExit("--max-iters must be > 0")

    only: set[int] | None = None
    if str(ns.only).strip():
        only = {int(x) for x in str(ns.only).split(",") if x.strip()}

    inp = Path(ns.input).resolve()
    puzzles = _load_submission_strings(inp, nmax=nmax)

    points_f32 = np.array(TREE_POINTS, dtype=np.float32)

    out_path = Path(ns.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    adjusted: list[int] = []
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "deg"])
        for n in range(1, nmax + 1):
            ps = puzzles[n]
            poses = ps.to_poses_xy(n)

            need = _has_fp32_overlap(points_f32, poses) is not None
            if only is not None:
                need = n in only

            if not need:
                for i in range(n):
                    x, y, deg = ps.rows[i]
                    w.writerow([f"{n:03d}_{i}", x, y, deg])
                continue

            cur = np.array(poses, dtype=float, copy=True)
            pair = None
            for it in range(int(ns.max_iters)):
                pair = _has_fp32_overlap(points_f32, cur)
                if pair is None:
                    break
                i, j = pair
                if ns.method == "push":
                    cur = _push_apart_xy(cur, i=i, j=j, step=float(ns.step), iteration=it)
                else:
                    # scale
                    scale = 1.0 + float(ns.step) * float(it + 1)
                    cur = _scale_about_centroid_xy(poses, scale=scale)
                cur = fit_xy_in_bounds(cur)
            else:
                raise SystemExit(f"Failed to detouch puzzle {n} after {ns.max_iters} iters; last pair={pair}")

            adjusted.append(n)
            for i in range(n):
                _, _, deg_str = ps.rows[i]
                w.writerow(
                    [
                        f"{n:03d}_{i}",
                        format_submission_value(float(cur[i, 0])),
                        format_submission_value(float(cur[i, 1])),
                        str(deg_str),
                    ]
                )

    if adjusted:
        print(f"wrote: {out_path}")
        print(f"adjusted puzzles: {adjusted[:10]}{'...' if len(adjusted) > 10 else ''}")
    else:
        print(f"wrote: {out_path} (no changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
