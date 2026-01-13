"""Safe per-puzzle blending utilities for `submission.csv`.

This module helps combine two submissions (base + candidate) while preserving
feasibility in a chosen overlap mode:
- per puzzle `n`, keep the candidate only if it is overlap-free and strictly
  improves the packing score `s_n` (side length).
- optionally, attempt to repair overlapping-but-promising candidate puzzles
  (bounded) using the same finalizer used by the generator.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np

from santa_packing.cli.generate_submission import _finalize_puzzle
from santa_packing.geom_np import packing_score
from santa_packing.scoring import OverlapMode, first_overlap_pair, load_submission
from santa_packing.submission_format import fit_xy_in_bounds, format_submission_value, quantize_for_submission
from santa_packing.tree_data import TREE_POINTS

RepairMode = Literal["none", "finalize"]


def _canonicalize_poses(poses: np.ndarray) -> np.ndarray:
    poses = np.array(poses, dtype=float, copy=True)
    if poses.shape[0] == 0:
        return poses
    poses[:, 2] = np.mod(poses[:, 2], 360.0)
    poses = fit_xy_in_bounds(poses)
    poses = quantize_for_submission(poses)
    return poses


def _write_submission(path: Path, puzzles: dict[int, np.ndarray], *, nmax: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "deg"])
        for n in range(1, int(nmax) + 1):
            poses = puzzles.get(n)
            if poses is None or poses.shape != (n, 3):
                raise ValueError(f"Missing puzzle {n} or wrong shape {None if poses is None else poses.shape}")
            poses = _canonicalize_poses(poses)
            for i, (x, y, deg) in enumerate(poses):
                w.writerow(
                    [
                        f"{n:03d}_{i}",
                        format_submission_value(float(x)),
                        format_submission_value(float(y)),
                        format_submission_value(float(deg)),
                    ]
                )


def blend_submissions(
    *,
    base_csv: Path,
    candidate_csv: Path,
    out_csv: Path,
    nmax: int = 200,
    overlap_mode: OverlapMode = "kaggle",
    repair_mode: RepairMode = "none",
    repair_seed: int = 123,
    repair_max_puzzles: int = 0,
    repair_min_gain: float = 0.0,
    repair_n_min: int = 1,
    repair_n_max: int | None = None,
    tol: float = 1e-12,
) -> tuple[Path, dict]:
    """Blend two submissions per puzzle, keeping only safe improvements.

    Args:
        base_csv: Overlap-free baseline submission (in `overlap_mode`).
        candidate_csv: Candidate submission to cherry-pick from.
        out_csv: Output path.
        nmax: Max puzzle size to blend (default 200).
        overlap_mode: Overlap predicate to enforce.
        repair_mode: If `finalize`, attempts bounded repairs for overlapping
            candidate puzzles that appear promising.
        repair_seed: Seed base for repairs.
        repair_max_puzzles: Max number of overlapping puzzles to try repairing.
        repair_min_gain: Minimum estimated term gain (delta of `s^2/n`) to
            consider repairing.
        repair_n_min: Only repair puzzles with `n >= repair_n_min`.
        repair_n_max: Only repair puzzles with `n <= repair_n_max` (or None).
        tol: Strict improvement tolerance.

    Returns:
        (out_csv, meta) where meta includes counts of used/repair attempts.
    """
    base_csv = Path(base_csv)
    candidate_csv = Path(candidate_csv)
    out_csv = Path(out_csv)

    nmax = int(nmax)
    if nmax <= 0 or nmax > 200:
        raise ValueError("nmax must be in [1, 200]")

    base = load_submission(base_csv, nmax=nmax)
    cand = load_submission(candidate_csv, nmax=nmax)
    missing_base = [n for n in range(1, nmax + 1) if n not in base or base[n].shape != (n, 3)]
    missing_cand = [n for n in range(1, nmax + 1) if n not in cand or cand[n].shape != (n, 3)]
    if missing_base:
        raise ValueError(f"Base CSV missing/invalid puzzles: {missing_base[:5]}{'...' if len(missing_base) > 5 else ''}")
    if missing_cand:
        raise ValueError(
            f"Candidate CSV missing/invalid puzzles: {missing_cand[:5]}{'...' if len(missing_cand) > 5 else ''}"
        )

    points = np.array(TREE_POINTS, dtype=float)

    chosen: dict[int, np.ndarray] = {}
    base_q_by_n: dict[int, np.ndarray] = {}
    s_base_by_n: dict[int, float] = {}

    used_candidate = 0
    used_candidate_direct = 0
    candidate_overlaps = 0
    candidate_worse_or_equal = 0

    repair_attempted = 0
    repair_used = 0
    repair_failed = 0
    repair_not_improved = 0

    # Store (gain_est, n, cand_q) for promising overlapping puzzles.
    to_repair: list[tuple[float, int, np.ndarray]] = []

    for n in range(1, nmax + 1):
        base_q = _canonicalize_poses(base[n])
        cand_q = _canonicalize_poses(cand[n])

        if first_overlap_pair(points, base_q, mode=overlap_mode) is not None:
            raise ValueError(f"Base CSV has overlap in puzzle {n} (mode={overlap_mode}).")

        s_base = float(packing_score(points, base_q))
        s_cand = float(packing_score(points, cand_q))

        base_q_by_n[n] = base_q
        s_base_by_n[n] = s_base

        if s_cand + float(tol) >= s_base:
            chosen[n] = base_q
            candidate_worse_or_equal += 1
            continue

        if first_overlap_pair(points, cand_q, mode=overlap_mode) is None:
            chosen[n] = cand_q
            used_candidate += 1
            used_candidate_direct += 1
            continue

        # Candidate is better but overlaps; keep base for now and optionally repair later.
        chosen[n] = base_q
        candidate_overlaps += 1
        gain_est = (s_base * s_base - s_cand * s_cand) / float(n)
        to_repair.append((float(gain_est), int(n), cand_q))

    if repair_mode == "finalize" and int(repair_max_puzzles) > 0 and to_repair:
        repair_n_max_eff = int(repair_n_max) if repair_n_max is not None else None
        filtered: list[tuple[float, int, np.ndarray]] = []
        for gain_est, n, cand_q in to_repair:
            if n < int(repair_n_min):
                continue
            if repair_n_max_eff is not None and n > repair_n_max_eff:
                continue
            if float(gain_est) + 1e-15 < float(repair_min_gain):
                continue
            filtered.append((float(gain_est), int(n), cand_q))

        filtered.sort(key=lambda t: t[0], reverse=True)
        for gain_est, n, cand_q in filtered[: int(repair_max_puzzles)]:
            repair_attempted += 1
            fixed = _finalize_puzzle(
                points,
                cand_q,
                seed=int(repair_seed) + 1_000_003 * int(n),
                puzzle_n=int(n),
                overlap_mode=overlap_mode,
            )
            fixed_q = _canonicalize_poses(fixed)
            if first_overlap_pair(points, fixed_q, mode=overlap_mode) is not None:
                repair_failed += 1
                continue

            s_fixed = float(packing_score(points, fixed_q))
            s_base = float(s_base_by_n[int(n)])
            if s_fixed + float(tol) < s_base:
                chosen[int(n)] = fixed_q
                used_candidate += 1
                repair_used += 1
            else:
                repair_not_improved += 1

    _write_submission(out_csv, chosen, nmax=nmax)
    meta = {
        "nmax": int(nmax),
        "overlap_mode": str(overlap_mode),
        "base_csv": str(base_csv),
        "candidate_csv": str(candidate_csv),
        "out_csv": str(out_csv),
        "used_candidate": int(used_candidate),
        "used_candidate_direct": int(used_candidate_direct),
        "candidate_overlaps": int(candidate_overlaps),
        "candidate_worse_or_equal": int(candidate_worse_or_equal),
        "repair_mode": str(repair_mode),
        "repair_seed": int(repair_seed),
        "repair_max_puzzles": int(repair_max_puzzles),
        "repair_min_gain": float(repair_min_gain),
        "repair_n_min": int(repair_n_min),
        "repair_n_max": int(repair_n_max) if repair_n_max is not None else None,
        "repair_attempted": int(repair_attempted),
        "repair_used": int(repair_used),
        "repair_failed": int(repair_failed),
        "repair_not_improved": int(repair_not_improved),
    }
    return out_csv, meta

