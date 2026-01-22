#!/usr/bin/env python3

"""Targeted small-n improvement hunt (safe per-puzzle replace).

This tool tries to shave the leaderboard-critical small-n terms by running a
stochastic local search (LNS) *per puzzle n* starting from an existing strong
baseline `submission.csv`.

It is intentionally conservative:
- Only replaces puzzle `n` if the finalized (quantized + validated) candidate
  strictly improves `s_n` in the chosen `--overlap-mode`.
- Never touches puzzles outside `--n-min..--n-max`.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from santa_packing.cli.generate_submission import _finalize_puzzle
from santa_packing.cli.improve_submission import _write_submission
from santa_packing.geom_np import packing_score
from santa_packing.postopt_np import large_neighborhood_search
from santa_packing.scoring import OverlapMode, first_overlap_pair, load_submission, score_submission
from santa_packing.tree_data import TREE_POINTS


def _parse_int_list(text: str) -> list[int]:
    raw = text.strip()
    if not raw:
        return []
    if raw.lower() in {"none", "off", "false"}:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            a, b = part.split("..", 1)
            start = int(a)
            end = int(b)
            step = 1 if end >= start else -1
            out.extend(list(range(start, end + step, step)))
            continue
        if "-" in part and part.count("-") == 1 and part[0] != "-":
            a, b = part.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
            continue
        out.append(int(part))
    return out


def _canonicalize_start(poses: np.ndarray, *, seed: int, n: int, overlap_mode: OverlapMode) -> np.ndarray:
    points = np.array(TREE_POINTS, dtype=float)
    fixed = _finalize_puzzle(points, poses, seed=int(seed), puzzle_n=int(n), overlap_mode=str(overlap_mode))
    if first_overlap_pair(points, fixed, mode=str(overlap_mode)) is not None:
        raise ValueError(f"baseline overlap after finalize (n={n}, overlap_mode={overlap_mode})")
    return fixed


@dataclass(frozen=True)
class HuntConfig:
    passes: int
    destroy_frac: float
    destroy_k_min: int
    destroy_k_max: int
    destroy_mode: str
    candidates: int
    angle_samples: int
    pad_scale: float
    recreate_topk: int
    group_moves: int
    group_size: int
    group_trans_sigma: float
    group_rot_sigma: float
    t_start: float
    t_end: float


def _destroy_k(n: int, cfg: HuntConfig) -> int:
    k = int(round(float(cfg.destroy_frac) * float(n)))
    k = max(int(cfg.destroy_k_min), k)
    k = min(int(cfg.destroy_k_max), k)
    k = min(k, max(1, n - 1))
    return k


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hunt small-n improvements via per-n LNS + safe replace.")
    ap.add_argument("--base", type=Path, default=Path("submission.csv"), help="Baseline submission.csv")
    ap.add_argument("--out", type=Path, default=Path("submission_hunt_smalln.csv"), help="Output submission.csv")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle size present in the CSV")
    ap.add_argument("--n-min", type=int, default=2, help="Min puzzle n to hunt (default: 2)")
    ap.add_argument("--n-max", type=int, default=30, help="Max puzzle n to hunt (default: 30)")
    ap.add_argument("--seeds", type=str, default="1..10", help="Seed list/range (e.g. 1..50 or 1,2,3)")
    ap.add_argument(
        "--overlap-mode",
        type=str,
        default="kaggle",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap predicate used for candidate validation/finalization.",
    )
    ap.add_argument("--tol", type=float, default=1e-12, help="Strict improvement tolerance for s_n")

    ap.add_argument("--passes", type=int, default=30, help="LNS passes per seed")
    ap.add_argument("--destroy-frac", type=float, default=0.35, help="destroy_k = round(destroy_frac * n)")
    ap.add_argument("--destroy-k-min", type=int, default=2)
    ap.add_argument("--destroy-k-max", type=int, default=12)
    ap.add_argument("--destroy-mode", type=str, default="alns", choices=["mixed", "boundary", "random", "cluster", "alns"])
    ap.add_argument("--candidates", type=int, default=64, help="Candidate centers per reinsertion")
    ap.add_argument("--angle-samples", type=int, default=8, help="Angle samples per reinsertion")
    ap.add_argument("--pad-scale", type=float, default=2.0, help="Padding for center sampling bounds (in radii)")
    ap.add_argument(
        "--recreate-topk",
        type=int,
        default=1,
        help="Randomly pick among the top-K reinsertion placements (default: 1 == greedy).",
    )
    ap.add_argument("--group-moves", type=int, default=1, help="Group-rotation moves per pass (0 disables)")
    ap.add_argument("--group-size", type=int, default=3)
    ap.add_argument("--group-trans-sigma", type=float, default=0.03)
    ap.add_argument("--group-rot-sigma", type=float, default=25.0)
    ap.add_argument("--t-start", type=float, default=0.02, help="SA temperature start (0 disables worse acceptance)")
    ap.add_argument("--t-end", type=float, default=0.005, help="SA temperature end (0 disables worse acceptance)")

    ap.add_argument("--resume", action="store_true", help="If --out exists, resume from it instead of --base")
    ap.add_argument("--checkpoint", action="store_true", help="Write --out after each completed n (resumable).")
    ap.add_argument("--log-every", type=int, default=1, help="Print progress every K seeds (default: 1).")
    ap.add_argument(
        "--time-limit-s",
        type=float,
        default=0.0,
        help="Optional total wall-clock time limit in seconds (0 disables).",
    )
    ns = ap.parse_args(argv)

    nmax = int(ns.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")

    overlap_mode: OverlapMode = str(ns.overlap_mode)  # type: ignore[assignment]
    seeds = _parse_int_list(str(ns.seeds))
    if not seeds:
        raise SystemExit("Empty --seeds")

    n_min = int(ns.n_min)
    n_max = int(ns.n_max)
    if n_min < 1 or n_min > nmax:
        raise SystemExit("--n-min must be in [1,nmax]")
    if n_max < n_min or n_max > nmax:
        raise SystemExit("--n-max must be in [n-min,nmax]")

    base_path = Path(ns.base).resolve()
    out_path = Path(ns.out).resolve()
    start_path = out_path if bool(ns.resume) and out_path.is_file() else base_path

    puzzles = load_submission(start_path, nmax=nmax)
    missing = [n for n in range(1, nmax + 1) if n not in puzzles or puzzles[n].shape != (n, 3)]
    if missing:
        raise SystemExit(f"submission missing/invalid puzzles: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    points = np.array(TREE_POINTS, dtype=float)
    best: dict[int, np.ndarray] = {n: _canonicalize_start(puzzles[n], seed=991 + n, n=n, overlap_mode=overlap_mode) for n in range(1, nmax + 1)}
    best_s: dict[int, float] = {n: float(packing_score(points, best[n])) for n in range(1, nmax + 1)}

    base_score = score_submission(start_path, nmax=nmax, check_overlap=True, overlap_mode=overlap_mode, require_complete=True).score
    print(f"start_csv: {start_path}")
    print(f"start_score: {base_score:.12f}")

    cfg = HuntConfig(
        passes=int(ns.passes),
        destroy_frac=float(ns.destroy_frac),
        destroy_k_min=int(ns.destroy_k_min),
        destroy_k_max=int(ns.destroy_k_max),
        destroy_mode=str(ns.destroy_mode),
        candidates=int(ns.candidates),
        angle_samples=int(ns.angle_samples),
        pad_scale=float(ns.pad_scale),
        recreate_topk=int(ns.recreate_topk),
        group_moves=int(ns.group_moves),
        group_size=int(ns.group_size),
        group_trans_sigma=float(ns.group_trans_sigma),
        group_rot_sigma=float(ns.group_rot_sigma),
        t_start=float(ns.t_start),
        t_end=float(ns.t_end),
    )

    improved = 0
    t0 = time.time()

    for n in range(n_min, n_max + 1):
        if float(ns.time_limit_s) > 0.0 and (time.time() - t0) >= float(ns.time_limit_s):
            print(f"[time-limit] stopping before n={n} (elapsed_s={time.time() - t0:.1f})", flush=True)
            break
        s0 = float(best_s[n])
        kk0 = _destroy_k(int(n), cfg)
        print(
            f"[n={n:03d}] start s={s0:.12f} destroy_k={kk0} seeds={len(seeds)} passes={cfg.passes}",
            flush=True,
        )
        for run_idx, seed in enumerate(seeds, start=1):
            if float(ns.time_limit_s) > 0.0 and (time.time() - t0) >= float(ns.time_limit_s):
                print(f"[time-limit] stopping during n={n} (elapsed_s={time.time() - t0:.1f})", flush=True)
                break
            kk = _destroy_k(int(n), cfg)
            try:
                cand = large_neighborhood_search(
                    points,
                    best[n],
                    seed=int(seed) + 1_000_003 * int(n),
                    passes=int(cfg.passes),
                    destroy_k=int(kk),
                    destroy_mode=str(cfg.destroy_mode),
                    candidates=int(cfg.candidates),
                    angle_samples=int(cfg.angle_samples),
                    pad_scale=float(cfg.pad_scale),
                    recreate_topk=int(cfg.recreate_topk),
                    group_moves=int(cfg.group_moves),
                    group_size=int(cfg.group_size),
                    group_trans_sigma=float(cfg.group_trans_sigma),
                    group_rot_sigma=float(cfg.group_rot_sigma),
                    t_start=float(cfg.t_start),
                    t_end=float(cfg.t_end),
                    overlap_mode=overlap_mode,
                )
            except Exception as exc:
                print(f"[n={n:03d}] seed={seed} LNS failed: {exc}", flush=True)
                continue

            cand = _finalize_puzzle(
                points,
                cand,
                seed=int(seed) + 13_000 + 1_000_003 * int(n),
                puzzle_n=int(n),
                overlap_mode=overlap_mode,
            )
            if first_overlap_pair(points, cand, mode=overlap_mode) is not None:
                continue

            s = float(packing_score(points, cand))
            if s + float(ns.tol) < float(best_s[n]):
                best[n] = np.array(cand, dtype=float, copy=True)
                best_s[n] = float(s)
                improved += 1
                dt = time.time() - t0
                print(
                    f"[n={n:03d}] seed={seed} run={run_idx:03d}/{len(seeds):03d} "
                    f"s {s0:.12f} -> {s:.12f} (Δ={s0 - s:.6g}) improved_total={improved} elapsed_s={dt:.1f}",
                    flush=True,
                )
                s0 = float(s)
            elif int(ns.log_every) > 0 and (run_idx % int(ns.log_every) == 0 or run_idx == len(seeds)):
                dt = time.time() - t0
                print(
                    f"[n={n:03d}] seed={seed} run={run_idx:03d}/{len(seeds):03d} best_s={float(best_s[n]):.12f} elapsed_s={dt:.1f}",
                    flush=True,
                )

        if bool(ns.checkpoint):
            _write_submission(out_path, best, nmax=nmax)
            dt = time.time() - t0
            print(f"[checkpoint] wrote {out_path} after n={n:03d} elapsed_s={dt:.1f}", flush=True)

    _write_submission(out_path, best, nmax=nmax)
    res = score_submission(out_path, nmax=nmax, check_overlap=True, overlap_mode=overlap_mode, require_complete=True)
    print(f"wrote: {out_path}")
    print(f"score: {res.score:.12f}")
    print(f"delta: {base_score - res.score:+.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
