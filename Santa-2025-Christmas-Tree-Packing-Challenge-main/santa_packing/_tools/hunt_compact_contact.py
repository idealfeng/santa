#!/usr/bin/env python3

"""Run many `bin/compact_contact` jobs and ensemble the best per puzzle `n`.

This tool is intended for massive local experiments:
- spawn many `compact_contact` runs with different seeds (parallel)
- parse per-`n` side lengths from each log (fast)
- build a strict per-`n` ensemble submission
- optionally apply Python subset-smoothing and/or C++ `post_opt` as a final polish
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from santa_packing.cli.improve_submission import _write_submission
from santa_packing.geom_np import packing_score
from santa_packing.scoring import first_overlap_pair, load_submission
from santa_packing.submission_format import fit_xy_in_bounds, quantize_for_submission
from santa_packing.tree_data import TREE_POINTS


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _parse_int_list(text: str) -> list[int]:
    raw = text.strip()
    if not raw:
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
            out.extend(range(start, end + step, step))
            continue
        if "-" in part and part.count("-") == 1 and part[0] != "-":
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
            continue
        out.append(int(part))
    return out


_POINTS_NP = np.array(TREE_POINTS, dtype=float)


@dataclass(frozen=True)
class Candidate:
    seed: int
    csv_path: Path
    log_path: Path
    score: float
    side: list[float]  # 1-indexed: side[n]
    elapsed_s: float


def _run_compact_contact(
    *,
    root: Path,
    base: Path,
    out_dir: Path,
    seed: int,
    nmax: int,
    target_range: tuple[int, int],
    passes: int,
    attempts_per_pass: int,
    patience: int,
    boundary_topk: int,
    push_bisect_iters: int,
    push_max_step_frac: float,
    plateau_eps: float,
    diag_frac: float,
    diag_rand: float,
    center_bias: float,
    interior_prob: float,
    shake_pos: float,
    shake_rot_deg: float,
    shake_prob: float,
    quantize_decimals: int,
) -> Candidate:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"cc_seed{seed}.csv"
    log_path = out_dir / f"cc_seed{seed}.log"

    cmd = [
        str((root / "bin" / "compact_contact").resolve()),
        "--base",
        str(base),
        "--out",
        str(out_csv),
        "--n-min",
        "1",
        "--n-max",
        str(nmax),
        "--target-range",
        f"{int(target_range[0])},{int(target_range[1])}",
        "--seed",
        str(int(seed)),
        "--passes",
        str(int(passes)),
        "--attempts-per-pass",
        str(int(attempts_per_pass)),
        "--patience",
        str(int(patience)),
        "--boundary-topk",
        str(int(boundary_topk)),
        "--push-bisect-iters",
        str(int(push_bisect_iters)),
        "--push-max-step-frac",
        str(float(push_max_step_frac)),
        "--plateau-eps",
        str(float(plateau_eps)),
        "--diag-frac",
        str(float(diag_frac)),
        "--diag-rand",
        str(float(diag_rand)),
        "--center-bias",
        str(float(center_bias)),
        "--interior-prob",
        str(float(interior_prob)),
        "--shake-pos",
        str(float(shake_pos)),
        "--shake-rot-deg",
        str(float(shake_rot_deg)),
        "--shake-prob",
        str(float(shake_prob)),
        "--quantize-decimals",
        str(int(quantize_decimals)),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t1 = time.time()
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"compact_contact failed (seed={seed}) rc={proc.returncode}")
    if not out_csv.is_file():
        raise RuntimeError(f"compact_contact did not write output: {out_csv}")

    puzzles = load_submission(out_csv, nmax=nmax)
    side = [0.0] * (nmax + 1)
    for n in range(1, nmax + 1):
        poses = puzzles.get(n)
        if poses is None or poses.shape != (n, 3):
            raise RuntimeError(f"compact_contact output missing puzzle {n} (seed={seed})")
        side[n] = packing_score(_POINTS_NP, poses)

    score = 0.0
    for n in range(1, nmax + 1):
        score += (float(side[n]) ** 2) / float(n)
    return Candidate(seed=seed, csv_path=out_csv, log_path=log_path, score=score, side=side, elapsed_s=t1 - t0)


def _ensemble_by_side(candidates: list[Candidate], *, nmax: int) -> tuple[dict[int, int], list[float]]:
    best_idx_by_n: dict[int, int] = {}
    best_side = [0.0] * (nmax + 1)
    for n in range(1, nmax + 1):
        best = None
        for idx, c in enumerate(candidates):
            s = float(c.side[n])
            if best is None or s < best[0]:
                best = (s, idx)
        assert best is not None
        s, idx = best
        best_idx_by_n[n] = idx
        best_side[n] = s
    return best_idx_by_n, best_side


def _write_ensemble_csv(
    *,
    out_csv: Path,
    candidates: list[Candidate],
    best_idx_by_n: dict[int, int],
    explains_csv: Path | None,
    nmax: int,
) -> None:
    # Load only the candidates that are actually used.
    used_idxs = sorted(set(best_idx_by_n.values()))
    puzzles_by_idx: dict[int, dict[int, np.ndarray]] = {}
    for idx in used_idxs:
        puzzles_by_idx[idx] = load_submission(candidates[idx].csv_path, nmax=nmax)

    ensemble: dict[int, np.ndarray] = {}
    for n in range(1, nmax + 1):
        idx = best_idx_by_n[n]
        poses = puzzles_by_idx[idx].get(n)
        if poses is None or poses.shape != (n, 3):
            raise RuntimeError(f"Candidate {candidates[idx].csv_path} missing puzzle {n}")
        ensemble[n] = poses

    _write_submission(out_csv, ensemble, nmax=nmax)

    if explains_csv is not None:
        explains_csv.parent.mkdir(parents=True, exist_ok=True)
        with explains_csv.open("w", encoding="utf-8") as f:
            f.write("puzzle,seed,score,side\n")
            for n in range(1, nmax + 1):
                idx = best_idx_by_n[n]
                c = candidates[idx]
                f.write(f"{n},{c.seed},{c.score:.12f},{c.side[n]:.17f}\n")


def _canonicalize_poses(poses: np.ndarray) -> np.ndarray:
    poses = np.array(poses, dtype=float, copy=True)
    if poses.shape[0] == 0:
        return poses
    poses[:, 2] = np.mod(poses[:, 2], 360.0)
    poses = fit_xy_in_bounds(poses)
    poses = quantize_for_submission(poses)
    return poses


def _blend_safe_per_puzzle(
    *,
    base_csv: Path,
    candidate_csv: Path,
    out_csv: Path,
    nmax: int,
    overlap_mode: str,
) -> tuple[Path, dict]:
    base = load_submission(base_csv, nmax=nmax)
    cand = load_submission(candidate_csv, nmax=nmax)

    chosen: dict[int, np.ndarray] = {}
    used_candidate = 0
    candidate_overlaps = 0
    candidate_worse_or_equal = 0

    for n in range(1, nmax + 1):
        base_poses = base.get(n)
        cand_poses = cand.get(n)
        if base_poses is None or base_poses.shape != (n, 3):
            raise RuntimeError(f"Base CSV missing puzzle {n} or wrong shape.")
        if cand_poses is None or cand_poses.shape != (n, 3):
            raise RuntimeError(f"Candidate CSV missing puzzle {n} or wrong shape.")

        base_q = _canonicalize_poses(base_poses)
        cand_q = _canonicalize_poses(cand_poses)

        if first_overlap_pair(_POINTS_NP, base_q, mode=overlap_mode) is not None:
            raise RuntimeError(f"Base CSV has overlap in puzzle {n} (mode={overlap_mode}).")

        if first_overlap_pair(_POINTS_NP, cand_q, mode=overlap_mode) is not None:
            candidate_overlaps += 1
            chosen[n] = base_q
            continue

        s_base = float(packing_score(_POINTS_NP, base_q))
        s_cand = float(packing_score(_POINTS_NP, cand_q))
        if s_cand + 1e-12 < s_base:
            chosen[n] = cand_q
            used_candidate += 1
        else:
            chosen[n] = base_q
            candidate_worse_or_equal += 1

    _write_submission(out_csv, chosen, nmax=nmax)
    meta = {
        "used_candidate": int(used_candidate),
        "candidate_overlaps": int(candidate_overlaps),
        "candidate_worse_or_equal": int(candidate_worse_or_equal),
        "nmax": int(nmax),
        "overlap_mode": str(overlap_mode),
        "base_csv": str(base_csv),
        "candidate_csv": str(candidate_csv),
        "out_csv": str(out_csv),
    }
    return out_csv, meta


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="Run many compact_contact seeds + strict per-n ensemble.")
    ap.add_argument("--base", type=Path, required=True, help="Base submission.csv to start from.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory to store candidates and ensemble.")
    ap.add_argument("--seeds", type=str, required=True, help="Seed list like '1..32' or '1,2,3'.")
    ap.add_argument("--jobs", type=int, default=16, help="Parallel jobs (default 16).")
    ap.add_argument("--nmax", type=int, default=200, help="Max n (default 200).")
    ap.add_argument("--target-range", type=str, default="1,200", help="Target n range 'a,b' (default 1,200).")

    ap.add_argument("--passes", type=int, default=300)
    ap.add_argument("--attempts-per-pass", type=int, default=150)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--boundary-topk", type=int, default=24)
    ap.add_argument("--push-bisect-iters", type=int, default=12)
    ap.add_argument("--push-max-step-frac", type=float, default=0.9)
    ap.add_argument("--plateau-eps", type=float, default=0.0)
    ap.add_argument("--diag-frac", type=float, default=0.3)
    ap.add_argument("--diag-rand", type=float, default=0.25)
    ap.add_argument("--center-bias", type=float, default=0.25)
    ap.add_argument("--interior-prob", type=float, default=0.15)
    ap.add_argument("--shake-pos", type=float, default=0.02)
    ap.add_argument("--shake-rot-deg", type=float, default=8.0)
    ap.add_argument("--shake-prob", type=float, default=0.25)
    ap.add_argument("--quantize-decimals", type=int, default=11)

    ap.add_argument("--ensemble-out", type=Path, default=None, help="Ensemble CSV path (default <out-dir>/ensemble.csv).")
    ap.add_argument("--choices-out", type=Path, default=None, help="Write per-n choices CSV (optional).")

    ap.add_argument("--smooth-window", type=int, default=0, help="If >0, run improve_submission smoothing window.")
    ap.add_argument("--post-opt", action="store_true", help="If set, run bin/post_opt after ensemble/smoothing.")
    ap.add_argument("--post-iters", type=int, default=4000)
    ap.add_argument("--post-restarts", type=int, default=4)
    ap.add_argument("--post-seed", type=int, default=1)
    ap.add_argument("--post-threads", type=int, default=0)
    ap.add_argument(
        "--post-overlap-mode",
        type=str,
        default="strict",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap mode enforced after post_opt (default: strict).",
    )

    ns = ap.parse_args(argv)

    root = Path.cwd().resolve()
    if not ns.base.is_file():
        raise SystemExit(f"--base not found: {ns.base}")
    out_dir = ns.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        a_str, b_str = str(ns.target_range).split(",", 1)
        target_range = (int(a_str), int(b_str))
    except Exception:
        raise SystemExit("--target-range must be 'a,b'") from None
    if target_range[0] < 1 or target_range[1] < target_range[0] or target_range[1] > int(ns.nmax):
        raise SystemExit("--target-range must satisfy 1 <= a <= b <= nmax")

    seeds = _parse_int_list(ns.seeds)
    if not seeds:
        raise SystemExit("--seeds produced an empty list")

    _eprint(f"Running compact_contact: seeds={len(seeds)} jobs={ns.jobs} nmax={ns.nmax} target={target_range[0]}..{target_range[1]}")

    candidates: list[Candidate] = []
    failures: list[tuple[int, str]] = []

    def _one(seed: int) -> Candidate:
        return _run_compact_contact(
            root=root,
            base=ns.base,
            out_dir=out_dir,
            seed=int(seed),
            nmax=int(ns.nmax),
            target_range=target_range,
            passes=int(ns.passes),
            attempts_per_pass=int(ns.attempts_per_pass),
            patience=int(ns.patience),
            boundary_topk=int(ns.boundary_topk),
            push_bisect_iters=int(ns.push_bisect_iters),
            push_max_step_frac=float(ns.push_max_step_frac),
            plateau_eps=float(ns.plateau_eps),
            diag_frac=float(ns.diag_frac),
            diag_rand=float(ns.diag_rand),
            center_bias=float(ns.center_bias),
            interior_prob=float(ns.interior_prob),
            shake_pos=float(ns.shake_pos),
            shake_rot_deg=float(ns.shake_rot_deg),
            shake_prob=float(ns.shake_prob),
            quantize_decimals=int(ns.quantize_decimals),
        )

    with ThreadPoolExecutor(max_workers=max(1, int(ns.jobs))) as ex:
        futs = {ex.submit(_one, seed): int(seed) for seed in seeds}
        for fut in as_completed(futs):
            seed = futs[fut]
            try:
                c = fut.result()
            except Exception as e:
                failures.append((seed, str(e)))
                _eprint(f"[seed={seed}] failed: {e}")
                continue
            candidates.append(c)
            _eprint(f"[seed={seed}] score={c.score:.12f} time={c.elapsed_s:.1f}s")

    if not candidates:
        raise SystemExit("All runs failed")
    if failures:
        _eprint(f"Failures: {len(failures)}/{len(seeds)} (continuing)")

    best_single = min(candidates, key=lambda c: c.score)
    _eprint(f"Best single: seed={best_single.seed} score={best_single.score:.12f} csv={best_single.csv_path}")

    best_idx_by_n, _ = _ensemble_by_side(candidates, nmax=int(ns.nmax))
    ensemble_out = ns.ensemble_out or (out_dir / "ensemble.csv")
    choices_out = ns.choices_out or (out_dir / "ensemble_choices.csv")
    _write_ensemble_csv(
        out_csv=ensemble_out,
        candidates=candidates,
        best_idx_by_n=best_idx_by_n,
        explains_csv=choices_out,
        nmax=int(ns.nmax),
    )

    # Fast score estimate (no overlap check): sum(side[n]^2 / n) from our selection.
    _, best_side = _ensemble_by_side(candidates, nmax=int(ns.nmax))
    ens_score = 0.0
    for n in range(1, int(ns.nmax) + 1):
        ens_score += (float(best_side[n]) ** 2) / float(n)
    _eprint(f"Ensemble score (no overlap check): {ens_score:.12f}")

    current_path = ensemble_out

    if int(ns.smooth_window) > 0:
        prev = current_path
        smooth_out = out_dir / f"ensemble_smooth{int(ns.smooth_window)}.csv"
        cmd = [
            sys.executable,
            "-m",
            "santa_packing.cli.improve_submission",
            str(current_path),
            "--out",
            str(smooth_out),
            "--smooth-window",
            str(int(ns.smooth_window)),
            "--overlap-mode",
            "strict",
        ]
        _eprint(f"Running smoothing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(root))
        from santa_packing.scoring import score_submission  # noqa: E402

        try:
            res = score_submission(smooth_out, nmax=int(ns.nmax), overlap_mode="strict")
            _eprint(f"Smoothed score (strict): {res.score:.12f}")
            current_path = smooth_out
        except Exception as exc:
            _eprint(f"[warn] Smoothing produced overlap; keeping previous ({prev}): {exc}")
            current_path = prev

    if bool(ns.post_opt):
        prev = current_path
        post_raw = out_dir / "ensemble_postopt_raw.csv"
        post_out = out_dir / "ensemble_postopt.csv"
        cmd = [
            str((root / "bin" / "post_opt").resolve()),
            "--input",
            str(current_path),
            "--output",
            str(post_raw),
            "--iters",
            str(int(ns.post_iters)),
            "--restarts",
            str(int(ns.post_restarts)),
            "--seed",
            str(int(ns.post_seed)),
            "--threads",
            str(int(ns.post_threads)),
        ]
        _eprint(f"Running post_opt: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(root))
        from santa_packing.scoring import score_submission  # noqa: E402

        # post_opt is not guaranteed to preserve feasibility. Enforce it by blending per puzzle:
        # keep post_opt only when it's overlap-free (in the requested mode) and better.
        try:
            blended, meta = _blend_safe_per_puzzle(
                base_csv=prev,
                candidate_csv=post_raw,
                out_csv=post_out,
                nmax=int(ns.nmax),
                overlap_mode=str(ns.post_overlap_mode),
            )
            _eprint(
                "Post-opt blend:",
                f"used_candidate={meta['used_candidate']}",
                f"candidate_overlaps={meta['candidate_overlaps']}",
                f"candidate_worse_or_equal={meta['candidate_worse_or_equal']}",
            )
            current_path = blended
        except Exception as exc:
            _eprint(f"[warn] Post-opt blend failed; keeping previous ({prev}): {exc}")
            current_path = prev

        try:
            res = score_submission(current_path, nmax=int(ns.nmax), overlap_mode=str(ns.post_overlap_mode))
            _eprint(f"Post-opt score ({ns.post_overlap_mode}, safe): {res.score:.12f}")
        except Exception as exc:
            _eprint(f"[warn] Post-opt output still overlaps ({ns.post_overlap_mode}): {exc}")
            res = score_submission(
                current_path,
                nmax=int(ns.nmax),
                check_overlap=False,
                overlap_mode=str(ns.post_overlap_mode),
            )
            _eprint(f"Post-opt score ({ns.post_overlap_mode}, no-overlap): {res.score:.12f}")

    print(str(current_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
