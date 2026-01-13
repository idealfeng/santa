#!/usr/bin/env python3

"""Run a portfolio of solvers and merge the best puzzles per `n`.

This script targets the older "independent puzzles" workflow (solve each `n`
separately) and merges multiple candidate submissions puzzle-by-puzzle.

Note: for the current Python pipeline (generate + multi-start + per-n ensemble),
prefer `python -m santa_packing._tools.sweep_ensemble` with `scripts/submission/portfolios/*.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if stdout_path is not None:
        stdout_path.write_text(proc.stdout, encoding="utf-8")
    if stderr_path is not None:
        stderr_path.write_text(proc.stderr, encoding="utf-8")

    if proc.returncode != 0:
        _eprint(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
        if proc.stdout.strip():
            _eprint("--- stdout ---")
            _eprint(proc.stdout.rstrip())
        if proc.stderr.strip():
            _eprint("--- stderr ---")
            _eprint(proc.stderr.rstrip())
        raise SystemExit(proc.returncode)

    return proc


def _run_json(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path | None = None,
) -> dict[str, Any]:
    proc = _run(cmd, cwd=cwd, stdout_path=stdout_path, stderr_path=stderr_path)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        _eprint(proc.stdout)
        raise SystemExit(f"Failed to parse JSON from: {' '.join(cmd)} ({e})")


def _ensure_bins(root: Path, *, no_build: bool, jobs: int | None) -> None:
    required = ["solve_all", "merge_submissions", "score_submission"]
    missing: list[str] = []
    for name in required:
        p = root / "bin" / name
        if not (p.is_file() and os.access(p, os.X_OK)):
            missing.append(name)

    if not missing:
        return
    if no_build:
        raise SystemExit(f"Missing binaries (build first): {', '.join(missing)}")

    build_dir = root / "build"
    _run(["cmake", "-S", ".", "-B", str(build_dir)], cwd=root)
    build_cmd = ["cmake", "--build", str(build_dir)]
    if jobs and jobs > 0:
        build_cmd += ["-j", str(jobs)]
    _run(build_cmd, cwd=root)


def _angles_csv(step_deg: int) -> str:
    if step_deg <= 0 or step_deg > 360:
        raise ValueError("step_deg must be in [1,360]")
    angles = list(range(0, 360, step_deg))
    if 0 not in angles:
        angles.insert(0, 0)
    return ",".join(str(a) for a in angles)


@dataclass(frozen=True)
class Recipe:
    name: str
    nmax: int
    solve_all_args: list[str]


def _default_recipes(args: argparse.Namespace) -> list[Recipe]:
    out: list[Recipe] = []

    if not args.no_0180:
        out.append(
            Recipe(
                name="angles_0_180",
                nmax=args.nmax,
                solve_all_args=[
                    "--init",
                    "bottom-left",
                    "--refine",
                    "none",
                    "--angles",
                    "0,180",
                    "--mode",
                    "cycle",
                    "--cycle",
                    "0,180",
                ],
            )
        )

    if not args.no_8:
        out.append(
            Recipe(
                name="angles_8x45",
                nmax=args.nmax,
                solve_all_args=[
                    "--init",
                    "bottom-left",
                    "--refine",
                    "none",
                    "--angles",
                    "0,45,90,135,180,225,270,315",
                ],
            )
        )

    if not args.no_dense:
        dense_nmax = min(args.dense_nmax, args.nmax)
        if dense_nmax > 0:
            out.append(
                Recipe(
                    name=f"dense_{args.dense_step}deg_n{dense_nmax}",
                    nmax=dense_nmax,
                    solve_all_args=[
                        "--init",
                        "bottom-left",
                        "--refine",
                        "none",
                        "--angles",
                        _angles_csv(args.dense_step),
                    ],
                )
            )

    if not out:
        raise SystemExit("No recipes selected. Use fewer --no-* flags.")
    return out


def main() -> int:
    """Execute the portfolio merge workflow and write the merged submission."""
    ap = argparse.ArgumentParser(
        description=(
            "Generate multiple solve_all submissions with different configs and merge puzzle-by-puzzle.\n"
            "Designed for 'independent puzzles' mode (each n solved separately)."
        )
    )
    ap.add_argument("--repo", type=Path, default=_repo_root_from_script(), help="Repo root (default: script parent)")
    ap.add_argument("--runs-dir", type=Path, default=None, help="Runs directory (default: <repo>/runs)")
    ap.add_argument("--tag", type=str, default="portfolio_merge", help="Run tag (used in runs/<tag>_<ts>/)")
    ap.add_argument("--nmax", type=int, default=200, help="Final nmax (default: 200)")
    ap.add_argument("--seed", type=int, default=1, help="Base seed passed to solve_all (default: 1)")
    ap.add_argument("--threads", type=int, default=0, help="Threads passed to solve_all (0=auto)")
    ap.add_argument("--jobs", type=int, default=0, help="Build parallelism for cmake --build (0=default)")
    ap.add_argument("--no-build", action="store_true", help="Do not build missing binaries")

    ap.add_argument("--no-0180", action="store_true", help="Skip the 0/180 recipe")
    ap.add_argument("--no-8", action="store_true", help="Skip the 8x45° recipe")
    ap.add_argument("--no-dense", action="store_true", help="Skip the dense-angles small-n recipe")
    ap.add_argument("--dense-nmax", type=int, default=30, help="nmax for dense recipe (default: 30)")
    ap.add_argument("--dense-step", type=int, default=5, help="Angle step in degrees for dense recipe (default: 5)")

    ap.add_argument("--out", type=Path, default=None, help="Optional: copy merged CSV to this path")
    ap.add_argument("--dry-run", action="store_true", help="Print planned commands and exit")
    ns = ap.parse_args()

    root = ns.repo.resolve()
    runs_dir = (ns.runs_dir or (root / "runs")).resolve()
    if ns.nmax <= 0 or ns.nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")
    if ns.dense_nmax < 0 or ns.dense_nmax > 200:
        raise SystemExit("--dense-nmax must be in [0,200]")

    recipes = _default_recipes(ns)
    stamp = _timestamp()
    run_dir = runs_dir / f"{ns.tag}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "tag": ns.tag,
        "timestamp": stamp,
        "nmax": ns.nmax,
        "seed": ns.seed,
        "threads": ns.threads,
        "recipes": [{"name": r.name, "nmax": r.nmax, "solve_all_args": r.solve_all_args} for r in recipes],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    solve_all_bin = root / "bin" / "solve_all"
    merge_bin = root / "bin" / "merge_submissions"
    score_bin = root / "bin" / "score_submission"

    if ns.dry_run:
        for r in recipes:
            out_recipe = run_dir / r.name
            submission = out_recipe / "submission.csv"
            solve_json = out_recipe / "solve_all.json"
            solve_cmd = [
                str(solve_all_bin),
                "--nmax",
                str(r.nmax),
                "--seed",
                str(ns.seed),
                "--out",
                str(submission),
                "--out-dir",
                str(out_recipe),
                "--out-json",
                str(solve_json),
                "--threads",
                str(ns.threads),
            ] + r.solve_all_args
            print(" ".join(solve_cmd))

        merged_dir = run_dir / "merged"
        merged_csv = merged_dir / "submission.csv"
        merge_json = merged_dir / "merge_report.json"
        merge_cmd = [
            str(merge_bin),
            "--out",
            str(merged_csv),
            "--out-json",
            str(merge_json),
            "--nmax",
            str(ns.nmax),
            "--allow-partial",
        ] + [str((run_dir / r.name / "submission.csv").resolve()) for r in recipes]
        print(" ".join(merge_cmd))
        return 0

    _ensure_bins(root, no_build=ns.no_build, jobs=(ns.jobs if ns.jobs > 0 else None))

    scores: dict[str, Any] = {}

    for r in recipes:
        out_recipe = run_dir / r.name
        out_recipe.mkdir(parents=True, exist_ok=True)

        submission = out_recipe / "submission.csv"
        solve_json = out_recipe / "solve_all.json"
        solve_log = out_recipe / "solve_all.log"

        solve_cmd = [
            str(solve_all_bin),
            "--nmax",
            str(r.nmax),
            "--seed",
            str(ns.seed),
            "--out",
            str(submission),
            "--out-dir",
            str(out_recipe),
            "--out-json",
            str(solve_json),
            "--threads",
            str(ns.threads),
        ] + r.solve_all_args
        _run(solve_cmd, cwd=root, stdout_path=out_recipe / "solve_all.stdout.json", stderr_path=solve_log)

        score_cmd = [
            str(score_bin),
            str(submission),
            "--nmax",
            str(ns.nmax),
            "--allow-partial",
            "--breakdown",
        ]
        score = _run_json(score_cmd, cwd=root, stdout_path=out_recipe / "score.json")
        scores[r.name] = score

    merged_dir = run_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = merged_dir / "submission.csv"
    merge_json = merged_dir / "merge_report.json"

    merge_cmd = [
        str(merge_bin),
        "--out",
        str(merged_csv),
        "--out-json",
        str(merge_json),
        "--nmax",
        str(ns.nmax),
        "--allow-partial",
    ] + [str((run_dir / r.name / "submission.csv").resolve()) for r in recipes]
    merge = _run_json(merge_cmd, cwd=root, stdout_path=merged_dir / "merge.json")

    merged_score = _run_json(
        [str(score_bin), str(merged_csv), "--nmax", str(ns.nmax), "--breakdown"],
        cwd=root,
        stdout_path=merged_dir / "score.json",
    )

    summary = {
        "run_dir": str(run_dir),
        "recipes": scores,
        "merge": merge,
        "merged_score": merged_score,
        "merged_csv": str(merged_csv),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))

    if ns.out is not None:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(merged_csv, ns.out)
        print(f"\nCopied merged CSV to: {ns.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
