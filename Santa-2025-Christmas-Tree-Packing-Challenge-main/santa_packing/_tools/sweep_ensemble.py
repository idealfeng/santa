#!/usr/bin/env python3

"""Tool to sweep multiple generation recipes and ensemble per puzzle size.

This tool runs `generate_submission` for multiple seeds/recipes and then
selects the best candidate per `n` (minimizing `s_n`).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from santa_packing.config import config_to_argv, default_config_path
from santa_packing.constants import EPS
from santa_packing.scoring import first_overlap_pair, load_submission
from santa_packing.tree_data import TREE_POINTS


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _repo_root_from_script() -> Path:
    # Prefer current working directory (works for both editable installs and packaged runs).
    return Path.cwd().resolve()


def _safe_name(text: str) -> str:
    out = []
    for ch in text.strip():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "recipe"


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
            start = int(a)
            end = int(b)
            out.extend(list(range(start, end + 1)))
            continue
        out.append(int(part))
    return out


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    timeout_s: float | None = None,
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
        timeout=timeout_s,
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
        raise RuntimeError(f"Command failed ({proc.returncode})")

    return proc


@dataclass(frozen=True)
class Recipe:
    """One generation recipe passed to `generate_submission`.

    Attributes:
        name: Human-readable identifier used in logs and output folders.
        args: List of CLI tokens (excluding `--out/--seed/--nmax`, handled by the sweep).
    """

    name: str
    args: list[str]


def _parse_recipe(text: str) -> Recipe:
    if ":" in text:
        name, args_str = text.split(":", 1)
    elif "=" in text:
        name, args_str = text.split("=", 1)
    else:
        raise ValueError("Recipe must be NAME:ARGS or NAME=ARGS")
    name = name.strip()
    args = shlex.split(args_str.strip())
    if not name:
        raise ValueError("Recipe name is empty")
    forbidden = {"--out", "--seed", "--nmax"}
    if any(a in forbidden for a in args):
        raise ValueError("Recipe args must not include --out/--seed/--nmax (handled by sweep script)")
    return Recipe(name=name, args=args)


def _load_recipes_json(path: Path) -> list[Recipe]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.get("recipes")
    else:
        items = data
    if not isinstance(items, list):
        raise ValueError("recipes json must be a list or {'recipes':[...]} object")
    out: list[Recipe] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each recipe must be an object")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("recipe missing 'name'")
        args = item.get("args", [])
        if isinstance(args, str):
            args_list = shlex.split(args)
        elif isinstance(args, list):
            args_list = [str(x) for x in args]
        else:
            raise ValueError("recipe 'args' must be string or list")
        forbidden = {"--out", "--seed", "--nmax"}
        if any(a in forbidden for a in args_list):
            raise ValueError(f"recipe {name} includes forbidden flags (--out/--seed/--nmax)")
        out.append(Recipe(name=name, args=args_list))
    if not out:
        raise ValueError("no recipes found in json")
    return out


@dataclass
class Candidate:
    """A completed (recipe, seed) run loaded from a generated `submission.csv`.

    Attributes:
        cid: Unique candidate id used in filenames/logs.
        recipe: Recipe name.
        seed: Seed used for generation.
        csv_path: Path to the candidate CSV.
        time_s: Wall-clock generation time in seconds.
        s: Array of per-puzzle `s_n` values with shape `(nmax+1,)` (index 0 unused).
        puzzles: Mapping `n -> poses` (each pose array shaped `(n, 3)`).
    """

    cid: str
    recipe: str
    seed: int
    csv_path: Path
    time_s: float
    s: np.ndarray  # shape (nmax+1,)
    puzzles: dict[int, np.ndarray]  # n -> (n,3)


def _prefix_total(s: np.ndarray, *, nmax: int) -> float:
    total = 0.0
    for n in range(1, nmax + 1):
        v = float(s[n])
        total += (v * v) / n
    return total


def _compute_s(points: np.ndarray, puzzles: dict[int, np.ndarray], *, nmax: int) -> np.ndarray:
    from santa_packing.geom_np import packing_score  # noqa: E402

    s = np.full((nmax + 1,), np.nan, dtype=float)
    s[0] = 0.0
    for n in range(1, nmax + 1):
        poses = puzzles.get(n)
        if poses is None or poses.shape[0] != n:
            raise ValueError(f"Candidate missing puzzle {n} or wrong size")
        s[n] = float(packing_score(points, poses))
    return s


def _write_per_n_csv(path: Path, candidate: Candidate, *, nmax: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["puzzle", "s"])
        for n in range(1, nmax + 1):
            w.writerow([n, f"{candidate.s[n]:.17f}"])


def _write_submission(path: Path, puzzles: dict[int, np.ndarray], *, nmax: int) -> None:
    from santa_packing.submission_format import (  # noqa: E402
        fit_xy_in_bounds,
        format_submission_value,
        quantize_for_submission,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "deg"])
        for n in range(1, nmax + 1):
            poses = puzzles.get(n)
            if poses is None or poses.shape[0] != n:
                raise ValueError(f"Missing puzzle {n}")
            poses = np.array(poses, dtype=float, copy=True)
            poses[:, 2] = np.mod(poses[:, 2], 360.0)
            poses = fit_xy_in_bounds(poses)
            poses = quantize_for_submission(poses)
            for i, (x, y, deg) in enumerate(poses):
                w.writerow(
                    [
                        f"{n:03d}_{i}",
                        format_submission_value(float(x)),
                        format_submission_value(float(y)),
                        format_submission_value(float(deg)),
                    ]
                )


def main(argv: list[str] | None = None) -> int:
    """Run a sweep and write an ensembled `submission.csv`."""
    argv = list(sys.argv[1:] if argv is None else argv)

    cfg_default = default_config_path("ensemble.json")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre.add_argument("--no-config", action="store_true")
    pre_args, _ = pre.parse_known_args(argv)
    if pre_args.no_config and pre_args.config is not None:
        raise SystemExit("Use either --config or --no-config, not both.")
    config_path = None if pre_args.no_config else (pre_args.config or cfg_default)
    config_args = (
        config_to_argv(config_path, section_keys=("sweep_ensemble", "ensemble")) if config_path is not None else []
    )

    ap = argparse.ArgumentParser(
        description=(
            "Run multiple generate_submission configurations (multi-start) and ensemble per puzzle n.\n"
            "This is the Python/JAX replacement for old C++ portfolio merge."
        )
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="JSON/YAML config file with defaults (defaults to configs/ensemble.json when present).",
    )
    ap.add_argument("--no-config", action="store_true", help="Disable loading the default config (if any).")
    ap.add_argument("--repo", type=Path, default=_repo_root_from_script(), help="Repo root (default: auto-detected)")
    ap.add_argument("--runs-dir", type=Path, default=None, help="Runs directory (default: <repo>/runs)")
    ap.add_argument("--tag", type=str, default="sweep_ensemble", help="Run tag (runs/<tag>_<ts>/)")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default: 200)")
    ap.add_argument("--seeds", type=str, default="1", help="Comma list or ranges (e.g. 1,2,3 or 1..5)")
    ap.add_argument("--recipe", action="append", default=[], help="Recipe: NAME:ARGS passed to generate_submission.py")
    ap.add_argument("--recipes-json", type=Path, default=None, help="JSON file with recipes [{name,args}]")
    ap.add_argument("--timeout-s", type=float, default=None, help="Timeout per candidate run (seconds)")
    ap.add_argument("--jobs", type=int, default=1, help="Concurrent candidate runs (default: 1)")
    ap.add_argument("--reuse", action="store_true", help="Reuse existing candidate CSVs if present")
    ap.add_argument("--keep-going", action="store_true", help="Skip failed candidates instead of aborting")
    ap.add_argument("--overlap-check", type=str, default="selected", choices=["selected", "none"])
    ap.add_argument("--out", type=Path, default=None, help="Optional: copy ensemble submission.csv to this path")
    ap.add_argument("--dry-run", action="store_true", help="Print planned commands and exit")
    ns = ap.parse_args(config_args + argv)

    root = ns.repo.resolve()
    runs_dir = (ns.runs_dir or (root / "runs")).resolve()

    if ns.nmax <= 0 or ns.nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")

    seeds = _parse_int_list(ns.seeds)
    if not seeds:
        raise SystemExit("Empty --seeds")

    recipes: list[Recipe] = []
    if ns.recipes_json is not None:
        recipes.extend(_load_recipes_json(ns.recipes_json))
    for r in ns.recipe:
        recipes.append(_parse_recipe(r))
    if not recipes:
        recipes = [Recipe(name="default", args=[])]

    stamp = _timestamp()
    run_dir = runs_dir / f"{_safe_name(ns.tag)}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if ns.config is not None:
        (run_dir / f"config{ns.config.suffix}").write_text(ns.config.read_text(encoding="utf-8"), encoding="utf-8")

    meta: dict[str, Any] = {
        "tag": ns.tag,
        "timestamp": stamp,
        "nmax": ns.nmax,
        "seeds": seeds,
        "jobs": ns.jobs,
        "recipes": [{"name": r.name, "args": r.args} for r in recipes],
        "generator": "python -m santa_packing.cli.generate_submission",
        "python": sys.version,
        "overlap_check": ns.overlap_check,
        "config": str(ns.config) if ns.config is not None else None,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    points = np.array(TREE_POINTS, dtype=float)

    planned: list[tuple[str, str, int, list[str], Path]] = []
    for recipe in recipes:
        for seed in seeds:
            safe_recipe = _safe_name(recipe.name)
            cid = f"{safe_recipe}_seed{seed}"
            out_dir = run_dir / "candidates" / cid
            out_csv = out_dir / "submission.csv"
            cmd = [
                sys.executable,
                "-m",
                "santa_packing.cli.generate_submission",
                "--out",
                str(out_csv),
                "--nmax",
                str(ns.nmax),
                "--seed",
                str(seed),
            ] + recipe.args
            planned.append((cid, safe_recipe, seed, cmd, out_csv))

    if ns.dry_run:
        for cid, _recipe_name, _seed, cmd, _out_csv in planned:
            print(f"[{cid}] {' '.join(cmd)}")
        return 0

    candidates: list[Candidate] = []
    jobs = int(ns.jobs)
    if jobs <= 0:
        jobs = max(1, int(os.cpu_count() or 1))

    def run_one(item: tuple[str, str, int, list[str], Path]) -> Candidate:
        cid, recipe_name, seed, cmd, out_csv = item
        out_dir = out_csv.parent
        stdout_path = out_dir / "stdout.log"
        stderr_path = out_dir / "stderr.log"

        t0 = time.time()
        if not (ns.reuse and out_csv.is_file()):
            out_dir.mkdir(parents=True, exist_ok=True)
            _run(cmd, cwd=root, stdout_path=stdout_path, stderr_path=stderr_path, timeout_s=ns.timeout_s)
        t1 = time.time()
        puzzles = load_submission(out_csv, nmax=ns.nmax)
        s = _compute_s(points, puzzles, nmax=ns.nmax)
        candidate = Candidate(
            cid=cid,
            recipe=recipe_name,
            seed=seed,
            csv_path=out_csv,
            time_s=t1 - t0,
            s=s,
            puzzles=puzzles,
        )
        _write_per_n_csv(out_dir / "per_n.csv", candidate, nmax=ns.nmax)
        return candidate

    if jobs == 1:
        for item in planned:
            cid = item[0]
            t0 = time.time()
            try:
                candidates.append(run_one(item))
            except Exception as e:
                t1 = time.time()
                _eprint(f"[{cid}] failed after {t1 - t0:.1f}s: {e}")
                if not ns.keep_going:
                    raise
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            fut_to_item = {ex.submit(run_one, item): item for item in planned}
            for fut in as_completed(fut_to_item):
                cid = fut_to_item[fut][0]
                try:
                    candidates.append(fut.result())
                except Exception as e:
                    _eprint(f"[{cid}] failed: {e}")
                    if not ns.keep_going:
                        raise

    if not candidates:
        raise SystemExit("No candidates succeeded")

    # Candidate summary
    cand_csv = run_dir / "candidates.csv"
    with cand_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "recipe", "seed", "time_s", "score_no_overlap", "path"])
        for c in candidates:
            w.writerow(
                [
                    c.cid,
                    c.recipe,
                    c.seed,
                    f"{c.time_s:.3f}",
                    f"{_prefix_total(c.s, nmax=ns.nmax):.12f}",
                    str(c.csv_path),
                ]
            )

    # Ensemble per puzzle n
    check_overlap = ns.overlap_check != "none"
    ensemble_puzzles: dict[int, np.ndarray] = {}
    ensemble_rows: list[dict[str, Any]] = []

    for n in range(1, ns.nmax + 1):
        order = sorted(range(len(candidates)), key=lambda i: float(candidates[i].s[n]))
        picked: Candidate | None = None
        picked_s = float("inf")
        for idx in order:
            c = candidates[idx]
            poses = c.puzzles.get(n)
            if poses is None or poses.shape[0] != n:
                continue
            if check_overlap and first_overlap_pair(points, poses, eps=EPS, mode="strict") is not None:
                continue
            picked = c
            picked_s = float(c.s[n])
            ensemble_puzzles[n] = poses
            break
        if picked is None:
            raise SystemExit(f"Failed to pick a feasible candidate for puzzle {n}")
        ensemble_rows.append({"puzzle": n, "candidate": picked.cid, "s": picked_s})

    ensemble_csv = run_dir / "ensemble_submission.csv"
    _write_submission(ensemble_csv, ensemble_puzzles, nmax=ns.nmax)

    # Write ensemble choices + score
    choices_csv = run_dir / "ensemble_choices.csv"
    with choices_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["puzzle", "candidate", "s"])
        for row in ensemble_rows:
            w.writerow([row["puzzle"], row["candidate"], f"{row['s']:.17f}"])

    puzzles_ens = load_submission(ensemble_csv, nmax=ns.nmax)
    s_ens = _compute_s(points, puzzles_ens, nmax=ns.nmax)
    ens_score = _prefix_total(s_ens, nmax=ns.nmax)
    from santa_packing.scoring import score_submission  # noqa: E402

    _ = score_submission(
        ensemble_csv,
        nmax=ns.nmax,
        check_overlap=check_overlap,
        overlap_mode="strict",
        require_complete=True,
    )

    best_single = min(candidates, key=lambda c: _prefix_total(c.s, nmax=ns.nmax))
    best_single_score = _prefix_total(best_single.s, nmax=ns.nmax)

    summary = (
        "# Sweep + per-n ensemble\n\n"
        f"- nmax: {ns.nmax}\n"
        f"- candidates: {len(candidates)} (recipes={len(recipes)} seeds={len(seeds)})\n"
        f"- overlap_check: {ns.overlap_check}\n\n"
        f"## Best single candidate\n"
        f"- candidate: {best_single.cid}\n"
        f"- score (no overlap check): {best_single_score:.12f}\n"
        f"- csv: {best_single.csv_path}\n\n"
        f"## Ensemble\n"
        f"- score (no overlap check): {ens_score:.12f}\n"
        f"- csv: {ensemble_csv}\n"
        f"- choices: {choices_csv}\n"
        f"- candidates table: {cand_csv}\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    code_audit = (
        "# Code audit: sweep + per-n ensemble\n\n"
        "## What this script does\n"
        "- Runs multiple calls to `python -m santa_packing.cli.generate_submission` (multi-start) for different recipes/seeds.\n"
        "- Computes `s_n` for every candidate and every puzzle `n`.\n"
        "- Builds an ensemble submission by picking, **for each n**, the candidate with the smallest `s_n` (optionally requiring no-overlap).\n\n"
        "## Key files\n"
        "- `santa_packing/cli/sweep_ensemble.py`: orchestration + ensemble selection.\n"
        "- `santa_packing/cli/generate_submission.py`: single-run generator (all knobs/recipes flow through it).\n"
        "- `santa_packing/postopt_np.py:has_overlaps`: overlap checker used for selecting feasible puzzles.\n\n"
        "## Safety / correctness\n"
        "- `--overlap-check selected` verifies feasibility puzzle-by-puzzle while selecting the ensemble.\n"
        "- If you use `--overlap-check none`, the ensemble may be invalid.\n\n"
        "## Next minimal experiments\n"
        "1) Sweep only seeds (fixed recipe) and measure ensemble gain.\n"
        "2) Add lattice variants (pattern/margin/rotations) before touching SA knobs.\n"
        "3) If JAX is available, include SA proposal variants and adaptive cooling as separate recipes.\n"
    )
    (run_dir / "code_audit.md").write_text(code_audit, encoding="utf-8")

    if ns.out is not None:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        ns.out.write_bytes(ensemble_csv.read_bytes())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
