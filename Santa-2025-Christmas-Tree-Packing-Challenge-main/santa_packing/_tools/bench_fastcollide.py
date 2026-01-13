#!/usr/bin/env python3

"""Tool to benchmark the optional `fastcollide` C++ extension."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from santa_packing.geom_np import transform_polygon
from santa_packing.scoring import _polygons_intersect_strict as _polygons_intersect
from santa_packing.tree_data import TREE_POINTS


def _repo_root_from_cwd() -> Path:
    cwd = Path.cwd().resolve()
    for cand in (cwd, *cwd.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    return cwd


def _try_import_fast():
    try:
        import santa_packing.fastcollide as fastcollide  # type: ignore
    except Exception:
        return None
    return fastcollide


def main() -> int:
    """Run the benchmark and print a timing/parity summary."""
    ap = argparse.ArgumentParser(description="Benchmark C++ fastcollide vs Python scoring._polygons_intersect_strict")
    ap.add_argument("--n-polys", type=int, default=256, help="Number of transformed polygons")
    ap.add_argument("--pairs", type=int, default=20000, help="Number of random pairs to test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", action="store_true", help="Build extension if not importable")
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    points = np.array(TREE_POINTS, dtype=float)

    # Create a pool of polygons with random poses.
    poses = np.zeros((args.n_polys, 3), dtype=float)
    poses[:, 0:2] = rng.uniform(-10.0, 10.0, size=(args.n_polys, 2))
    poses[:, 2] = rng.uniform(0.0, 360.0, size=(args.n_polys,))
    polys = [transform_polygon(points, pose) for pose in poses]

    pairs = rng.integers(0, args.n_polys, size=(args.pairs, 2), endpoint=False, dtype=np.int64)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.shape[0] == 0:
        raise SystemExit("All sampled pairs were identical indices; try increasing --pairs.")

    fast = _try_import_fast()
    if fast is None and args.build:
        import subprocess

        root = _repo_root_from_cwd()
        cmd = [sys.executable, str(root / "scripts" / "build" / "build_fastcollide.py")]
        print("+", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(root))
        fast = _try_import_fast()

    if fast is None:
        raise SystemExit("fastcollide not importable. Run: python3 scripts/build/build_fastcollide.py")

    # Parity check + timing (python)
    t0 = time.perf_counter()
    py_hits = 0
    py_out = np.empty((pairs.shape[0],), dtype=bool)
    for k, (i, j) in enumerate(pairs):
        v = bool(_polygons_intersect(polys[i], polys[j]))
        py_out[k] = v
        py_hits += int(v)
    t1 = time.perf_counter()

    # Timing (C++)
    t2 = time.perf_counter()
    cc_hits = 0
    cc_out = np.empty((pairs.shape[0],), dtype=bool)
    for k, (i, j) in enumerate(pairs):
        v = bool(fast.polygons_intersect(polys[i], polys[j]))
        cc_out[k] = v
        cc_hits += int(v)
    t3 = time.perf_counter()

    mism = int(np.sum(py_out != cc_out))
    print("bench_fastcollide")
    print(f"n_polys: {args.n_polys}  pairs: {pairs.shape[0]}")
    print(f"python_hits: {py_hits}  cpp_hits: {cc_hits}  mismatches: {mism}")
    print(f"python_s: {t1 - t0:.3f}  cpp_s: {t3 - t2:.3f}")
    if (t3 - t2) > 1e-12:
        print(f"speedup: {(t1 - t0) / (t3 - t2):.2f}x")
    if mism:
        raise SystemExit("Mismatch detected between python and C++ results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
