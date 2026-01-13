# %% Kaggle/local bootstrap (deps + env)
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

IS_KAGGLE = Path("/kaggle").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
if IS_KAGGLE:
    # Evita o JAX pre-alocar 100% da memoria (especialmente em GPU)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# JAX e necessario para o L2O. Se falhar no Kaggle, habilite Internet e rode novamente.
try:
    import jax  # noqa: F401
except Exception:
    print("[setup] JAX nao encontrado; instalando jax[cpu]...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "jax[cpu]"])
    import jax  # noqa: F401

# %% Setup
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from datetime import datetime
import itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, Tuple

import os
import importlib
import inspect
import jax
import jax.numpy as jnp
import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Resolve repo root (local ou Kaggle)
def resolve_repo_root() -> Path:
    env_root = os.environ.get("SANTA_REPO_ROOT") or os.environ.get("REPO_ROOT") or os.environ.get("PROJECT_ROOT")
    if env_root:
        p = Path(env_root).expanduser()
        if (p / "santa_packing").exists():
            return p.resolve()

    cwd = Path.cwd()
    if (cwd / "santa_packing").exists():
        return cwd.resolve()
    if (cwd.parent / "santa_packing").exists():
        return cwd.parent.resolve()

    # Kaggle: codigo normalmente esta em /kaggle/input/<dataset>/...
    kaggle_input = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
    if kaggle_input.exists():
        candidates: list[Path] = []
        for base in kaggle_input.iterdir():
            if not base.is_dir():
                continue
            if (base / "santa_packing").exists():
                candidates.append(base)
                continue
            # comum: dataset/<repo_root>/*
            for child in base.iterdir():
                if child.is_dir() and (child / "santa_packing").exists():
                    candidates.append(child)

        def _score(p: Path) -> tuple[int, str]:
            s = 0
            if (p / "santa_packing" / "l2o.py").exists():
                s += 2
            if (p / "santa_packing" / "cli" / "generate_submission.py").exists():
                s += 1
            if (p / "notebooks" / "run_l2o.ipynb").exists():
                s += 1
            return (-s, str(p))

        uniq = {p.resolve() for p in candidates}
        candidates = sorted(uniq, key=_score)
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        "Nao encontrei o repo root. Rode do root/notebooks ou defina SANTA_REPO_ROOT (ou REPO_ROOT)."
    )


ROOT = resolve_repo_root()

# Pasta gravavel (Kaggle: /kaggle/working)
WORK_DIR = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")) if Path("/kaggle/working").exists() else ROOT
WORK_DIR.mkdir(parents=True, exist_ok=True)

print("[paths] ROOT =", ROOT)
print("[paths] WORK_DIR =", WORK_DIR)

sys.path.insert(0, str(ROOT))

from santa_packing.geom_np import (  # noqa: E402
    packing_score,
    polygon_bbox,
    polygon_radius,
    prefix_score,
    shift_poses_to_origin,
    transform_polygon,
)
import santa_packing.l2o as l2o_mod  # noqa: E402
from santa_packing.optimizer import run_sa_batch  # noqa: E402
import santa_packing.cli.train_l2o as train_l2o_mod  # noqa: E402
from santa_packing.tree_data import TREE_POINTS  # noqa: E402

train_l2o_mod = importlib.reload(train_l2o_mod)
l2o_mod = importlib.reload(l2o_mod)
L2OConfig = l2o_mod.L2OConfig
load_params_npz = l2o_mod.load_params_npz
save_params_npz = l2o_mod.save_params_npz
optimize_with_l2o = l2o_mod.optimize_with_l2o


def train_l2o_model_safe(**kwargs):
    seed = kwargs.get("seed")
    if seed is not None:
        try:
            seed_int = int(seed)
        except Exception:
            seed_int = None
        if seed_int is not None:
            np.random.seed(seed_int)
            random.seed(seed_int)

    sig = inspect.signature(train_l2o_mod.train_l2o_model)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    missing = sorted(set(kwargs) - set(allowed))
    if missing:
        print(f"[warn] train_l2o_model ignorou parametros nao suportados: {missing}")
    return train_l2o_mod.train_l2o_model(**allowed)


def parse_int_list(text: str) -> list[int]:
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


# %% Initial layouts
def grid_initial(n: int, spacing: float) -> np.ndarray:
    cols = int(np.ceil(np.sqrt(n)))
    poses = np.zeros((n, 3), dtype=float)
    for i in range(n):
        row = i // cols
        col = i % cols
        poses[i] = (col * spacing, row * spacing, 0.0)
    return poses


def random_initial(n: int, spacing: float, rng: np.random.Generator, rand_scale: float) -> np.ndarray:
    scale = spacing * math.sqrt(max(n, 1)) * rand_scale
    xy = rng.uniform(-scale, scale, size=(n, 2))
    theta = rng.uniform(0.0, 360.0, size=(n, 1))
    return np.concatenate([xy, theta], axis=1)


def make_initial(
    points: np.ndarray,
    n: int,
    spacing: float,
    seed: int,
    init_mode: str,
    rand_scale: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if init_mode == "grid":
        poses = grid_initial(n, spacing)
    elif init_mode == "random":
        poses = random_initial(n, spacing, rng, rand_scale)
    else:
        poses = grid_initial(n, spacing) if seed % 2 == 0 else random_initial(n, spacing, rng, rand_scale)
    return shift_poses_to_origin(points, poses)


# %% Scoring/plots/utilities
def prefix_packing_score_np(points: np.ndarray, poses: np.ndarray) -> float:
    if poses.shape[0] == 0:
        return 0.0
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    s_values: List[float] = []
    for pose in poses:
        bbox = polygon_bbox(transform_polygon(points, pose))
        min_x = min(min_x, float(bbox[0]))
        min_y = min(min_y, float(bbox[1]))
        max_x = max(max_x, float(bbox[2]))
        max_y = max(max_y, float(bbox[3]))
        width = max_x - min_x
        height = max_y - min_y
        s_values.append(max(width, height))
    return float(prefix_score(s_values))


def plot_packing(poses: np.ndarray, title: str, out_path: Path | None = None) -> None:
    points = np.array(TREE_POINTS, dtype=float)
    plt.figure(figsize=(6, 6))
    for pose in poses:
        poly = transform_polygon(points, pose)
        p = np.vstack([poly, poly[0]])
        plt.plot(p[:, 0], p[:, 1], "g-")
    plt.axis("equal")
    plt.title(title)
    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}")


def run_cmd_capture(cmd: List[str]) -> str:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}:\n{result.stdout}")
    return result.stdout


def score_csv(csv_path: Path, *, nmax: int, check_overlap: bool) -> Dict[str, object]:
    # Prefer in-process scorer (faster + avoids subprocess overhead).
    try:
        from santa_packing.scoring import score_submission  # noqa: E402
    except Exception:
        cmd = [
            sys.executable,
            "-m",
            "santa_packing.cli.score_submission",
            str(csv_path),
            "--nmax",
            str(nmax),
        ]
        if not check_overlap:
            cmd.append("--no-overlap")
        out = run_cmd_capture(cmd).strip()
        return json.loads(out) if out else {}

    result = score_submission(
        csv_path,
        nmax=int(nmax),
        check_overlap=bool(check_overlap),
        require_complete=True,
    )
    try:
        return dict(result.to_json())
    except Exception:
        # Fallback (should not happen): keep a minimal stable shape.
        return {"nmax": int(nmax), "overlap_check": bool(check_overlap)}


def generate_submission(
    out_csv: Path,
    *,
    seed: int,
    nmax: int,
    args: Dict[str, object],
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "santa_packing.cli.generate_submission",
        "--out",
        str(out_csv),
        "--seed",
        str(seed),
        "--nmax",
        str(nmax),
    ]
    for key, value in args.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd += [flag, str(value)]
    run_cmd(cmd)


def l2o_config_from_meta(meta: Dict[str, object], *, reward: str, deterministic: bool) -> L2OConfig:
    def _get_int(key: str, default: int) -> int:
        val = meta.get(key, default)
        return int(val) if hasattr(val, "__int__") else default

    def _get_float(key: str, default: float) -> float:
        val = meta.get(key, default)
        if isinstance(val, (float, np.floating)):
            return float(val)
        if isinstance(val, (int, np.integer)):
            return float(val)
        if isinstance(val, np.ndarray) and val.shape == ():
            return float(val.item())
        return default

    def _get_bool(key: str, default: bool) -> bool:
        val = meta.get(key, default)
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, (int, np.integer)):
            return bool(int(val))
        if isinstance(val, np.ndarray) and val.shape == ():
            return bool(val.item())
        return default

    policy = str(meta.get("policy", "mlp"))
    knn_k = _get_int("knn_k", 4)
    hidden = _get_int("hidden", 32)
    mlp_depth = _get_int("mlp_depth", 1)
    gnn_steps = _get_int("gnn_steps", 1)
    gnn_attention = _get_bool("gnn_attention", False)
    action_scale = _get_float("action_scale", 1.0)
    feature_mode = str(meta.get("feature_mode", "raw"))
    overlap_penalty = _get_float("overlap_penalty", 50.0)
    overlap_lambda = _get_float("overlap_lambda", 0.0)

    return L2OConfig(
        hidden_size=hidden,
        policy=policy,
        knn_k=knn_k,
        reward=reward,
        mlp_depth=mlp_depth,
        gnn_steps=gnn_steps,
        gnn_attention=gnn_attention,
        feature_mode=feature_mode,
        action_scale=action_scale,
        overlap_penalty=overlap_penalty,
        overlap_lambda=overlap_lambda,
        action_noise=not deterministic,
    )


# %% Evaluation helpers
def evaluate_solver(
    name: str,
    solve_fn: Callable[[int, int], np.ndarray | Tuple[np.ndarray, Dict[str, str]]],
    n_list: Iterable[int],
    seeds: Iterable[int],
    points: np.ndarray,
    split: str,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for n in n_list:
        for seed in seeds:
            result = solve_fn(n, seed)
            info: Dict[str, object] = {}
            if isinstance(result, tuple):
                poses, info = result
            else:
                poses = result
            prefix = prefix_packing_score_np(points, poses)
            pack = packing_score(points, poses)
            row = {
                "split": split,
                "model": name,
                "n": int(n),
                "seed": int(seed),
                "prefix_score": float(prefix),
                "packing_score": float(pack),
            }
            if isinstance(info, dict) and "selected" in info:
                row["selected"] = str(info["selected"])
            rows.append(row)
    return rows


def summarize_results(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, float]]] = {}
    for row in rows:
        key = (row["split"], row["model"], int(row["n"]))
        grouped.setdefault(key, []).append(row)

    summary: List[Dict[str, float]] = []
    for (split, model, n), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        prefix_vals = np.array([r["prefix_score"] for r in items], dtype=float)
        pack_vals = np.array([r["packing_score"] for r in items], dtype=float)
        summary.append(
            {
                "split": split,
                "model": model,
                "n": int(n),
                "samples": int(prefix_vals.size),
                "prefix_mean": float(prefix_vals.mean()),
                "prefix_std": float(prefix_vals.std()),
                "packing_mean": float(pack_vals.mean()),
                "packing_std": float(pack_vals.std()),
            }
        )
    return summary


def summarize_overall(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault((row["split"], row["model"]), []).append(row)

    overall: List[Dict[str, float]] = []
    for (split, model), items in sorted(grouped.items()):
        prefix_vals = np.array([r["prefix_score"] for r in items], dtype=float)
        pack_vals = np.array([r["packing_score"] for r in items], dtype=float)
        overall.append(
            {
                "split": split,
                "model": model,
                "samples": int(prefix_vals.size),
                "prefix_mean": float(prefix_vals.mean()),
                "prefix_std": float(prefix_vals.std()),
                "packing_mean": float(pack_vals.mean()),
                "packing_std": float(pack_vals.std()),
            }
        )
    return overall


def challenge_score_from_results(rows: List[Dict[str, float]], model: str, split: str) -> float:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        if row.get("model") != model or row.get("split") != split:
            continue
        grouped.setdefault(int(row["n"]), []).append(float(row["packing_score"]))
    total = 0.0
    for n in sorted(grouped):
        mean_s = float(np.mean(grouped[n]))
        total += (mean_s * mean_s) / n
    return total


def print_challenge_scores(
    rows: List[Dict[str, float]],
    *,
    split: str,
    models: Iterable[str] | None = None,
    title: str | None = None,
) -> Dict[str, float]:
    """Print the Kaggle metric (sum(s_n^2/n)) on the evaluated subset.

    Note: this uses the mean packing_score per (n) for the given split.
    """
    available = {str(r.get("model")) for r in rows if str(r.get("split")) == split}
    if models is None:
        selected = sorted(available)
    else:
        selected = [str(m) for m in models if str(m) in available]

    if not selected:
        return {}

    scores = {m: float(challenge_score_from_results(rows, m, split)) for m in selected}
    if title:
        print(title)
    for m in sorted(scores, key=lambda k: (scores[k], k)):
        print(f"  {split:>5s}  {m:<16s}  score={scores[m]:.6f}")
    return scores


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_eval_artifacts(
    run_dir: Path,
    rows: List[Dict[str, float]],
    summary: List[Dict[str, float]],
    overall: List[Dict[str, float]],
    meta: Dict[str, object],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "metrics.csv", rows)
    write_csv(run_dir / "per_n.csv", summary)
    write_csv(run_dir / "overall.csv", overall)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    lines = [
        "# L2O evaluation summary",
        "",
        "Lower is better. Prefix score matches the leaderboard aggregate.",
        "",
        "## Overall (mean across n and seeds)",
    ]
    for row in overall:
        lines.append(
            f"- [{row['split']}] {row['model']}: prefix={row['prefix_mean']:.4f} +/- {row['prefix_std']:.4f}, "
            f"packing={row['packing_mean']:.4f} +/- {row['packing_std']:.4f}"
        )
    (run_dir / "summary.md").write_text("\n".join(lines))


def plot_eval_curves(summary: List[Dict[str, float]], out_path: Path | None = None) -> None:
    if not summary:
        return
    grouped: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for row in summary:
        grouped.setdefault((row["split"], row["model"]), []).append(row)

    plt.figure(figsize=(7, 4))
    for (split, model), items in grouped.items():
        items = sorted(items, key=lambda r: r["n"])
        ns = [r["n"] for r in items]
        means = [r["prefix_mean"] for r in items]
        stds = [r["prefix_std"] for r in items]
        plt.plot(ns, means, marker="o", label=f"{model} ({split})")
        plt.fill_between(ns, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)
    plt.xlabel("n")
    plt.ylabel("prefix score")
    plt.title("Prefix score vs n")
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


# %% Configuracoes
# === Configuracoes ===
TRAIN_N_LIST = [8, 10, 12]
VAL_N_LIST = [6, 9, 11]
TRAIN_STEPS = 400
ROLLOUT_STEPS = 50
BATCH = 64
REWARD = "prefix"  # "packing" ou "prefix"
HIDDEN_SIZE = 32
ACTION_SCALE = 0.05
FEATURE_MODE = "raw"  # raw | bbox_norm
TRAIN_INIT_MODE = "all"  # grid | random | mix | lattice | all
TRAIN_LATTICE_PATTERN = "hex"
TRAIN_LATTICE_MARGIN = 0.02
TRAIN_LATTICE_ROTATE = 0.0
TRAIN_CURRICULUM = False
TRAIN_CURRICULUM_START_MAX = None
TRAIN_CURRICULUM_END_MAX = None
TRAIN_CURRICULUM_STEPS = None

BASELINE_MODE = "ema"  # "batch" (baseline por batch) | "ema" (media movel)
BASELINE_DECAY = 0.9

MLP_DEPTH = 2
GNN_STEPS = 2
GNN_ATTENTION = False

TRAIN_EVAL_SEEDS = [0, 1, 2]
VAL_EVAL_SEEDS = [3, 4, 5]
EVAL_STEPS = 50
INIT_MODE = "grid"  # grid | random | mix
RAND_SCALE = 0.3

# ---- L2O sweep (muitos experimentos) ----
# Ideia: explorar varias configs (seed/hparams) rapidamente, rankear no "val" e
# depois re-treinar apenas o melhor MLP/GNN com o budget cheio.
RUN_L2O_SWEEP = True
L2O_SWEEP_SEEDS = "1..8"  # ex.: "1..30" para MUITOS seeds
L2O_SWEEP_FEATURE_MODES = ["raw", "bbox_norm", "rich"]
L2O_SWEEP_HIDDEN_SIZES = [32, 64]
L2O_SWEEP_ACTION_SCALES = [0.05, 0.1]
L2O_SWEEP_MLP_DEPTHS = [1, 2, 3]
L2O_SWEEP_GNN_STEPS = [1, 2, 3]
L2O_SWEEP_GNN_ATTENTION = [False, True]
L2O_SWEEP_KNN_K = [4, 6]
L2O_SWEEP_LR = [1e-3, 3e-4]
L2O_SWEEP_OVERLAP_LAMBDA = [0.0, 0.01]

L2O_SWEEP_TRAIN_STEPS = 200
L2O_SWEEP_ROLLOUT_STEPS = ROLLOUT_STEPS
L2O_SWEEP_EVAL_N_LIST = VAL_N_LIST
L2O_SWEEP_EVAL_SEEDS = VAL_EVAL_SEEDS
L2O_SWEEP_EVAL_STEPS = EVAL_STEPS
L2O_SWEEP_MAX_EXPERIMENTS = None  # None = roda tudo (pode explodir)
L2O_SWEEP_TOPK_PER_POLICY = 5  # exporta top-K p/ usar no generate_submission (opcional)
L2O_SWEEP_RETRAIN_FINAL = True
L2O_FINAL_TRAIN_STEPS = TRAIN_STEPS
L2O_FINAL_ROLLOUT_STEPS = ROLLOUT_STEPS

SA_STEPS = 300
SA_TRANS_SIGMA = 0.2
SA_ROT_SIGMA = 15.0
SA_ROT_PROB = 0.3
SA_OBJECTIVE = REWARD

RUN_BC_PIPELINE = True
BC_POLICY = "gnn"
BC_KNN_K = 4
BC_RUNS_PER_N = 3
BC_STEPS = 200
BC_TRAIN_STEPS = 200
BC_SEED = 0
BC_INIT_MODE = "all"  # grid | random | mix | lattice | all
BC_RAND_SCALE = 0.3
BC_LATTICE_PATTERN = "hex"
BC_LATTICE_MARGIN = 0.02
BC_LATTICE_ROTATE = 0.0
BC_CURRICULUM = False
BC_CURRICULUM_START_MAX = None
BC_CURRICULUM_END_MAX = None
BC_CURRICULUM_STEPS = None
BC_DATASET_PATH = None  # sobrescreva para reutilizar dataset
BC_POLICY_PATH = None  # sobrescreva para reutilizar policy

RUN_META_TRAIN = True
META_INIT_MODEL_PATH = None
META_TRAIN_STEPS = 30
META_ES_POP = 6
META_SA_STEPS = 150

RUN_HEATMAP_TRAIN = True
HEATMAP_MODEL_PATH = None
HEATMAP_TRAIN_STEPS = 30
HEATMAP_ES_POP = 6
HEATMAP_STEPS = 200

RUN_ENSEMBLE = True
ENSEMBLE_SCORE = "prefix"  # criterio de selecao no ensemble
L2O_REFINE_GRID = False
L2O_REFINE_SA = False
REFINE_STEPS = 100

# %% Treino das politicas (setup)
RUN_DIR = WORK_DIR / "runs" / f"l2o_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

points = np.array(TREE_POINTS, dtype=float)
spacing = 2.0 * polygon_radius(points) * 1.2
VIS_N = TRAIN_N_LIST[0]
init = shift_poses_to_origin(points, grid_initial(VIS_N, spacing))

# %% Solvers (needed early for sweep/eval)
initial_cache: Dict[Tuple[int, int], np.ndarray] = {}


def get_initial(n: int, seed: int) -> np.ndarray:
    key = (n, seed)
    if key not in initial_cache:
        initial_cache[key] = make_initial(points, n, spacing, seed, INIT_MODE, RAND_SCALE)
    return initial_cache[key]


def solve_grid(n: int, seed: int) -> np.ndarray:
    return get_initial(n, seed)


def solve_sa(n: int, seed: int) -> np.ndarray:
    init_pose = get_initial(n, seed)
    init_batch = jnp.array(init_pose)[None, :, :]
    key = jax.random.PRNGKey(seed)
    best_poses, _ = run_sa_batch(
        key,
        SA_STEPS,
        n,
        init_batch,
        trans_sigma=SA_TRANS_SIGMA,
        rot_sigma=SA_ROT_SIGMA,
        rot_prob=SA_ROT_PROB,
        objective=SA_OBJECTIVE,
    )
    return np.array(best_poses[0])


def solve_l2o(params, cfg: L2OConfig, *, steps: int | None = None) -> Callable[[int, int], np.ndarray]:
    def _solve(n: int, seed: int) -> np.ndarray:
        init_pose = get_initial(n, seed)
        key = jax.random.PRNGKey(seed)
        nsteps = EVAL_STEPS if steps is None else int(steps)
        poses = optimize_with_l2o(key, params, jnp.array(init_pose), nsteps, cfg)
        return np.array(poses)

    return _solve


def solve_l2o_refine(
    base_solver: Callable[[int, int], np.ndarray],
    params,
    cfg: L2OConfig,
    *,
    steps: int | None = None,
) -> Callable[[int, int], np.ndarray]:
    def _solve(n: int, seed: int) -> np.ndarray:
        base_pose = base_solver(n, seed)
        key = jax.random.PRNGKey(seed)
        nsteps = REFINE_STEPS if steps is None else int(steps)
        poses = optimize_with_l2o(key, params, jnp.array(base_pose), nsteps, cfg)
        return np.array(poses)

    return _solve


# %% Treino L2O (sweep -> seleciona os melhores)
L2O_SWEEP_TOP_MODELS: Dict[str, Path] = {}
mlp_meta: Dict[str, object] = {
    "policy": "mlp",
    "hidden": HIDDEN_SIZE,
    "mlp_depth": MLP_DEPTH,
    "gnn_steps": GNN_STEPS,
    "gnn_attention": GNN_ATTENTION,
    "feature_mode": FEATURE_MODE,
    "action_scale": ACTION_SCALE,
    "knn_k": 4,
    "overlap_penalty": 50.0,
    "overlap_lambda": 0.0,
}
gnn_meta: Dict[str, object] = {
    "policy": "gnn",
    "hidden": HIDDEN_SIZE,
    "knn_k": 4,
    "mlp_depth": MLP_DEPTH,
    "gnn_steps": GNN_STEPS,
    "gnn_attention": GNN_ATTENTION,
    "feature_mode": FEATURE_MODE,
    "action_scale": ACTION_SCALE,
    "overlap_penalty": 50.0,
    "overlap_lambda": 0.0,
}

if RUN_L2O_SWEEP:
    sweep_dir = RUN_DIR / "l2o_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    def _cfg_id(data: Dict[str, object]) -> str:
        blob = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:10]

    seeds = parse_int_list(L2O_SWEEP_SEEDS)
    if not seeds:
        seeds = [1]

    planned: List[Dict[str, object]] = []

    # MLP space
    for seed, feature_mode, hidden, action_scale, lr, overlap_lambda, mlp_depth in itertools.product(
        seeds,
        L2O_SWEEP_FEATURE_MODES,
        L2O_SWEEP_HIDDEN_SIZES,
        L2O_SWEEP_ACTION_SCALES,
        L2O_SWEEP_LR,
        L2O_SWEEP_OVERLAP_LAMBDA,
        L2O_SWEEP_MLP_DEPTHS,
    ):
        meta = {
            "policy": "mlp",
            "seed": int(seed),
            "hidden": int(hidden),
            "mlp_depth": int(mlp_depth),
            "gnn_steps": int(GNN_STEPS),
            "gnn_attention": bool(GNN_ATTENTION),
            "knn_k": 4,
            "feature_mode": str(feature_mode),
            "action_scale": float(action_scale),
            "lr": float(lr),
            "baseline_mode": str(BASELINE_MODE),
            "baseline_decay": float(BASELINE_DECAY),
            "overlap_penalty": 50.0,
            "overlap_lambda": float(overlap_lambda),
        }
        cid = _cfg_id(meta)
        planned.append({"name": f"mlp_{cid}", "meta": meta})

    # GNN space
    for (
        seed,
        feature_mode,
        hidden,
        action_scale,
        lr,
        overlap_lambda,
        gnn_steps,
        gnn_attention,
        knn_k,
    ) in itertools.product(
        seeds,
        L2O_SWEEP_FEATURE_MODES,
        L2O_SWEEP_HIDDEN_SIZES,
        L2O_SWEEP_ACTION_SCALES,
        L2O_SWEEP_LR,
        L2O_SWEEP_OVERLAP_LAMBDA,
        L2O_SWEEP_GNN_STEPS,
        L2O_SWEEP_GNN_ATTENTION,
        L2O_SWEEP_KNN_K,
    ):
        meta = {
            "policy": "gnn",
            "seed": int(seed),
            "hidden": int(hidden),
            "knn_k": int(knn_k),
            "mlp_depth": int(MLP_DEPTH),
            "gnn_steps": int(gnn_steps),
            "gnn_attention": bool(gnn_attention),
            "feature_mode": str(feature_mode),
            "action_scale": float(action_scale),
            "lr": float(lr),
            "baseline_mode": str(BASELINE_MODE),
            "baseline_decay": float(BASELINE_DECAY),
            "overlap_penalty": 50.0,
            "overlap_lambda": float(overlap_lambda),
        }
        cid = _cfg_id(meta)
        planned.append({"name": f"gnn_{cid}", "meta": meta})

    planned = sorted(planned, key=lambda r: str(r["name"]))
    if L2O_SWEEP_MAX_EXPERIMENTS is not None:
        planned = planned[: int(L2O_SWEEP_MAX_EXPERIMENTS)]

    (sweep_dir / "sweep_meta.json").write_text(
        json.dumps(
            {
                "planned_total": int(len(planned)),
                "seeds": list(seeds),
                "train_steps": int(L2O_SWEEP_TRAIN_STEPS),
                "rollout_steps": int(L2O_SWEEP_ROLLOUT_STEPS),
                "eval_n_list": list(L2O_SWEEP_EVAL_N_LIST),
                "eval_seeds": list(L2O_SWEEP_EVAL_SEEDS),
                "eval_steps": int(L2O_SWEEP_EVAL_STEPS),
                "max_experiments": L2O_SWEEP_MAX_EXPERIMENTS,
            },
            indent=2,
        )
    )

    sweep_rows: List[Dict[str, object]] = []
    for idx, item in enumerate(planned, start=1):
        name = str(item["name"])
        meta = dict(item["meta"])  # type: ignore[arg-type]

        out_path = sweep_dir / f"{name}.npz"
        if out_path.exists():
            params, saved_meta = load_params_npz(out_path)
            meta = dict(saved_meta)
            loss = []
        else:
            params, loss = train_l2o_model_safe(
                seed=int(meta["seed"]),
                n_list=TRAIN_N_LIST,
                batch=BATCH,
                train_steps=L2O_SWEEP_TRAIN_STEPS,
                steps=L2O_SWEEP_ROLLOUT_STEPS,
                lr=float(meta.get("lr", 1e-3)),
                hidden_size=int(meta.get("hidden", HIDDEN_SIZE)),
                policy=str(meta.get("policy", "mlp")),
                reward=REWARD,
                action_scale=float(meta.get("action_scale", ACTION_SCALE)),
                knn_k=int(meta.get("knn_k", 4)),
                mlp_depth=int(meta.get("mlp_depth", MLP_DEPTH)),
                gnn_steps=int(meta.get("gnn_steps", GNN_STEPS)),
                gnn_attention=bool(meta.get("gnn_attention", False)),
                init_mode=TRAIN_INIT_MODE,
                rand_scale=RAND_SCALE,
                lattice_pattern=TRAIN_LATTICE_PATTERN,
                lattice_margin=TRAIN_LATTICE_MARGIN,
                lattice_rotate=TRAIN_LATTICE_ROTATE,
                baseline_mode=str(meta.get("baseline_mode", BASELINE_MODE)),
                baseline_decay=float(meta.get("baseline_decay", BASELINE_DECAY)),
                curriculum=TRAIN_CURRICULUM,
                curriculum_start_max=TRAIN_CURRICULUM_START_MAX,
                curriculum_end_max=TRAIN_CURRICULUM_END_MAX,
                curriculum_steps=TRAIN_CURRICULUM_STEPS,
                feature_mode=str(meta.get("feature_mode", FEATURE_MODE)),
                overlap_lambda=float(meta.get("overlap_lambda", 0.0)),
                verbose_freq=0,
            )
            save_params_npz(out_path, params, meta=meta)

        eval_cfg = l2o_config_from_meta(meta, reward=REWARD, deterministic=True)
        eval_rows = evaluate_solver(
            name,
            solve_l2o(params, eval_cfg, steps=L2O_SWEEP_EVAL_STEPS),
            L2O_SWEEP_EVAL_N_LIST,
            L2O_SWEEP_EVAL_SEEDS,
            points,
            split="sweep_val",
        )
        val_score = float(challenge_score_from_results(eval_rows, name, "sweep_val"))

        sweep_rows.append(
            {
                "name": name,
                "policy": str(meta.get("policy")),
                "seed": int(meta.get("seed", -1)),
                "hidden": int(meta.get("hidden", -1)),
                "feature_mode": str(meta.get("feature_mode")),
                "action_scale": float(meta.get("action_scale", float("nan"))),
                "lr": float(meta.get("lr", float("nan"))),
                "overlap_lambda": float(meta.get("overlap_lambda", float("nan"))),
                "val_score": val_score,
                "path": str(out_path),
                "loss_last": float(loss[-1]) if loss else float("nan"),
            }
        )

        if idx == 1 or idx % 5 == 0 or idx == len(planned):
            print(f"[l2o_sweep] {idx}/{len(planned)} done. best_val={min(r['val_score'] for r in sweep_rows):.6f}")

    sweep_rows = sorted(sweep_rows, key=lambda r: (float(r.get("val_score") or float("inf")), str(r.get("name"))))
    write_csv(sweep_dir / "rank.csv", sweep_rows)
    (sweep_dir / "rank.json").write_text(json.dumps(sweep_rows, indent=2))

    def _topk(policy: str) -> List[Dict[str, object]]:
        items = [r for r in sweep_rows if str(r.get("policy")) == policy]
        return items[: int(max(L2O_SWEEP_TOPK_PER_POLICY, 0))]

    top_mlp = _topk("mlp")
    top_gnn = _topk("gnn")
    for r in top_mlp + top_gnn:
        name = str(r["name"])
        L2O_SWEEP_TOP_MODELS[name] = Path(str(r["path"]))

    best_mlp = top_mlp[0] if top_mlp else None
    best_gnn = top_gnn[0] if top_gnn else None
    (sweep_dir / "best.json").write_text(json.dumps({"best_mlp": best_mlp, "best_gnn": best_gnn}, indent=2))
    (sweep_dir / "top_models.json").write_text(json.dumps({k: str(v) for k, v in L2O_SWEEP_TOP_MODELS.items()}, indent=2))

    def _fmt_top(items: List[Dict[str, object]], k: int = 5) -> str:
        parts = []
        for r in items[:k]:
            parts.append(f"{r['name']}:{float(r['val_score']):.6f}")
        return ", ".join(parts) if parts else "(none)"

    print(f"[l2o_sweep] top_mlp: {_fmt_top(top_mlp)}")
    print(f"[l2o_sweep] top_gnn: {_fmt_top(top_gnn)}")

    summary_lines = [
        "# L2O sweep summary",
        "",
        f"- experiments: {len(planned)}",
        f"- eval split: sweep_val (n={list(L2O_SWEEP_EVAL_N_LIST)} seeds={list(L2O_SWEEP_EVAL_SEEDS)} steps={int(L2O_SWEEP_EVAL_STEPS)})",
        "",
        "## Top MLP",
    ]
    for r in top_mlp[:10]:
        summary_lines.append(f"- {r['name']}: val_score={float(r['val_score']):.6f}  path={r['path']}")
    summary_lines.append("")
    summary_lines.append("## Top GNN")
    for r in top_gnn[:10]:
        summary_lines.append(f"- {r['name']}: val_score={float(r['val_score']):.6f}  path={r['path']}")
    (sweep_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")

    if L2O_SWEEP_RETRAIN_FINAL and best_mlp is not None:
        meta = load_params_npz(Path(str(best_mlp["path"])))[1]
        mlp_meta = dict(meta)
        mlp_params, mlp_loss = train_l2o_model_safe(
            seed=int(mlp_meta.get("seed", 1)),
            n_list=TRAIN_N_LIST,
            batch=BATCH,
            train_steps=L2O_FINAL_TRAIN_STEPS,
            steps=L2O_FINAL_ROLLOUT_STEPS,
            lr=float(mlp_meta.get("lr", 1e-3)),
            hidden_size=int(mlp_meta.get("hidden", HIDDEN_SIZE)),
            policy="mlp",
            reward=REWARD,
            action_scale=float(mlp_meta.get("action_scale", ACTION_SCALE)),
            mlp_depth=int(mlp_meta.get("mlp_depth", MLP_DEPTH)),
            init_mode=TRAIN_INIT_MODE,
            rand_scale=RAND_SCALE,
            lattice_pattern=TRAIN_LATTICE_PATTERN,
            lattice_margin=TRAIN_LATTICE_MARGIN,
            lattice_rotate=TRAIN_LATTICE_ROTATE,
            baseline_mode=str(mlp_meta.get("baseline_mode", BASELINE_MODE)),
            baseline_decay=float(mlp_meta.get("baseline_decay", BASELINE_DECAY)),
            curriculum=TRAIN_CURRICULUM,
            curriculum_start_max=TRAIN_CURRICULUM_START_MAX,
            curriculum_end_max=TRAIN_CURRICULUM_END_MAX,
            curriculum_steps=TRAIN_CURRICULUM_STEPS,
            feature_mode=str(mlp_meta.get("feature_mode", FEATURE_MODE)),
            overlap_lambda=float(mlp_meta.get("overlap_lambda", 0.0)),
            verbose_freq=10,
        )
    else:
        mlp_params, mlp_loss = train_l2o_model_safe(
            seed=1,
            n_list=TRAIN_N_LIST,
            batch=BATCH,
            train_steps=TRAIN_STEPS,
            steps=ROLLOUT_STEPS,
            hidden_size=HIDDEN_SIZE,
            policy="mlp",
            reward=REWARD,
            action_scale=ACTION_SCALE,
            mlp_depth=MLP_DEPTH,
            gnn_steps=GNN_STEPS,
            gnn_attention=GNN_ATTENTION,
            init_mode=TRAIN_INIT_MODE,
            rand_scale=RAND_SCALE,
            lattice_pattern=TRAIN_LATTICE_PATTERN,
            lattice_margin=TRAIN_LATTICE_MARGIN,
            lattice_rotate=TRAIN_LATTICE_ROTATE,
            baseline_mode=BASELINE_MODE,
            baseline_decay=BASELINE_DECAY,
            curriculum=TRAIN_CURRICULUM,
            curriculum_start_max=TRAIN_CURRICULUM_START_MAX,
            curriculum_end_max=TRAIN_CURRICULUM_END_MAX,
            curriculum_steps=TRAIN_CURRICULUM_STEPS,
            feature_mode=FEATURE_MODE,
            verbose_freq=10,
        )

    if L2O_SWEEP_RETRAIN_FINAL and best_gnn is not None:
        meta = load_params_npz(Path(str(best_gnn["path"])))[1]
        gnn_meta = dict(meta)
        gnn_params, gnn_loss = train_l2o_model_safe(
            seed=int(gnn_meta.get("seed", 2)),
            n_list=TRAIN_N_LIST,
            batch=BATCH,
            train_steps=L2O_FINAL_TRAIN_STEPS,
            steps=L2O_FINAL_ROLLOUT_STEPS,
            lr=float(gnn_meta.get("lr", 1e-3)),
            hidden_size=int(gnn_meta.get("hidden", HIDDEN_SIZE)),
            policy="gnn",
            reward=REWARD,
            action_scale=float(gnn_meta.get("action_scale", ACTION_SCALE)),
            knn_k=int(gnn_meta.get("knn_k", 4)),
            mlp_depth=int(gnn_meta.get("mlp_depth", MLP_DEPTH)),
            gnn_steps=int(gnn_meta.get("gnn_steps", GNN_STEPS)),
            gnn_attention=bool(gnn_meta.get("gnn_attention", False)),
            init_mode=TRAIN_INIT_MODE,
            rand_scale=RAND_SCALE,
            lattice_pattern=TRAIN_LATTICE_PATTERN,
            lattice_margin=TRAIN_LATTICE_MARGIN,
            lattice_rotate=TRAIN_LATTICE_ROTATE,
            baseline_mode=str(gnn_meta.get("baseline_mode", BASELINE_MODE)),
            baseline_decay=float(gnn_meta.get("baseline_decay", BASELINE_DECAY)),
            curriculum=TRAIN_CURRICULUM,
            curriculum_start_max=TRAIN_CURRICULUM_START_MAX,
            curriculum_end_max=TRAIN_CURRICULUM_END_MAX,
            curriculum_steps=TRAIN_CURRICULUM_STEPS,
            feature_mode=str(gnn_meta.get("feature_mode", FEATURE_MODE)),
            overlap_lambda=float(gnn_meta.get("overlap_lambda", 0.0)),
            verbose_freq=10,
        )
    else:
        gnn_params, gnn_loss = train_l2o_model_safe(
            seed=2,
            n_list=TRAIN_N_LIST,
            batch=BATCH,
            train_steps=TRAIN_STEPS,
            steps=ROLLOUT_STEPS,
            hidden_size=HIDDEN_SIZE,
            policy="gnn",
            reward=REWARD,
            action_scale=ACTION_SCALE,
            knn_k=4,
            mlp_depth=MLP_DEPTH,
            gnn_steps=GNN_STEPS,
            gnn_attention=GNN_ATTENTION,
            init_mode=TRAIN_INIT_MODE,
            rand_scale=RAND_SCALE,
            lattice_pattern=TRAIN_LATTICE_PATTERN,
            lattice_margin=TRAIN_LATTICE_MARGIN,
            lattice_rotate=TRAIN_LATTICE_ROTATE,
            baseline_mode=BASELINE_MODE,
            baseline_decay=BASELINE_DECAY,
            curriculum=TRAIN_CURRICULUM,
            curriculum_start_max=TRAIN_CURRICULUM_START_MAX,
            curriculum_end_max=TRAIN_CURRICULUM_END_MAX,
            curriculum_steps=TRAIN_CURRICULUM_STEPS,
            feature_mode=FEATURE_MODE,
            verbose_freq=10,
        )
else:
    # Baseline (sem sweep)
    mlp_params, mlp_loss = train_l2o_model_safe(
        seed=1,
        n_list=TRAIN_N_LIST,
        batch=BATCH,
        train_steps=TRAIN_STEPS,
        steps=ROLLOUT_STEPS,
        hidden_size=HIDDEN_SIZE,
        policy="mlp",
        reward=REWARD,
        action_scale=ACTION_SCALE,
        mlp_depth=MLP_DEPTH,
        gnn_steps=GNN_STEPS,
        gnn_attention=GNN_ATTENTION,
        init_mode=TRAIN_INIT_MODE,
        rand_scale=RAND_SCALE,
        lattice_pattern=TRAIN_LATTICE_PATTERN,
        lattice_margin=TRAIN_LATTICE_MARGIN,
        lattice_rotate=TRAIN_LATTICE_ROTATE,
        baseline_mode=BASELINE_MODE,
        baseline_decay=BASELINE_DECAY,
        curriculum=TRAIN_CURRICULUM,
        curriculum_start_max=TRAIN_CURRICULUM_START_MAX,
        curriculum_end_max=TRAIN_CURRICULUM_END_MAX,
        curriculum_steps=TRAIN_CURRICULUM_STEPS,
        feature_mode=FEATURE_MODE,
        verbose_freq=10,
    )

    gnn_params, gnn_loss = train_l2o_model_safe(
        seed=2,
        n_list=TRAIN_N_LIST,
        batch=BATCH,
        train_steps=TRAIN_STEPS,
        steps=ROLLOUT_STEPS,
        hidden_size=HIDDEN_SIZE,
        policy="gnn",
        reward=REWARD,
        action_scale=ACTION_SCALE,
        knn_k=4,
        mlp_depth=MLP_DEPTH,
        gnn_steps=GNN_STEPS,
        gnn_attention=GNN_ATTENTION,
        init_mode=TRAIN_INIT_MODE,
        rand_scale=RAND_SCALE,
        lattice_pattern=TRAIN_LATTICE_PATTERN,
        lattice_margin=TRAIN_LATTICE_MARGIN,
        lattice_rotate=TRAIN_LATTICE_ROTATE,
        baseline_mode=BASELINE_MODE,
        baseline_decay=BASELINE_DECAY,
        curriculum=TRAIN_CURRICULUM,
        curriculum_start_max=TRAIN_CURRICULUM_START_MAX,
        curriculum_end_max=TRAIN_CURRICULUM_END_MAX,
        curriculum_steps=TRAIN_CURRICULUM_STEPS,
        feature_mode=FEATURE_MODE,
        verbose_freq=10,
    )

# %% Plot losses
plt.figure(figsize=(6, 4))
plt.plot(mlp_loss, label="MLP")
plt.plot(gnn_loss, label="GNN")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("L2O Loss")
plt.legend()
plt.tight_layout()
plt.savefig(RUN_DIR / "loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Visualizacao de packing (MLP)
key = jax.random.PRNGKey(0)
config = l2o_config_from_meta(mlp_meta, reward=REWARD, deterministic=True)
mlp_poses = optimize_with_l2o(key, mlp_params, jnp.array(init), ROLLOUT_STEPS, config)
plot_packing(np.array(mlp_poses), "MLP packing", RUN_DIR / "mlp_packing.png")

# %% Visualizacao de packing (GNN)
key = jax.random.PRNGKey(1)
config = l2o_config_from_meta(gnn_meta, reward=REWARD, deterministic=True)
gnn_poses = optimize_with_l2o(key, gnn_params, jnp.array(init), ROLLOUT_STEPS, config)
plot_packing(np.array(gnn_poses), "GNN packing", RUN_DIR / "gnn_packing.png")

# %% Pipelines opcionais (BC / meta / heatmap)
bc_policy_path: Path | None = Path(BC_POLICY_PATH) if BC_POLICY_PATH else None

# %% Pipeline opcional: BC (imitation learning)
if RUN_BC_PIPELINE:
    bc_dataset = Path(BC_DATASET_PATH) if BC_DATASET_PATH else RUN_DIR / "bc_dataset.npz"
    bc_policy_path = Path(BC_POLICY_PATH) if BC_POLICY_PATH else RUN_DIR / "bc_policy.npz"
    run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "data" / "collect_sa_dataset.py"),
            "--n-list",
            ",".join(str(n) for n in TRAIN_N_LIST),
            "--runs-per-n",
            str(BC_RUNS_PER_N),
            "--steps",
            str(BC_STEPS),
            "--seed",
            str(BC_SEED),
            "--init",
            BC_INIT_MODE,
            "--rand-scale",
            str(BC_RAND_SCALE),
            "--lattice-pattern",
            BC_LATTICE_PATTERN,
            "--lattice-margin",
            str(BC_LATTICE_MARGIN),
            "--lattice-rotate",
            str(BC_LATTICE_ROTATE),
            "--out",
            str(bc_dataset),
        ]
    )
    train_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "training" / "train_l2o_bc.py"),
        "--dataset",
        str(bc_dataset),
        "--policy",
        BC_POLICY,
        "--knn-k",
        str(BC_KNN_K),
        "--train-steps",
        str(BC_TRAIN_STEPS),
        "--seed",
        str(BC_SEED + 1),
        "--reward",
        REWARD,
        "--hidden",
        str(HIDDEN_SIZE),
        "--mlp-depth",
        str(MLP_DEPTH),
        "--gnn-steps",
        str(GNN_STEPS),
        "--feature-mode",
        FEATURE_MODE,
        "--out",
        str(bc_policy_path),
    ]
    if BC_CURRICULUM:
        train_cmd.append("--curriculum")
    if BC_CURRICULUM_START_MAX is not None:
        train_cmd += ["--curriculum-start-max", str(int(BC_CURRICULUM_START_MAX))]
    if BC_CURRICULUM_END_MAX is not None:
        train_cmd += ["--curriculum-end-max", str(int(BC_CURRICULUM_END_MAX))]
    if BC_CURRICULUM_STEPS is not None:
        train_cmd += ["--curriculum-steps", str(int(BC_CURRICULUM_STEPS))]
    if GNN_ATTENTION:
        train_cmd.append("--gnn-attention")
    run_cmd(train_cmd)

# %% Pipeline opcional: meta-init
meta_init_path: Path | None = Path(META_INIT_MODEL_PATH) if META_INIT_MODEL_PATH else None
if RUN_META_TRAIN:
    meta_init_path = Path(META_INIT_MODEL_PATH) if META_INIT_MODEL_PATH else RUN_DIR / "meta_init.npz"
    run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "training" / "train_meta_init.py"),
            "--n-list",
            ",".join(str(n) for n in TRAIN_N_LIST),
            "--train-steps",
            str(META_TRAIN_STEPS),
            "--es-pop",
            str(META_ES_POP),
            "--sa-steps",
            str(META_SA_STEPS),
            "--out",
            str(meta_init_path),
        ]
    )

# %% Pipeline opcional: heatmap
heatmap_path: Path | None = Path(HEATMAP_MODEL_PATH) if HEATMAP_MODEL_PATH else None
if RUN_HEATMAP_TRAIN:
    heatmap_path = Path(HEATMAP_MODEL_PATH) if HEATMAP_MODEL_PATH else RUN_DIR / "heatmap_meta.npz"
    run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "training" / "train_heatmap_meta.py"),
            "--n-list",
            ",".join(str(n) for n in TRAIN_N_LIST),
            "--train-steps",
            str(HEATMAP_TRAIN_STEPS),
            "--es-pop",
            str(HEATMAP_ES_POP),
            "--heatmap-steps",
            str(HEATMAP_STEPS),
            "--policy",
            "gnn",
            "--knn-k",
            str(BC_KNN_K),
            "--out",
            str(heatmap_path),
        ]
    )

# %% Solvers e configuracao de avaliacao
if set(TRAIN_N_LIST) & set(VAL_N_LIST):
    raise ValueError("TRAIN_N_LIST and VAL_N_LIST must be disjoint to avoid leakage.")

# %% Modelos opcionais carregados de disco
bc_params = None
bc_config = None
if bc_policy_path is not None and bc_policy_path.exists():
    bc_params, bc_meta = load_params_npz(bc_policy_path)
    bc_config = l2o_config_from_meta(bc_meta, reward=REWARD, deterministic=True)

def solve_meta_init_sa(n: int, seed: int) -> np.ndarray:
    if meta_init_path is None or not meta_init_path.exists():
        return solve_sa(n, seed)
    try:
        from santa_packing.meta_init import MetaInitConfig, apply_meta_init, load_meta_params  # noqa: E402
    except Exception:
        return solve_sa(n, seed)
    params, meta = load_meta_params(meta_init_path)
    config = MetaInitConfig(
        hidden_size=int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32,
        delta_xy=float(meta.get("delta_xy", 0.2)),
        delta_theta=float(meta.get("delta_theta", 10.0)),
    )
    init_pose = get_initial(n, seed)
    init_pose = np.array(apply_meta_init(params, jnp.array(init_pose), config))
    init_batch = jnp.array(init_pose)[None, :, :]
    key = jax.random.PRNGKey(seed)
    best_poses, _ = run_sa_batch(
        key,
        SA_STEPS,
        n,
        init_batch,
        trans_sigma=SA_TRANS_SIGMA,
        rot_sigma=SA_ROT_SIGMA,
        rot_prob=SA_ROT_PROB,
        objective=SA_OBJECTIVE,
    )
    return np.array(best_poses[0])


def solve_heatmap(n: int, seed: int) -> np.ndarray:
    if heatmap_path is None or not heatmap_path.exists():
        return solve_grid(n, seed)
    try:
        from santa_packing.heatmap_meta import HeatmapConfig, heatmap_search, load_params  # noqa: E402
    except Exception:
        return solve_grid(n, seed)
    params, meta = load_params(heatmap_path)
    config = HeatmapConfig(
        hidden_size=int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32,
        policy=str(meta.get("policy", "gnn")),
        knn_k=int(meta.get("knn_k", 4)) if hasattr(meta.get("knn_k", 4), "__int__") else 4,
        heatmap_lr=float(meta.get("heatmap_lr", 0.1)),
        trans_sigma=float(meta.get("trans_sigma", 0.2)),
        rot_sigma=float(meta.get("rot_sigma", 10.0)),
    )
    base = get_initial(n, seed)
    rng = np.random.default_rng(seed)
    poses, _ = heatmap_search(params, base, config, HEATMAP_STEPS, rng)
    return np.array(poses)


def solve_ensemble(candidates: Dict[str, Callable[[int, int], np.ndarray]]) -> Callable[[int, int], Tuple[np.ndarray, Dict[str, str]]]:
    def _solve(n: int, seed: int) -> Tuple[np.ndarray, Dict[str, str]]:
        best_score = float("inf")
        best_pose: np.ndarray | None = None
        best_name = "none"
        for name, fn in candidates.items():
            poses = fn(n, seed)
            if ENSEMBLE_SCORE == "packing":
                score = packing_score(points, poses)
            else:
                score = prefix_packing_score_np(points, poses)
            if score < best_score:
                best_score = float(score)
                best_pose = poses
                best_name = name
        if best_pose is None:
            best_pose = candidates[next(iter(candidates))](n, seed)
        return best_pose, {"selected": best_name}

    return _solve


# %% Rodar avaliacao (configs)
results: List[Dict[str, float]] = []
l2o_mlp_cfg = l2o_config_from_meta(mlp_meta, reward=REWARD, deterministic=True)
l2o_gnn_cfg = l2o_config_from_meta(gnn_meta, reward=REWARD, deterministic=True)

# %% Avaliacao: baselines
results += evaluate_solver("grid", solve_grid, TRAIN_N_LIST, TRAIN_EVAL_SEEDS, points, split="train")
results += evaluate_solver("grid", solve_grid, VAL_N_LIST, VAL_EVAL_SEEDS, points, split="val")
results += evaluate_solver("sa", solve_sa, TRAIN_N_LIST, TRAIN_EVAL_SEEDS, points, split="train")
results += evaluate_solver("sa", solve_sa, VAL_N_LIST, VAL_EVAL_SEEDS, points, split="val")
print_challenge_scores(results, split="train", models=["grid", "sa"], title="[eval] baselines (Kaggle metric on subset):")
print_challenge_scores(results, split="val", models=["grid", "sa"])

# %% Avaliacao: L2O
results += evaluate_solver(
    "l2o_mlp",
    solve_l2o(mlp_params, l2o_mlp_cfg),
    TRAIN_N_LIST,
    TRAIN_EVAL_SEEDS,
    points,
    split="train",
)
results += evaluate_solver(
    "l2o_gnn",
    solve_l2o(gnn_params, l2o_gnn_cfg),
    TRAIN_N_LIST,
    TRAIN_EVAL_SEEDS,
    points,
    split="train",
)
results += evaluate_solver(
    "l2o_mlp",
    solve_l2o(mlp_params, l2o_mlp_cfg),
    VAL_N_LIST,
    VAL_EVAL_SEEDS,
    points,
    split="val",
)
results += evaluate_solver(
    "l2o_gnn",
    solve_l2o(gnn_params, l2o_gnn_cfg),
    VAL_N_LIST,
    VAL_EVAL_SEEDS,
    points,
    split="val",
)
print_challenge_scores(results, split="train", models=["l2o_mlp", "l2o_gnn"], title="[eval] l2o (Kaggle metric on subset):")
print_challenge_scores(results, split="val", models=["l2o_mlp", "l2o_gnn"])

# %% Avaliacao: modelos opcionais
if bc_params is not None and bc_config is not None:
    results += evaluate_solver(
        "l2o_bc",
        solve_l2o(bc_params, bc_config),
        TRAIN_N_LIST,
        TRAIN_EVAL_SEEDS,
        points,
        split="train",
    )
    results += evaluate_solver(
        "l2o_bc",
        solve_l2o(bc_params, bc_config),
        VAL_N_LIST,
        VAL_EVAL_SEEDS,
        points,
        split="val",
    )

if meta_init_path is not None and meta_init_path.exists():
    results += evaluate_solver(
        "sa_meta_init",
        solve_meta_init_sa,
        TRAIN_N_LIST,
        TRAIN_EVAL_SEEDS,
        points,
        split="train",
    )
    results += evaluate_solver(
        "sa_meta_init",
        solve_meta_init_sa,
        VAL_N_LIST,
        VAL_EVAL_SEEDS,
        points,
        split="val",
    )

if heatmap_path is not None and heatmap_path.exists():
    results += evaluate_solver(
        "heatmap",
        solve_heatmap,
        TRAIN_N_LIST,
        TRAIN_EVAL_SEEDS,
        points,
        split="train",
    )
    results += evaluate_solver(
        "heatmap",
        solve_heatmap,
        VAL_N_LIST,
        VAL_EVAL_SEEDS,
        points,
        split="val",
    )

if L2O_REFINE_GRID:
    results += evaluate_solver(
        "l2o_refine_grid",
        solve_l2o_refine(solve_grid, mlp_params, l2o_mlp_cfg),
        TRAIN_N_LIST,
        TRAIN_EVAL_SEEDS,
        points,
        split="train",
    )
    results += evaluate_solver(
        "l2o_refine_grid",
        solve_l2o_refine(solve_grid, mlp_params, l2o_mlp_cfg),
        VAL_N_LIST,
        VAL_EVAL_SEEDS,
        points,
        split="val",
    )

if L2O_REFINE_SA:
    results += evaluate_solver(
        "l2o_refine_sa",
        solve_l2o_refine(solve_sa, gnn_params, l2o_gnn_cfg),
        TRAIN_N_LIST,
        TRAIN_EVAL_SEEDS,
        points,
        split="train",
    )
    results += evaluate_solver(
        "l2o_refine_sa",
        solve_l2o_refine(solve_sa, gnn_params, l2o_gnn_cfg),
        VAL_N_LIST,
        VAL_EVAL_SEEDS,
        points,
        split="val",
    )

print_challenge_scores(
    results,
    split="train",
    models=["l2o_bc", "sa_meta_init", "heatmap", "l2o_refine_grid", "l2o_refine_sa"],
    title="[eval] optional models (Kaggle metric on subset):",
)
print_challenge_scores(results, split="val", models=["l2o_bc", "sa_meta_init", "heatmap", "l2o_refine_grid", "l2o_refine_sa"])

# %% Avaliacao: ensemble
if RUN_ENSEMBLE:
    ensemble_candidates = {
        "grid": solve_grid,
        "sa": solve_sa,
        "l2o_mlp": solve_l2o(mlp_params, l2o_mlp_cfg),
        "l2o_gnn": solve_l2o(gnn_params, l2o_gnn_cfg),
    }
    if bc_params is not None and bc_config is not None:
        ensemble_candidates["l2o_bc"] = solve_l2o(bc_params, bc_config)
    if meta_init_path is not None and meta_init_path.exists():
        ensemble_candidates["sa_meta_init"] = solve_meta_init_sa
    if heatmap_path is not None and heatmap_path.exists():
        ensemble_candidates["heatmap"] = solve_heatmap
    results += evaluate_solver(
        "ensemble",
        solve_ensemble(ensemble_candidates),
        TRAIN_N_LIST,
        TRAIN_EVAL_SEEDS,
        points,
        split="train",
    )
    results += evaluate_solver(
        "ensemble",
        solve_ensemble(ensemble_candidates),
        VAL_N_LIST,
        VAL_EVAL_SEEDS,
        points,
        split="val",
    )
    print_challenge_scores(results, split="train", models=["ensemble"], title="[eval] ensemble (Kaggle metric on subset):")
    print_challenge_scores(results, split="val", models=["ensemble"])

# %% Resumos e artifacts
per_n = summarize_results(results)
overall = summarize_overall(results)

meta = {
    "reward": REWARD,
    "train_n_list": TRAIN_N_LIST,
    "val_n_list": VAL_N_LIST,
    "train_eval_seeds": TRAIN_EVAL_SEEDS,
    "val_eval_seeds": VAL_EVAL_SEEDS,
    "train_steps": TRAIN_STEPS,
    "rollout_steps": ROLLOUT_STEPS,
    "batch": BATCH,
    "eval_steps": EVAL_STEPS,
    "hidden_size": HIDDEN_SIZE,
    "mlp_depth": MLP_DEPTH,
    "gnn_steps": GNN_STEPS,
    "gnn_attention": GNN_ATTENTION,
    "action_scale": ACTION_SCALE,
    "feature_mode": FEATURE_MODE,
    "baseline_mode": BASELINE_MODE,
    "baseline_decay": BASELINE_DECAY,
    "init_mode": INIT_MODE,
    "rand_scale": RAND_SCALE,
    "sa_steps": SA_STEPS,
    "sa_trans_sigma": SA_TRANS_SIGMA,
    "sa_rot_sigma": SA_ROT_SIGMA,
    "sa_rot_prob": SA_ROT_PROB,
    "sa_objective": SA_OBJECTIVE,
    "run_bc_pipeline": RUN_BC_PIPELINE,
    "bc_policy": BC_POLICY,
    "bc_runs_per_n": BC_RUNS_PER_N,
    "bc_steps": BC_STEPS,
    "bc_train_steps": BC_TRAIN_STEPS,
    "bc_policy_path": str(bc_policy_path) if bc_policy_path else None,
    "run_meta_train": RUN_META_TRAIN,
    "meta_init_path": str(meta_init_path) if meta_init_path else None,
    "run_heatmap_train": RUN_HEATMAP_TRAIN,
    "heatmap_path": str(heatmap_path) if heatmap_path else None,
    "run_ensemble": RUN_ENSEMBLE,
    "ensemble_score": ENSEMBLE_SCORE,
    "l2o_refine_grid": L2O_REFINE_GRID,
    "l2o_refine_sa": L2O_REFINE_SA,
    "refine_steps": REFINE_STEPS,
}

save_eval_artifacts(RUN_DIR, results, per_n, overall, meta)
plot_eval_curves(per_n, RUN_DIR / "eval_curve.png")

# %% Scores (Kaggle metric on eval subset)
# Este score usa a mesma fórmula do Kaggle (sum(s_n^2 / n)), mas calculada apenas
# no subset avaliado (TRAIN_N_LIST/VAL_N_LIST e seeds associados).
challenge_scores = {
    "train": print_challenge_scores(results, split="train", title="[eval] Kaggle metric on subset:"),
    "val": print_challenge_scores(results, split="val"),
}
(RUN_DIR / "challenge_scores.json").write_text(json.dumps(challenge_scores, indent=2), encoding="utf-8")
print("Eval artifacts saved to", RUN_DIR)

# %% Gerar submission.csv (Kaggle)
SUBMISSION_NMAX = 200
SUBMISSION_SEED = 1
SUBMISSION_OVERLAP_CHECK = True  # para score final, mantenha True

RUN_SUBMISSION_SWEEP = False  # True = gera/score varias receitas + seeds
SWEEP_NMAX = 50  # use 200 para score final
SWEEP_SEEDS: List[int] | str = "1..4"
SWEEP_SCORE_OVERLAP_CHECK = False  # durante sweep rapido, pode ser False; no final use True
SWEEP_BUILD_ENSEMBLE = True
SWEEP_JOBS = max(1, int(os.cpu_count() or 1))
SWEEP_REUSE = True
SWEEP_KEEP_GOING = True

# Quanto maior, mais combinacoes (receitas) entram no sweep.
# Use `None` para "maximo" (pode demorar bastante).
RECIPES_MAX_RECIPES_PER_FAMILY = None
RECIPES_MAX_LATTICE_VARIANTS = None

# %% Salvar politicas treinadas (para usar no generate_submission/guided SA)
# Salvar as politicas treinadas neste notebook (para usar no generate_submission/guided SA)
MLP_POLICY_PATH = RUN_DIR / "l2o_mlp.npz"
GNN_POLICY_PATH = RUN_DIR / "l2o_gnn.npz"
save_params_npz(
    MLP_POLICY_PATH,
    mlp_params,
    meta={**mlp_meta, "reward": REWARD},
)
save_params_npz(
    GNN_POLICY_PATH,
    gnn_params,
    meta={**gnn_meta, "reward": REWARD},
)


# %% Submission helpers (ensemble por puzzle)
def _best_per_puzzle_ensemble(
    out_csv: Path,
    candidates: Dict[str, Path],
    *,
    nmax: int,
    check_overlap: bool,
) -> Dict[str, object]:
    try:
        from santa_packing.scoring import load_submission  # noqa: E402
    except Exception as exc:
        raise RuntimeError("Failed to import scoring.load_submission") from exc
    try:
        from santa_packing.postopt_np import has_overlaps  # noqa: E402
    except Exception as exc:
        raise RuntimeError("Failed to import postopt_np.has_overlaps") from exc

    points = np.array(TREE_POINTS, dtype=float)
    loaded = {name: load_submission(path, nmax=nmax) for name, path in candidates.items()}

    selected: Dict[int, str] = {}
    best_poses: Dict[int, np.ndarray] = {}
    for n in range(1, nmax + 1):
        best_s = float("inf")
        best_name = None
        best_pose = None
        for name, puzzles in loaded.items():
            poses = puzzles.get(n)
            if poses is None or poses.shape[0] != n:
                continue
            poses = np.array(poses, dtype=float, copy=True)
            poses[:, 2] = np.mod(poses[:, 2], 360.0)
            poses = shift_poses_to_origin(points, poses)
            if check_overlap and has_overlaps(points, poses):
                continue
            s = float(packing_score(points, poses))
            if s < best_s:
                best_s = s
                best_name = name
                best_pose = poses
        if best_name is None or best_pose is None:
            raise ValueError(f"No feasible candidates for puzzle {n} (check_overlap={check_overlap})")
        selected[n] = best_name
        best_poses[n] = np.array(best_pose, dtype=float)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])
        for n in range(1, nmax + 1):
            poses = np.array(best_poses[n], dtype=float, copy=True)
            poses[:, 2] = np.mod(poses[:, 2], 360.0)
            poses = shift_poses_to_origin(points, poses)
            for i, (x, y, deg) in enumerate(poses):
                writer.writerow([f"{n:03d}_{i}", f"s{float(x):.17f}", f"s{float(y):.17f}", f"s{float(deg):.17f}"])

    return {"selected_by_puzzle": {str(k): v for k, v in selected.items()}}

# %% Modelos disponiveis (paths)
META_INIT_MODEL = str(meta_init_path) if (meta_init_path is not None and meta_init_path.exists()) else None
HEATMAP_MODEL = str(heatmap_path) if (heatmap_path is not None and heatmap_path.exists()) else None

# L2O models trained in this run (plus optional BC policy).
CANDIDATE_L2O_MODELS: Dict[str, Path] = {
    "reinforce_gnn": GNN_POLICY_PATH,
    "reinforce_mlp": MLP_POLICY_PATH,
}
if bc_policy_path is not None and bc_policy_path.exists():
    CANDIDATE_L2O_MODELS["bc"] = bc_policy_path

# Opcional: inclui os melhores modelos do sweep (rapidos) para aumentar o portfolio.
for name, path in sorted(L2O_SWEEP_TOP_MODELS.items()):
    if path.exists():
        CANDIDATE_L2O_MODELS[f"sweep_{name}"] = path
CANDIDATE_GUIDED_MODELS: Dict[str, Path] = dict(CANDIDATE_L2O_MODELS)


# %% [markdown]
# ## Experimentos de submission (1 célula = 1 experimento)
#
# Objetivo: cada célula abaixo gera um `submission.csv` (via `python -m santa_packing.cli.generate_submission`)
# e já imprime o **score do desafio** (métrica do Kaggle) calculado pelo scorer local
# (`santa_packing/scoring.py`, equivalente ao `python -m santa_packing.cli.score_submission`).
#
# Dica: deixe `REUSE=True` para não re-gerar CSVs já existentes.
# %%
SUBMISSION_NMAX = 200
SUBMISSION_SEED = 1
SUBMISSION_OVERLAP_CHECK = True
REUSE = True

SUBMISSION_CELL_DIR = RUN_DIR / "submission_cells"
SUBMISSION_CELL_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_name(name: str) -> str:
    out = []
    for ch in str(name).strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("._-")
    return cleaned or "exp"


def _path_exists(value: object) -> bool:
    if value is None:
        return False
    try:
        p = Path(str(value))
    except Exception:
        return False
    return p.exists()


def _check_required_models(args: Dict[str, object]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for k, v in args.items():
        if v is None:
            continue
        if k.endswith("_model") or k == "guided_model":
            if not _path_exists(v):
                missing.append(k)
    return (len(missing) == 0), missing


def run_submission_experiment(
    name: str,
    args: Dict[str, object],
    *,
    seed: int = SUBMISSION_SEED,
    nmax: int = SUBMISSION_NMAX,
    check_overlap: bool = SUBMISSION_OVERLAP_CHECK,
    reuse: bool = REUSE,
) -> Dict[str, object] | None:
    exp = _sanitize_name(name)
    ok, missing = _check_required_models(args)
    if not ok:
        print(f"[skip] {exp} (modelos ausentes: {missing})")
        return None

    out_csv = SUBMISSION_CELL_DIR / f"{exp}_seed{int(seed)}_n{int(nmax)}.csv"
    out_score = SUBMISSION_CELL_DIR / f"{exp}_seed{int(seed)}_n{int(nmax)}.score.json"
    out_meta = SUBMISSION_CELL_DIR / f"{exp}_seed{int(seed)}_n{int(nmax)}.meta.json"

    if not (reuse and out_csv.exists()):
        generate_submission(out_csv, seed=int(seed), nmax=int(nmax), args=args)

    score = score_csv(out_csv, nmax=int(nmax), check_overlap=bool(check_overlap))
    out_score.write_text(json.dumps(score, indent=2), encoding="utf-8")
    out_meta.write_text(
        json.dumps(
            {
                "name": name,
                "exp": exp,
                "seed": int(seed),
                "nmax": int(nmax),
                "check_overlap": bool(check_overlap),
                "csv": str(out_csv),
                "args": args,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    val = score.get("score")
    s_max = score.get("s_max")
    print(f"[{exp}] score={val}  s_max={s_max}  csv={out_csv.name}")
    return score


# Base "vazio": lattice como fallback. Cada experimento sobrescreve o que precisa.
BASE_ARGS: Dict[str, object] = {
    "mother_prefix": False,
    "mother_reorder": "radial",
    "lattice_pattern": "hex",
    "lattice_margin": 0.02,
    "lattice_rotate": 0.0,
    "lattice_rotations": "0,15,30",
    "sa_nmax": 0,
    "sa_batch": 64,
    "sa_steps": 400,
    "sa_trans_sigma": 0.2,
    "sa_rot_sigma": 15.0,
    "sa_rot_prob": 0.3,
    "sa_rot_prob_end": -1.0,
    "sa_swap_prob": 0.0,
    "sa_swap_prob_end": -1.0,
    "sa_cooling": "geom",
    "sa_cooling_power": 1.0,
    "sa_trans_nexp": 0.0,
    "sa_rot_nexp": 0.0,
    "sa_sigma_nref": 50.0,
    "sa_objective": "packing",
    "sa_proposal": "random",
    "sa_smart_prob": 1.0,
    "sa_smart_beta": 8.0,
    "sa_smart_drift": 1.0,
    "sa_smart_noise": 0.25,
    "sa_overlap_lambda": 0.0,
    "sa_allow_collisions": False,
    "meta_init_model": None,
    "heatmap_model": None,
    "heatmap_nmax": 0,
    "heatmap_steps": 200,
    "l2o_model": None,
    "l2o_init": "lattice",
    "l2o_nmax": 0,
    "l2o_steps": 250,
    "l2o_trans_sigma": 0.2,
    "l2o_rot_sigma": 10.0,
    "l2o_deterministic": True,
    "refine_nmin": 0,
    "refine_batch": 16,
    "refine_steps": 0,
    "refine_trans_sigma": 0.2,
    "refine_rot_sigma": 15.0,
    "refine_rot_prob": 0.3,
    "refine_rot_prob_end": -1.0,
    "refine_swap_prob": 0.0,
    "refine_swap_prob_end": -1.0,
    "refine_cooling": "geom",
    "refine_cooling_power": 1.0,
    "refine_trans_nexp": 0.0,
    "refine_rot_nexp": 0.0,
    "refine_sigma_nref": 50.0,
    "refine_objective": "packing",
    "refine_proposal": "random",
    "refine_smart_prob": 1.0,
    "refine_smart_beta": 8.0,
    "refine_smart_drift": 1.0,
    "refine_smart_noise": 0.25,
    "refine_overlap_lambda": 0.0,
    "refine_allow_collisions": False,
    "lns_nmax": 0,
    "lns_passes": 0,
    "lns_destroy_k": 8,
    "lns_destroy_mode": "mixed",
    "lns_tabu_tenure": 0,
    "lns_candidates": 64,
    "lns_angle_samples": 8,
    "lns_pad_scale": 2.0,
    "lns_group_moves": 0,
    "lns_group_size": 3,
    "lns_group_trans_sigma": 0.05,
    "lns_group_rot_sigma": 20.0,
    "lns_t_start": 0.0,
    "lns_t_end": 0.0,
    "guided_model": None,
    "guided_prob": 1.0,
    "guided_pmax": 0.05,
    "guided_prob_end": -1.0,
    "guided_pmax_end": -1.0,
    "block_nmax": 0,
    "block_size": 2,
    "block_batch": 32,
    "block_steps": 0,
    "block_trans_sigma": 0.2,
    "block_rot_sigma": 15.0,
    "block_rot_prob": 0.25,
    "block_rot_prob_end": -1.0,
    "block_cooling": "geom",
    "block_cooling_power": 1.0,
    "block_trans_nexp": 0.0,
    "block_rot_nexp": 0.0,
    "block_sigma_nref": 50.0,
    "block_overlap_lambda": 0.0,
    "block_allow_collisions": False,
    "block_objective": "packing",
    "block_init": "cluster",
    "block_template_pattern": "hex",
    "block_template_margin": 0.02,
    "block_template_rotate": 0.0,
    "hc_nmax": 0,
    "hc_passes": 0,
    "hc_step_xy": 0.01,
    "hc_step_deg": 2.0,
    "ga_nmax": 0,
    "ga_pop": 24,
    "ga_gens": 0,
    "ga_elite_frac": 0.25,
    "ga_crossover_prob": 0.5,
    "ga_mut_sigma_xy": 0.01,
    "ga_mut_sigma_deg": 2.0,
    "ga_directed_prob": 0.5,
    "ga_directed_step_xy": 0.02,
    "ga_directed_k": 8,
    "ga_repair_iters": 200,
    "ga_hc_passes": 0,
    "ga_hc_step_xy": 0.01,
    "ga_hc_step_deg": 2.0,
}


# %% [markdown]
# ## Baselines (lattice)
# %% (lattice_hex_rots0_15_30)
args = {**BASE_ARGS, "sa_nmax": 0, "lattice_pattern": "hex", "lattice_margin": 0.02, "lattice_rotations": "0,15,30"}
run_submission_experiment("lattice_hex_rots0_15_30", args)

# %% (lattice_hex_rots0_5_10_15_20_25_30)
args = {**BASE_ARGS, "sa_nmax": 0, "lattice_pattern": "hex", "lattice_margin": 0.005, "lattice_rotations": "0,5,10,15,20,25,30"}
run_submission_experiment("lattice_hex_rots0_5_10_15_20_25_30", args)

# %% (lattice_square_rots0_15_30)
args = {**BASE_ARGS, "sa_nmax": 0, "lattice_pattern": "square", "lattice_margin": 0.02, "lattice_rotations": "0,15,30"}
run_submission_experiment("lattice_square_rots0_15_30", args)

# %% (lattice_hex_const_rot15)
args = {**BASE_ARGS, "sa_nmax": 0, "lattice_pattern": "hex", "lattice_margin": 0.02, "lattice_rotations": "none", "lattice_rotate": 15.0}
run_submission_experiment("lattice_hex_const_rot15", args)


# %% [markdown]
# ## SA (Simulated Annealing)
# %% (sa_packing_random_geom)
args = {
    **BASE_ARGS,
    "sa_nmax": 50,
    "sa_steps": 800,
    "sa_batch": 64,
    "sa_objective": "packing",
    "sa_proposal": "random",
    "sa_cooling": "geom",
}
run_submission_experiment("sa_packing_random_geom", args)

# %% (sa_packing_mixed_log)
args = {
    **BASE_ARGS,
    "sa_nmax": 50,
    "sa_steps": 800,
    "sa_batch": 64,
    "sa_objective": "packing",
    "sa_proposal": "mixed",
    "sa_cooling": "log",
    "sa_smart_prob": 0.7,
}
run_submission_experiment("sa_packing_mixed_log", args)

# %% (sa_packing_bbox_inward_geom)
args = {
    **BASE_ARGS,
    "sa_nmax": 50,
    "sa_steps": 800,
    "sa_batch": 64,
    "sa_objective": "packing",
    "sa_proposal": "bbox_inward",
    "sa_cooling": "geom",
}
run_submission_experiment("sa_packing_bbox_inward_geom", args)

# %% (sa_prefix_mixed_log_swap)
args = {
    **BASE_ARGS,
    "sa_nmax": 50,
    "sa_steps": 900,
    "sa_batch": 64,
    "sa_objective": "prefix",
    "sa_proposal": "mixed",
    "sa_cooling": "log",
    "sa_swap_prob": 0.05,
    "sa_swap_prob_end": 0.0,
    "sa_smart_prob": 0.7,
}
run_submission_experiment("sa_prefix_mixed_log_swap", args)

# %% (sa_prefix_smart_log_swap)
args = {
    **BASE_ARGS,
    "sa_nmax": 50,
    "sa_steps": 900,
    "sa_batch": 64,
    "sa_objective": "prefix",
    "sa_proposal": "smart",
    "sa_cooling": "log",
    "sa_swap_prob": 0.05,
    "sa_swap_prob_end": 0.0,
    "sa_smart_prob": 1.0,
}
run_submission_experiment("sa_prefix_smart_log_swap", args)

# %% (sa_prefix_mixed_log_overlap001)
args = {
    **BASE_ARGS,
    "sa_nmax": 50,
    "sa_steps": 900,
    "sa_batch": 64,
    "sa_objective": "prefix",
    "sa_proposal": "mixed",
    "sa_cooling": "log",
    "sa_swap_prob": 0.05,
    "sa_swap_prob_end": 0.0,
    "sa_overlap_lambda": 0.01,
    "sa_smart_prob": 0.7,
}
run_submission_experiment("sa_prefix_mixed_log_overlap001", args)


# %% [markdown]
# ## Refine (SA sobre o lattice/solver base)
# %% (refine_packing_mixed_geom)
args = {
    **BASE_ARGS,
    "refine_nmin": 80,
    "refine_steps": 800,
    "refine_batch": 24,
    "refine_objective": "packing",
    "refine_proposal": "mixed",
    "refine_cooling": "geom",
    "refine_smart_prob": 0.7,
}
run_submission_experiment("refine_packing_mixed_geom", args)

# %% (refine_prefix_mixed_log)
args = {
    **BASE_ARGS,
    "refine_nmin": 80,
    "refine_steps": 900,
    "refine_batch": 24,
    "refine_objective": "prefix",
    "refine_proposal": "mixed",
    "refine_cooling": "log",
    "refine_smart_prob": 0.7,
}
run_submission_experiment("refine_prefix_mixed_log", args)

# %% (refine_prefix_mixed_log_overlap001)
args = {
    **BASE_ARGS,
    "refine_nmin": 80,
    "refine_steps": 900,
    "refine_batch": 24,
    "refine_objective": "prefix",
    "refine_proposal": "mixed",
    "refine_cooling": "log",
    "refine_overlap_lambda": 0.01,
    "refine_smart_prob": 0.7,
}
run_submission_experiment("refine_prefix_mixed_log_overlap001", args)

# %% (refine_prefix_bbox_inward_log)
args = {
    **BASE_ARGS,
    "refine_nmin": 80,
    "refine_steps": 900,
    "refine_batch": 24,
    "refine_objective": "prefix",
    "refine_proposal": "bbox_inward",
    "refine_cooling": "log",
}
run_submission_experiment("refine_prefix_bbox_inward_log", args)


# %% [markdown]
# ## Block SA (meta-model)
# %% (block_cluster_b2_prefix)
args = {
    **BASE_ARGS,
    "block_nmax": 200,
    "block_init": "cluster",
    "block_size": 2,
    "block_steps": 350,
    "block_batch": 32,
    "block_objective": "prefix",
}
run_submission_experiment("block_cluster_b2_prefix", args)

# %% (block_cluster_b3_prefix)
args = {
    **BASE_ARGS,
    "block_nmax": 200,
    "block_init": "cluster",
    "block_size": 3,
    "block_steps": 350,
    "block_batch": 32,
    "block_objective": "prefix",
}
run_submission_experiment("block_cluster_b3_prefix", args)

# %% (block_cluster_b4_prefix)
args = {
    **BASE_ARGS,
    "block_nmax": 200,
    "block_init": "cluster",
    "block_size": 4,
    "block_steps": 350,
    "block_batch": 32,
    "block_objective": "prefix",
}
run_submission_experiment("block_cluster_b4_prefix", args)

# %% (block_template_hex_b2_prefix)
args = {
    **BASE_ARGS,
    "block_nmax": 200,
    "block_init": "template",
    "block_template_pattern": "hex",
    "block_template_margin": 0.02,
    "block_template_rotate": 0.0,
    "block_size": 2,
    "block_steps": 350,
    "block_batch": 32,
    "block_objective": "prefix",
}
run_submission_experiment("block_template_hex_b2_prefix", args)

# %% (block_template_square_b2_prefix)
args = {
    **BASE_ARGS,
    "block_nmax": 200,
    "block_init": "template",
    "block_template_pattern": "square",
    "block_template_margin": 0.02,
    "block_template_rotate": 0.0,
    "block_size": 2,
    "block_steps": 350,
    "block_batch": 32,
    "block_objective": "prefix",
}
run_submission_experiment("block_template_square_b2_prefix", args)


# %% [markdown]
# ## LNS / ALNS (post-opt)
# %% (lns_mixed)
args = {**BASE_ARGS, "lns_nmax": 200, "lns_passes": 10, "lns_destroy_mode": "mixed", "lns_destroy_k": 8, "lns_group_moves": 4}
run_submission_experiment("lns_mixed", args)

# %% (lns_mixed_tabu5)
args = {**BASE_ARGS, "lns_nmax": 200, "lns_passes": 10, "lns_destroy_mode": "mixed", "lns_tabu_tenure": 5, "lns_destroy_k": 8, "lns_group_moves": 4}
run_submission_experiment("lns_mixed_tabu5", args)

# %% (lns_boundary_tabu5)
args = {**BASE_ARGS, "lns_nmax": 200, "lns_passes": 10, "lns_destroy_mode": "boundary", "lns_tabu_tenure": 5, "lns_destroy_k": 10, "lns_group_moves": 4}
run_submission_experiment("lns_boundary_tabu5", args)

# %% (lns_cluster_tabu5)
args = {**BASE_ARGS, "lns_nmax": 200, "lns_passes": 10, "lns_destroy_mode": "cluster", "lns_tabu_tenure": 5, "lns_destroy_k": 10, "lns_group_moves": 4}
run_submission_experiment("lns_cluster_tabu5", args)

# %% (lns_random)
args = {**BASE_ARGS, "lns_nmax": 200, "lns_passes": 10, "lns_destroy_mode": "random", "lns_destroy_k": 8, "lns_group_moves": 4}
run_submission_experiment("lns_random", args)

# %% (lns_alns)
args = {**BASE_ARGS, "lns_nmax": 200, "lns_passes": 10, "lns_destroy_mode": "alns", "lns_destroy_k": 8, "lns_group_moves": 4}
run_submission_experiment("lns_alns", args)

# %% (lns_alns_tabu5)
args = {**BASE_ARGS, "lns_nmax": 200, "lns_passes": 10, "lns_destroy_mode": "alns", "lns_tabu_tenure": 5, "lns_destroy_k": 8, "lns_group_moves": 4}
run_submission_experiment("lns_alns_tabu5", args)

# %% (lns_alns_sa_accept_tabu10)
args = {
    **BASE_ARGS,
    "lns_nmax": 200,
    "lns_passes": 20,
    "lns_destroy_mode": "alns",
    "lns_tabu_tenure": 10,
    "lns_destroy_k": 10,
    "lns_candidates": 96,
    "lns_angle_samples": 12,
    "lns_group_moves": 6,
    "lns_group_size": 4,
    "lns_t_start": 0.2,
    "lns_t_end": 0.02,
}
run_submission_experiment("lns_alns_sa_accept_tabu10", args)


# %% [markdown]
# ## GA / Hill-climb (n pequeno)
# %% (hc20_lattice)
args = {**BASE_ARGS, "hc_nmax": 20, "hc_passes": 2, "hc_step_xy": 0.01, "hc_step_deg": 2.0}
run_submission_experiment("hc20_lattice", args)

# %% (ga20_lattice)
args = {**BASE_ARGS, "ga_nmax": 20, "ga_gens": 20, "ga_pop": 24, "ga_mut_sigma_xy": 0.01, "ga_mut_sigma_deg": 2.0}
run_submission_experiment("ga20_lattice", args)

# %% (sa20_ga20)
args = {
    **BASE_ARGS,
    "sa_nmax": 20,
    "sa_steps": 600,
    "sa_batch": 64,
    "sa_objective": "packing",
    "sa_proposal": "mixed",
    "sa_cooling": "geom",
    "ga_nmax": 20,
    "ga_gens": 20,
    "ga_pop": 24,
}
run_submission_experiment("sa20_ga20", args)


# %% [markdown]
# ## Pipelines combinados (SA/block + refine + (A)LNS)
# %% (sa50_ref80_prefix)
args = {**BASE_ARGS, "sa_nmax": 50, "sa_steps": 800, "sa_batch": 64, "sa_objective": "prefix", "sa_proposal": "mixed", "sa_cooling": "log", "sa_swap_prob": 0.05, "sa_swap_prob_end": 0.0, "refine_nmin": 80, "refine_steps": 900, "refine_batch": 24, "refine_objective": "prefix", "refine_proposal": "mixed", "refine_cooling": "log"}
run_submission_experiment("sa50_ref80_prefix", args)

# %% (sa50_ref80_prefix_lns_alns_tabu5)
args = {
    **BASE_ARGS,
    "sa_nmax": 50,
    "sa_steps": 800,
    "sa_batch": 64,
    "sa_objective": "prefix",
    "sa_proposal": "mixed",
    "sa_cooling": "log",
    "sa_swap_prob": 0.05,
    "sa_swap_prob_end": 0.0,
    "refine_nmin": 80,
    "refine_steps": 900,
    "refine_batch": 24,
    "refine_objective": "prefix",
    "refine_proposal": "mixed",
    "refine_cooling": "log",
    "lns_nmax": 200,
    "lns_passes": 10,
    "lns_destroy_mode": "alns",
    "lns_tabu_tenure": 5,
    "lns_group_moves": 4,
}
run_submission_experiment("sa50_ref80_prefix_lns_alns_tabu5", args)

# %% (block_b2_ref200_prefix_lns_alns_tabu5)
args = {
    **BASE_ARGS,
    "block_nmax": 200,
    "block_init": "cluster",
    "block_size": 2,
    "block_steps": 350,
    "block_batch": 32,
    "block_objective": "prefix",
    "lns_nmax": 200,
    "lns_passes": 10,
    "lns_destroy_mode": "alns",
    "lns_tabu_tenure": 5,
    "lns_group_moves": 4,
    "refine_nmin": 200,
    "refine_steps": 1200,
    "refine_batch": 64,
    "refine_objective": "prefix",
    "refine_proposal": "mixed",
    "refine_cooling": "log",
}
run_submission_experiment("block_b2_ref200_prefix_lns_alns_tabu5", args)


# %% [markdown]
# ## Model-based (meta-init / heatmap / L2O / guided)
# %% (meta_init_sa_prefix)  # requer meta_init_path existir
args = {
    **BASE_ARGS,
    "meta_init_model": str(meta_init_path) if (meta_init_path is not None and meta_init_path.exists()) else None,
    "sa_nmax": 50,
    "sa_steps": 900,
    "sa_batch": 64,
    "sa_objective": "prefix",
    "sa_proposal": "mixed",
    "sa_cooling": "log",
    "sa_swap_prob": 0.05,
    "sa_swap_prob_end": 0.0,
}
run_submission_experiment("meta_init_sa_prefix", args)

# %% (heatmap20_only)  # requer heatmap_path existir
args = {
    **BASE_ARGS,
    "heatmap_model": str(heatmap_path) if (heatmap_path is not None and heatmap_path.exists()) else None,
    "heatmap_nmax": 20,
    "heatmap_steps": 250,
}
run_submission_experiment("heatmap20_only", args)

# %% (heatmap20_sa50_ref80_prefix)  # requer heatmap_path existir
args = {
    **BASE_ARGS,
    "heatmap_model": str(heatmap_path) if (heatmap_path is not None and heatmap_path.exists()) else None,
    "heatmap_nmax": 20,
    "heatmap_steps": 250,
    "sa_nmax": 50,
    "sa_steps": 800,
    "sa_batch": 64,
    "sa_objective": "prefix",
    "sa_proposal": "mixed",
    "sa_cooling": "log",
    "sa_swap_prob": 0.05,
    "sa_swap_prob_end": 0.0,
    "refine_nmin": 80,
    "refine_steps": 900,
    "refine_batch": 24,
    "refine_objective": "prefix",
    "refine_proposal": "mixed",
    "refine_cooling": "log",
}
run_submission_experiment("heatmap20_sa50_ref80_prefix", args)

# %% (l2o20_gnn)  # requer policy treinada existir (GNN_POLICY_PATH)
args = {
    **BASE_ARGS,
    "l2o_model": str(GNN_POLICY_PATH) if GNN_POLICY_PATH.exists() else None,
    "l2o_init": "lattice",
    "l2o_nmax": 20,
    "l2o_steps": 300,
    "l2o_deterministic": True,
}
run_submission_experiment("l2o20_gnn", args)

# %% (guided_refine_gnn_lns_alns_tabu5)  # requer policy existir
args = {
    **BASE_ARGS,
    "guided_model": str(GNN_POLICY_PATH) if GNN_POLICY_PATH.exists() else None,
    "guided_prob": 1.0,
    "guided_pmax": 0.05,
    "sa_nmax": 50,
    "sa_steps": 800,
    "sa_batch": 64,
    "sa_objective": "prefix",
    "sa_proposal": "mixed",
    "sa_cooling": "log",
    "sa_swap_prob": 0.05,
    "sa_swap_prob_end": 0.0,
    "refine_nmin": 80,
    "refine_steps": 900,
    "refine_batch": 24,
    "refine_objective": "prefix",
    "refine_proposal": "mixed",
    "refine_cooling": "log",
    "lns_nmax": 200,
    "lns_passes": 10,
    "lns_destroy_mode": "alns",
    "lns_tabu_tenure": 5,
    "lns_group_moves": 4,
}
run_submission_experiment("guided_refine_gnn_lns_alns_tabu5", args)

# %% (l2o20_mlp)  # requer policy treinada existir (MLP_POLICY_PATH)
args = {
    **BASE_ARGS,
    "l2o_model": str(MLP_POLICY_PATH) if MLP_POLICY_PATH.exists() else None,
    "l2o_init": "lattice",
    "l2o_nmax": 20,
    "l2o_steps": 300,
    "l2o_deterministic": True,
}
run_submission_experiment("l2o20_mlp", args)

# %% (guided_refine_mlp_lns_alns_tabu5)  # requer policy existir
args = {
    **BASE_ARGS,
    "guided_model": str(MLP_POLICY_PATH) if MLP_POLICY_PATH.exists() else None,
    "guided_prob": 1.0,
    "guided_pmax": 0.05,
    "sa_nmax": 50,
    "sa_steps": 800,
    "sa_batch": 64,
    "sa_objective": "prefix",
    "sa_proposal": "mixed",
    "sa_cooling": "log",
    "sa_swap_prob": 0.05,
    "sa_swap_prob_end": 0.0,
    "refine_nmin": 80,
    "refine_steps": 900,
    "refine_batch": 24,
    "refine_objective": "prefix",
    "refine_proposal": "mixed",
    "refine_cooling": "log",
    "lns_nmax": 200,
    "lns_passes": 10,
    "lns_destroy_mode": "alns",
    "lns_tabu_tenure": 5,
    "lns_group_moves": 4,
}
run_submission_experiment("guided_refine_mlp_lns_alns_tabu5", args)

# %% (l2o20_bc)  # requer bc_policy_path existir
args = {
    **BASE_ARGS,
    "l2o_model": str(bc_policy_path) if (bc_policy_path is not None and bc_policy_path.exists()) else None,
    "l2o_init": "lattice",
    "l2o_nmax": 20,
    "l2o_steps": 300,
    "l2o_deterministic": True,
}
run_submission_experiment("l2o20_bc", args)

# %% (guided_refine_bc_lns_alns_tabu5)  # requer bc_policy_path existir
args = {
    **BASE_ARGS,
    "guided_model": str(bc_policy_path) if (bc_policy_path is not None and bc_policy_path.exists()) else None,
    "guided_prob": 1.0,
    "guided_pmax": 0.05,
    "sa_nmax": 50,
    "sa_steps": 800,
    "sa_batch": 64,
    "sa_objective": "prefix",
    "sa_proposal": "mixed",
    "sa_cooling": "log",
    "sa_swap_prob": 0.05,
    "sa_swap_prob_end": 0.0,
    "refine_nmin": 80,
    "refine_steps": 900,
    "refine_batch": 24,
    "refine_objective": "prefix",
    "refine_proposal": "mixed",
    "refine_cooling": "log",
    "lns_nmax": 200,
    "lns_passes": 10,
    "lns_destroy_mode": "alns",
    "lns_tabu_tenure": 5,
    "lns_group_moves": 4,
}
run_submission_experiment("guided_refine_bc_lns_alns_tabu5", args)


# %% [markdown]
# ## Mother-prefix (resolve N=200 uma vez e emite prefixes)
# %% (mother_refine2000_prefix)
args = {
    **BASE_ARGS,
    "mother_prefix": True,
    "mother_reorder": "radial",
    "sa_nmax": 0,
    "refine_nmin": 200,
    "refine_steps": 2000,
    "refine_batch": 64,
    "refine_objective": "prefix",
    "refine_proposal": "mixed",
    "refine_cooling": "log",
    "refine_trans_sigma": 0.35,
    "refine_rot_sigma": 25.0,
    "refine_rot_prob": 0.4,
    "refine_rot_prob_end": 0.1,
}
run_submission_experiment("mother_refine2000_prefix", args)

# %% (mother_block_alns_tabu_refine2000_prefix)
args = {
    **BASE_ARGS,
    "mother_prefix": True,
    "mother_reorder": "radial",
    "block_nmax": 200,
    "block_init": "cluster",
    "block_size": 2,
    "block_steps": 500,
    "block_batch": 32,
    "block_objective": "prefix",
    "lns_nmax": 200,
    "lns_passes": 10,
    "lns_destroy_mode": "alns",
    "lns_tabu_tenure": 5,
    "lns_group_moves": 4,
    "refine_nmin": 200,
    "refine_steps": 2000,
    "refine_batch": 64,
    "refine_objective": "prefix",
    "refine_proposal": "mixed",
    "refine_cooling": "log",
    "refine_trans_sigma": 0.35,
    "refine_rot_sigma": 25.0,
    "refine_rot_prob": 0.4,
    "refine_rot_prob_end": 0.1,
}
run_submission_experiment("mother_block_alns_tabu_refine2000_prefix", args)


# %% [markdown]
# ## Sweep automático de receitas (opcional)
#
# Gera uma pool grande de receitas (combinações de flags do `python -m santa_packing.cli.generate_submission`),
# filtra as que dependem de modelos inexistentes e permite:
# - `RUN_SUBMISSION_SWEEP=True`: sweep 2-stage (rápido + final) + (opcional) ensemble por puzzle
# - `RUN_SUBMISSION_SWEEP=False`: gera 1 `submission.csv` com uma receita forte e calcula o score

# %% Receitas automáticas (pool + filtro)
def _stable_hash_dict(data: Dict[str, object]) -> str:
    blob = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


def _build_recipe_pool() -> tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]], Dict[str, object]]:
    """Gera uma pool de receitas cobrindo todas as features do generate_submission.py.

    Mantem as receitas como um dict de flags (somente chaves suportadas pelo script).
    Metadados (familia/modelo) ficam em um dict separado para auditoria.
    """

    # Base: desliga tudo e deixa lattice como fallback garantido.
    base: Dict[str, object] = {
        "sa_nmax": 0,
        "sa_batch": 64,
        "sa_steps": 400,
        "sa_trans_sigma": 0.2,
        "sa_rot_sigma": 15.0,
        "sa_rot_prob": 0.3,
        "sa_rot_prob_end": -1.0,
        "sa_swap_prob": 0.0,
        "sa_swap_prob_end": -1.0,
        "sa_cooling": "geom",
        "sa_cooling_power": 1.0,
        "sa_trans_nexp": 0.0,
        "sa_rot_nexp": 0.0,
        "sa_sigma_nref": 50.0,
        "sa_objective": "packing",
        "sa_proposal": "random",
        "sa_smart_prob": 1.0,
        "sa_smart_beta": 8.0,
        "sa_smart_drift": 1.0,
        "sa_smart_noise": 0.25,
        "sa_overlap_lambda": 0.0,
        "sa_allow_collisions": False,
        "meta_init_model": None,
        "heatmap_model": None,
        "heatmap_nmax": 0,
        "heatmap_steps": 200,
        "l2o_model": None,
        "l2o_init": "grid",
        "l2o_nmax": 0,
        "l2o_steps": 200,
        "l2o_trans_sigma": 0.2,
        "l2o_rot_sigma": 10.0,
        "l2o_deterministic": True,
        "lattice_pattern": "hex",
        "lattice_margin": 0.02,
        "lattice_rotate": 0.0,
        "lattice_rotations": "0,15,30",
        "mother_prefix": False,
        "mother_reorder": "radial",
        "refine_nmin": 0,
        "refine_batch": 16,
        "refine_steps": 0,
        "refine_trans_sigma": 0.2,
        "refine_rot_sigma": 15.0,
        "refine_rot_prob": 0.3,
        "refine_rot_prob_end": -1.0,
        "refine_swap_prob": 0.0,
        "refine_swap_prob_end": -1.0,
        "refine_cooling": "geom",
        "refine_cooling_power": 1.0,
        "refine_trans_nexp": 0.0,
        "refine_rot_nexp": 0.0,
        "refine_sigma_nref": 50.0,
        "refine_objective": "packing",
        "refine_proposal": "random",
        "refine_smart_prob": 1.0,
        "refine_smart_beta": 8.0,
        "refine_smart_drift": 1.0,
        "refine_smart_noise": 0.25,
        "refine_overlap_lambda": 0.0,
        "refine_allow_collisions": False,
        "lns_nmax": 0,
        "lns_passes": 0,
        "lns_destroy_k": 8,
        "lns_destroy_mode": "mixed",
        "lns_tabu_tenure": 0,
        "lns_candidates": 64,
        "lns_angle_samples": 8,
        "lns_pad_scale": 2.0,
        "lns_group_moves": 0,
        "lns_group_size": 3,
        "lns_group_trans_sigma": 0.05,
        "lns_group_rot_sigma": 20.0,
        "lns_t_start": 0.0,
        "lns_t_end": 0.0,
        "guided_model": None,
        "guided_prob": 1.0,
        "guided_pmax": 0.05,
        "guided_prob_end": -1.0,
        "guided_pmax_end": -1.0,
        "block_nmax": 0,
        "block_size": 2,
        "block_batch": 32,
        "block_steps": 0,
        "block_trans_sigma": 0.2,
        "block_rot_sigma": 15.0,
        "block_rot_prob": 0.25,
        "block_rot_prob_end": -1.0,
        "block_cooling": "geom",
        "block_cooling_power": 1.0,
        "block_trans_nexp": 0.0,
        "block_rot_nexp": 0.0,
        "block_sigma_nref": 50.0,
        "block_overlap_lambda": 0.0,
        "block_allow_collisions": False,
        "block_objective": "packing",
        "block_init": "cluster",
        "block_template_pattern": "hex",
        "block_template_margin": 0.02,
        "block_template_rotate": 0.0,
        "hc_nmax": 0,
        "hc_passes": 2,
        "hc_step_xy": 0.01,
        "hc_step_deg": 2.0,
        "ga_nmax": 0,
        "ga_pop": 24,
        "ga_gens": 20,
        "ga_elite_frac": 0.25,
        "ga_crossover_prob": 0.5,
        "ga_mut_sigma_xy": 0.01,
        "ga_mut_sigma_deg": 2.0,
        "ga_directed_prob": 0.5,
        "ga_directed_step_xy": 0.02,
        "ga_directed_k": 8,
        "ga_repair_iters": 200,
        "ga_hc_passes": 0,
        "ga_hc_step_xy": 0.01,
        "ga_hc_step_deg": 2.0,
    }

    # ===== Experiment grids (ajuste aqui) =====
    # Lattice sweep (base/fallback de tudo).
    lattice_variants_all: List[Dict[str, object]] = []
    lattice_margins = [0.0, 0.005, 0.01, 0.015, 0.02]
    lattice_rot_sets = [
        "0,15,30",
        "0,10,20,30",
        "0,5,10,15,20,25,30",
        "0,9,18,27,36",
    ]
    lattice_const_rots = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]

    # Multi-rotation: picks best lattice per n.
    for pattern, margin, rots in itertools.product(["hex", "square"], lattice_margins, lattice_rot_sets):
        lattice_variants_all.append(
            {
                "lattice_pattern": pattern,
                "lattice_margin": float(margin),
                "lattice_rotations": str(rots),
            }
        )

    # Constant rotation: disable rotate-set and use `lattice_rotate`.
    for pattern, margin, rot in itertools.product(["hex", "square"], lattice_margins, lattice_const_rots):
        lattice_variants_all.append(
            {
                "lattice_pattern": pattern,
                "lattice_margin": float(margin),
                "lattice_rotations": "none",
                "lattice_rotate": float(rot),
            }
        )

    # SA (n pequeno): custo principal do solver -> vale experimentar.
    sa_level_presets: Dict[str, Dict[str, object]] = {
        "sa30": {"sa_nmax": 30, "sa_batch": 64, "sa_steps": 400, "sa_trans_sigma": 0.2, "sa_rot_sigma": 15.0, "sa_rot_prob": 0.3},
        "sa50": {"sa_nmax": 50, "sa_batch": 64, "sa_steps": 500, "sa_trans_sigma": 0.2, "sa_rot_sigma": 15.0, "sa_rot_prob": 0.3},
        "sa80": {"sa_nmax": 80, "sa_batch": 96, "sa_steps": 600, "sa_trans_sigma": 0.2, "sa_rot_sigma": 18.0, "sa_rot_prob": 0.35},
    }
    sa_objectives = ["packing", "prefix"]
    sa_proposals = ["random", "mixed", "bbox_inward", "bbox", "inward", "smart"]
    sa_coolings = ["geom", "linear", "log"]
    sa_overlap_lambdas = [0.0, 0.01]
    sa_presets: Dict[str, Dict[str, object]] = {}
    for lvl_name, lvl_cfg in sa_level_presets.items():
        for objective, proposal, cooling, overlap_lambda in itertools.product(
            sa_objectives,
            sa_proposals,
            sa_coolings,
            sa_overlap_lambdas,
        ):
            cfg: Dict[str, object] = dict(lvl_cfg)
            cfg.update(
                {
                    "sa_objective": str(objective),
                    "sa_proposal": str(proposal),
                    "sa_cooling": str(cooling),
                    "sa_overlap_lambda": float(overlap_lambda),
                    # Swap é útil principalmente quando objetivo=prefix (reordena prefixos sem mexer na geometria).
                    "sa_swap_prob": 0.05 if objective == "prefix" else 0.0,
                    "sa_swap_prob_end": 0.0 if objective == "prefix" else -1.0,
                }
            )
            if proposal != "random":
                cfg.update(
                    {
                        "sa_smart_prob": 0.7,
                        "sa_smart_beta": 8.0,
                        "sa_smart_drift": 1.0,
                        "sa_smart_noise": 0.25,
                    }
                )

            key = f"{lvl_name}_{objective}_{proposal}_{cooling}_ol{overlap_lambda:g}"
            sa_presets[key] = cfg

    # Refine (n alto): warm-start lattice/l2o e refina via SA.
    refine_level_presets: Dict[str, Dict[str, object]] = {
        "ref80_200": {"refine_nmin": 80, "refine_batch": 16, "refine_steps": 200, "refine_trans_sigma": 0.2, "refine_rot_sigma": 15.0, "refine_rot_prob": 0.3},
        "ref80_400": {"refine_nmin": 80, "refine_batch": 24, "refine_steps": 400, "refine_trans_sigma": 0.2, "refine_rot_sigma": 15.0, "refine_rot_prob": 0.3},
        "ref120_300": {"refine_nmin": 120, "refine_batch": 24, "refine_steps": 300, "refine_trans_sigma": 0.2, "refine_rot_sigma": 15.0, "refine_rot_prob": 0.3},
        # heavy: usado principalmente com mother_prefix (n=200)
        "ref200_2000": {"refine_nmin": 200, "refine_batch": 64, "refine_steps": 2000, "refine_trans_sigma": 0.35, "refine_rot_sigma": 25.0, "refine_rot_prob": 0.4, "refine_rot_prob_end": 0.1},
    }
    refine_objectives = ["packing", "prefix"]
    refine_proposals = ["random", "mixed", "bbox_inward", "bbox", "inward", "smart"]
    refine_coolings = ["geom", "linear", "log"]
    refine_overlap_lambdas = [0.0, 0.01]
    refine_presets: Dict[str, Dict[str, object]] = {}
    for lvl_name, lvl_cfg in refine_level_presets.items():
        for objective, proposal, cooling, overlap_lambda in itertools.product(
            refine_objectives,
            refine_proposals,
            refine_coolings,
            refine_overlap_lambdas,
        ):
            cfg: Dict[str, object] = dict(lvl_cfg)
            cfg.update(
                {
                    "refine_objective": str(objective),
                    "refine_proposal": str(proposal),
                    "refine_cooling": str(cooling),
                    "refine_overlap_lambda": float(overlap_lambda),
                    # schedules úteis para "esfriar" no fim
                    "refine_rot_prob_end": float(lvl_cfg.get("refine_rot_prob_end", -1.0)),
                }
            )
            if proposal != "random":
                cfg.update(
                    {
                        "refine_smart_prob": 0.7,
                        "refine_smart_beta": 8.0,
                        "refine_smart_drift": 1.0,
                        "refine_smart_noise": 0.25,
                    }
                )
            key = f"{lvl_name}_{objective}_{proposal}_{cooling}_ol{overlap_lambda:g}"
            refine_presets[key] = cfg

    # Meta-model blocks (2..4): cluster trees into rigid groups and optimize blocks before per-tree refinement.
    block_sizes = [2, 3, 4]
    block_objectives = ["packing", "prefix"]
    block_steps_list = [200, 350, 500]
    block_presets: Dict[str, Dict[str, object]] = {}

    # Cluster-init blocks (cheap).
    for size, objective, steps in itertools.product(block_sizes, block_objectives, block_steps_list):
        cfg: Dict[str, object] = {
            "block_nmax": 200,
            "block_size": int(size),
            "block_batch": 32,
            "block_steps": int(steps),
            "block_trans_sigma": 0.2,
            "block_rot_sigma": 20.0,
            "block_rot_prob": 0.25,
            "block_objective": str(objective),
            "block_init": "cluster",
        }
        key = f"blk200_cluster_b{size}_{objective}_s{steps}"
        block_presets[key] = cfg

    # Template-init blocks (more structured).
    template_patterns = ["hex", "square"]
    template_rotates = [0.0, 15.0, 30.0]
    template_margins = [0.0, 0.02]
    for size, objective, steps, tpat, trot, tmar in itertools.product(
        block_sizes,
        block_objectives,
        block_steps_list,
        template_patterns,
        template_rotates,
        template_margins,
    ):
        cfg = {
            "block_nmax": 200,
            "block_size": int(size),
            "block_batch": 32,
            "block_steps": int(steps),
            "block_trans_sigma": 0.2,
            "block_rot_sigma": 20.0,
            "block_rot_prob": 0.25,
            "block_objective": str(objective),
            "block_init": "template",
            "block_template_pattern": str(tpat),
            "block_template_margin": float(tmar),
            "block_template_rotate": float(trot),
        }
        key = f"blk200_template_{tpat}_r{trot:g}_m{tmar:g}_b{size}_{objective}_s{steps}"
        block_presets[key] = cfg

    # Large Neighborhood Search / ALNS-style post-opt (n pequeno/medio).
    lns_presets: Dict[str, Dict[str, object]] = {
        "lns10": {
            "lns_nmax": 200,
            "lns_passes": 10,
            "lns_destroy_k": 8,
            "lns_destroy_mode": "mixed",
            "lns_tabu_tenure": 0,
            "lns_candidates": 64,
            "lns_angle_samples": 8,
            "lns_pad_scale": 2.0,
            "lns_group_moves": 4,
            "lns_group_size": 3,
            "lns_group_trans_sigma": 0.05,
            "lns_group_rot_sigma": 20.0,
            "lns_t_start": 0.0,
            "lns_t_end": 0.0,
        },
        "lns10_mixed_t5": {
            "lns_nmax": 200,
            "lns_passes": 10,
            "lns_destroy_k": 8,
            "lns_destroy_mode": "mixed",
            "lns_tabu_tenure": 5,
            "lns_candidates": 64,
            "lns_angle_samples": 8,
            "lns_pad_scale": 2.0,
            "lns_group_moves": 4,
            "lns_group_size": 3,
            "lns_group_trans_sigma": 0.05,
            "lns_group_rot_sigma": 20.0,
            "lns_t_start": 0.0,
            "lns_t_end": 0.0,
        },
        "lns10_sa": {
            "lns_nmax": 200,
            "lns_passes": 10,
            "lns_destroy_k": 8,
            "lns_destroy_mode": "mixed",
            "lns_tabu_tenure": 0,
            "lns_candidates": 64,
            "lns_angle_samples": 8,
            "lns_pad_scale": 2.0,
            "lns_group_moves": 4,
            "lns_group_size": 3,
            "lns_group_trans_sigma": 0.05,
            "lns_group_rot_sigma": 20.0,
            "lns_t_start": 0.2,
            "lns_t_end": 0.02,
        },
        "lns10_boundary_t5": {
            "lns_nmax": 200,
            "lns_passes": 10,
            "lns_destroy_k": 10,
            "lns_destroy_mode": "boundary",
            "lns_tabu_tenure": 5,
            "lns_candidates": 96,
            "lns_angle_samples": 10,
            "lns_pad_scale": 2.0,
            "lns_group_moves": 4,
            "lns_group_size": 3,
            "lns_group_trans_sigma": 0.05,
            "lns_group_rot_sigma": 20.0,
            "lns_t_start": 0.0,
            "lns_t_end": 0.0,
        },
        "lns10_cluster_t5": {
            "lns_nmax": 200,
            "lns_passes": 10,
            "lns_destroy_k": 10,
            "lns_destroy_mode": "cluster",
            "lns_tabu_tenure": 5,
            "lns_candidates": 96,
            "lns_angle_samples": 10,
            "lns_pad_scale": 2.0,
            "lns_group_moves": 4,
            "lns_group_size": 3,
            "lns_group_trans_sigma": 0.05,
            "lns_group_rot_sigma": 20.0,
            "lns_t_start": 0.0,
            "lns_t_end": 0.0,
        },
        "lns10_random": {
            "lns_nmax": 200,
            "lns_passes": 10,
            "lns_destroy_k": 8,
            "lns_destroy_mode": "random",
            "lns_tabu_tenure": 0,
            "lns_candidates": 64,
            "lns_angle_samples": 8,
            "lns_pad_scale": 2.0,
            "lns_group_moves": 4,
            "lns_group_size": 3,
            "lns_group_trans_sigma": 0.05,
            "lns_group_rot_sigma": 20.0,
            "lns_t_start": 0.0,
            "lns_t_end": 0.0,
        },
        # New: adaptive LNS (ALNS) + tabu
        "lns10_alns": {
            "lns_nmax": 200,
            "lns_passes": 10,
            "lns_destroy_k": 8,
            "lns_destroy_mode": "alns",
            "lns_tabu_tenure": 0,
            "lns_candidates": 64,
            "lns_angle_samples": 8,
            "lns_pad_scale": 2.0,
            "lns_group_moves": 4,
            "lns_group_size": 3,
            "lns_group_trans_sigma": 0.05,
            "lns_group_rot_sigma": 20.0,
            "lns_t_start": 0.0,
            "lns_t_end": 0.0,
        },
        "lns10_alns_t5": {
            "lns_nmax": 200,
            "lns_passes": 10,
            "lns_destroy_k": 8,
            "lns_destroy_mode": "alns",
            "lns_tabu_tenure": 5,
            "lns_candidates": 64,
            "lns_angle_samples": 8,
            "lns_pad_scale": 2.0,
            "lns_group_moves": 4,
            "lns_group_size": 3,
            "lns_group_trans_sigma": 0.05,
            "lns_group_rot_sigma": 20.0,
            "lns_t_start": 0.0,
            "lns_t_end": 0.0,
        },
        "lns20_alns_sa_t10": {
            "lns_nmax": 200,
            "lns_passes": 20,
            "lns_destroy_k": 10,
            "lns_destroy_mode": "alns",
            "lns_tabu_tenure": 10,
            "lns_candidates": 96,
            "lns_angle_samples": 12,
            "lns_pad_scale": 2.0,
            "lns_group_moves": 6,
            "lns_group_size": 4,
            "lns_group_trans_sigma": 0.05,
            "lns_group_rot_sigma": 20.0,
            "lns_t_start": 0.2,
            "lns_t_end": 0.02,
        },
    }

    # Guided SA knobs (policy como proposal quando confiante).
    guided_presets: Dict[str, Dict[str, object]] = {
        "guided_p005": {"guided_prob": 1.0, "guided_pmax": 0.05},
        "guided_p02": {"guided_prob": 1.0, "guided_pmax": 0.02},
        "guided_mix": {"guided_prob": 0.5, "guided_pmax": 0.05},
        "guided_sched": {"guided_prob": 1.0, "guided_pmax": 0.05, "guided_prob_end": 0.5, "guided_pmax_end": 0.08},
        "guided_sched2": {"guided_prob": 0.8, "guided_pmax": 0.03, "guided_prob_end": 0.4, "guided_pmax_end": 0.06},
    }

    # L2O knobs (n pequeno): policy pode substituir SA inicial.
    l2o_presets: Dict[str, Dict[str, object]] = {
        "l2o10_det": {"l2o_init": "lattice", "l2o_nmax": 10, "l2o_steps": 200, "l2o_trans_sigma": 0.2, "l2o_rot_sigma": 10.0, "l2o_deterministic": True},
        "l2o10_stoch": {"l2o_init": "lattice", "l2o_nmax": 10, "l2o_steps": 250, "l2o_trans_sigma": 0.2, "l2o_rot_sigma": 10.0, "l2o_deterministic": False},
        "l2o20_det": {"l2o_init": "lattice", "l2o_nmax": 20, "l2o_steps": 250, "l2o_trans_sigma": 0.2, "l2o_rot_sigma": 10.0, "l2o_deterministic": True},
        "l2o30_det": {"l2o_init": "lattice", "l2o_nmax": 30, "l2o_steps": 300, "l2o_trans_sigma": 0.2, "l2o_rot_sigma": 10.0, "l2o_deterministic": True},
        "l2o20_grid": {"l2o_init": "grid", "l2o_nmax": 20, "l2o_steps": 250, "l2o_trans_sigma": 0.2, "l2o_rot_sigma": 10.0, "l2o_deterministic": True},
    }

    # Heatmap knobs (n muito pequeno): meta-optimizer alternativo ao SA/L2O.
    heatmap_presets: Dict[str, Dict[str, object]] = {
        "heat10": {"heatmap_nmax": 10, "heatmap_steps": 200},
        "heat20": {"heatmap_nmax": 20, "heatmap_steps": 250},
        "heat30": {"heatmap_nmax": 30, "heatmap_steps": 350},
    }

    # Post-opt knobs (n pequeno): hill-climb e GA (opcional).
    hc_presets: Dict[str, Dict[str, object]] = {
        "hc20": {"hc_nmax": 20, "hc_passes": 2, "hc_step_xy": 0.01, "hc_step_deg": 2.0},
        "hc50": {"hc_nmax": 50, "hc_passes": 2, "hc_step_xy": 0.01, "hc_step_deg": 2.0},
    }
    ga_presets: Dict[str, Dict[str, object]] = {
        "ga20": {"ga_nmax": 20, "ga_pop": 24, "ga_gens": 20, "ga_elite_frac": 0.25, "ga_crossover_prob": 0.5, "ga_mut_sigma_xy": 0.01, "ga_mut_sigma_deg": 2.0, "ga_repair_iters": 200},
        "ga50": {"ga_nmax": 50, "ga_pop": 32, "ga_gens": 25, "ga_elite_frac": 0.25, "ga_crossover_prob": 0.5, "ga_mut_sigma_xy": 0.01, "ga_mut_sigma_deg": 2.0, "ga_repair_iters": 250},
    }

    # Mother-prefix presets: resolve N=nmax once e emite prefixes.
    mother_presets: Dict[str, Dict[str, object]] = {
        "mother_ref2000_prefix": {
            "mother_prefix": True,
            "mother_reorder": "radial",
            "sa_nmax": 0,
            "refine_nmin": 200,
            "refine_steps": 2000,
            "refine_batch": 64,
            "refine_objective": "prefix",
            "refine_proposal": "mixed",
            "refine_cooling": "log",
            "refine_trans_sigma": 0.35,
            "refine_rot_sigma": 25.0,
            "refine_rot_prob": 0.4,
            "refine_rot_prob_end": 0.1,
        },
        "mother_sa5000_prefix": {
            "mother_prefix": True,
            "mother_reorder": "radial",
            "sa_nmax": 200,
            "sa_batch": 128,
            "sa_steps": 5000,
            "sa_trans_sigma": 0.25,
            "sa_rot_sigma": 25.0,
            "sa_rot_prob": 0.4,
            "sa_rot_prob_end": 0.1,
            "sa_objective": "prefix",
            "sa_proposal": "mixed",
            "sa_cooling": "log",
            "sa_swap_prob": 0.05,
            "sa_swap_prob_end": 0.0,
            "sa_overlap_lambda": 0.01,
            "refine_nmin": 200,
            "refine_steps": 1000,
            "refine_batch": 64,
            "refine_objective": "prefix",
            "refine_proposal": "mixed",
            "refine_cooling": "log",
            "refine_trans_sigma": 0.35,
            "refine_rot_sigma": 25.0,
            "refine_rot_prob": 0.4,
            "refine_rot_prob_end": 0.1,
        },
        "mother_blk_ref_prefix": {
            "mother_prefix": True,
            "mother_reorder": "radial",
            "sa_nmax": 0,
            "block_nmax": 200,
            "block_size": 2,
            "block_steps": 350,
            "block_batch": 32,
            "block_objective": "prefix",
            "block_init": "cluster",
            "lns_nmax": 200,
            "lns_passes": 10,
            "lns_destroy_k": 8,
            "lns_destroy_mode": "alns",
            "lns_tabu_tenure": 5,
            "lns_candidates": 64,
            "lns_angle_samples": 8,
            "lns_pad_scale": 2.0,
            "lns_group_moves": 4,
            "lns_group_size": 3,
            "lns_group_trans_sigma": 0.05,
            "lns_group_rot_sigma": 20.0,
            "refine_nmin": 200,
            "refine_steps": 2000,
            "refine_batch": 64,
            "refine_objective": "prefix",
            "refine_proposal": "mixed",
            "refine_cooling": "log",
            "refine_trans_sigma": 0.35,
            "refine_rot_sigma": 25.0,
            "refine_rot_prob": 0.4,
            "refine_rot_prob_end": 0.1,
        },
    }

    # Limites para nao explodir combinacoes (aumente/disable para explorar mais).
    # `None` = sem limite (maximo de receitas).
    MAX_RECIPES_PER_FAMILY = RECIPES_MAX_RECIPES_PER_FAMILY
    MAX_LATTICE_VARIANTS = RECIPES_MAX_LATTICE_VARIANTS

    def limit_variants(variants: List[Dict[str, object]], max_n: int | None) -> List[Dict[str, object]]:
        if max_n is None or len(variants) <= max_n:
            return variants
        scored = [(json.dumps(v, sort_keys=True), v) for v in variants]
        scored.sort(key=lambda x: hashlib.sha1(x[0].encode("utf-8")).hexdigest())
        return [v for _s, v in scored[:max_n]]

    lattice_variants = limit_variants(lattice_variants_all, MAX_LATTICE_VARIANTS)

    recipes: Dict[str, Dict[str, object]] = {}
    meta: Dict[str, Dict[str, object]] = {}

    def add_recipe(family: str, recipe: Dict[str, object], *, meta_extra: Dict[str, object] | None = None) -> None:
        full = dict(base)
        full.update(recipe)
        rid = _stable_hash_dict(full)
        name = f"{family}_{rid}"
        if name in recipes:
            return
        recipes[name] = full
        meta[name] = {"family": family}
        if meta_extra:
            meta[name].update(meta_extra)

    # ---- Baselines ----
    for lat in lattice_variants:
        add_recipe("lattice", lat, meta_extra={"lattice": lat})

    # ---- SA / SA+refine ----
    for lat in lattice_variants:
        for sa_name, sa_cfg in sa_presets.items():
            add_recipe("sa", {**lat, **sa_cfg}, meta_extra={"lattice": lat, "sa": sa_name})
            for ref_name, ref_cfg in refine_presets.items():
                add_recipe("sa_refine", {**lat, **sa_cfg, **ref_cfg}, meta_extra={"lattice": lat, "sa": sa_name, "refine": ref_name})
                if META_INIT_MODEL is not None:
                    add_recipe(
                        "sa_refine_meta",
                        {**lat, **sa_cfg, **ref_cfg, "meta_init_model": META_INIT_MODEL},
                        meta_extra={"lattice": lat, "sa": sa_name, "refine": ref_name, "meta_init": True},
                    )

    # ---- Block meta-model variants ----
    for lat in lattice_variants:
        for blk_name, blk_cfg in block_presets.items():
            for ref_name, ref_cfg in refine_presets.items():
                add_recipe("block_refine", {**lat, **blk_cfg, **ref_cfg}, meta_extra={"lattice": lat, "block": blk_name, "refine": ref_name})
                if META_INIT_MODEL is not None:
                    add_recipe(
                        "block_refine_meta",
                        {**lat, **blk_cfg, **ref_cfg, "meta_init_model": META_INIT_MODEL},
                        meta_extra={"lattice": lat, "block": blk_name, "refine": ref_name, "meta_init": True},
                    )
            for sa_name, sa_cfg in sa_presets.items():
                for ref_name, ref_cfg in refine_presets.items():
                    add_recipe(
                        "block_sa_refine",
                        {**lat, **blk_cfg, **sa_cfg, **ref_cfg},
                        meta_extra={"lattice": lat, "block": blk_name, "sa": sa_name, "refine": ref_name},
                    )
                    if META_INIT_MODEL is not None:
                        add_recipe(
                            "block_sa_refine_meta",
                            {**lat, **blk_cfg, **sa_cfg, **ref_cfg, "meta_init_model": META_INIT_MODEL},
                            meta_extra={"lattice": lat, "block": blk_name, "sa": sa_name, "refine": ref_name, "meta_init": True},
                        )

    # ---- LNS / ALNS post-opt variants ----
    for lat in lattice_variants:
        for lns_name, lns_cfg in lns_presets.items():
            add_recipe("lns", {**lat, **lns_cfg}, meta_extra={"lattice": lat, "lns": lns_name})
            for ref_name, ref_cfg in refine_presets.items():
                add_recipe("refine_lns", {**lat, **ref_cfg, **lns_cfg}, meta_extra={"lattice": lat, "refine": ref_name, "lns": lns_name})
            for sa_name, sa_cfg in sa_presets.items():
                for ref_name, ref_cfg in refine_presets.items():
                    add_recipe(
                        "sa_refine_lns",
                        {**lat, **sa_cfg, **ref_cfg, **lns_cfg},
                        meta_extra={"lattice": lat, "sa": sa_name, "refine": ref_name, "lns": lns_name},
                    )
            for blk_name, blk_cfg in block_presets.items():
                for ref_name, ref_cfg in refine_presets.items():
                    add_recipe(
                        "block_refine_lns",
                        {**lat, **blk_cfg, **ref_cfg, **lns_cfg},
                        meta_extra={"lattice": lat, "block": blk_name, "refine": ref_name, "lns": lns_name},
                    )

    # ---- Guided SA (em cima do melhor preset base) ----
    guided_lattice = lattice_variants[:2] if lattice_variants else [{"lattice_pattern": "hex", "lattice_margin": 0.02, "lattice_rotate": 0.0, "lattice_rotations": "0,15,30"}]
    guided_sa = sa_presets.get("sa50_prefix_mixed_log_ol0", next(iter(sa_presets.values())))
    guided_ref = refine_presets.get("ref80_200_prefix_mixed_log_ol0", next(iter(refine_presets.values())))
    for model_name, model_path in CANDIDATE_GUIDED_MODELS.items():
        for lat in guided_lattice:
            for g_name, g_cfg in guided_presets.items():
                add_recipe(
                    "guided_refine",
                    {**lat, **guided_sa, **guided_ref, **g_cfg, "guided_model": str(model_path)},
                    meta_extra={"guided_model": model_name, "guided": g_name, "lattice": lat},
                )
                if META_INIT_MODEL is not None:
                    add_recipe(
                        "guided_refine_meta",
                        {**lat, **guided_sa, **guided_ref, **g_cfg, "guided_model": str(model_path), "meta_init_model": META_INIT_MODEL},
                        meta_extra={"guided_model": model_name, "guided": g_name, "lattice": lat, "meta_init": True},
                    )
                for lns_name, lns_cfg in lns_presets.items():
                    add_recipe(
                        "guided_refine_lns",
                        {**lat, **guided_sa, **guided_ref, **g_cfg, "guided_model": str(model_path), **lns_cfg},
                        meta_extra={"guided_model": model_name, "guided": g_name, "lns": lns_name, "lattice": lat},
                    )
                    if META_INIT_MODEL is not None:
                        add_recipe(
                            "guided_refine_lns_meta",
                            {**lat, **guided_sa, **guided_ref, **g_cfg, "guided_model": str(model_path), **lns_cfg, "meta_init_model": META_INIT_MODEL},
                            meta_extra={"guided_model": model_name, "guided": g_name, "lns": lns_name, "lattice": lat, "meta_init": True},
                        )

                for blk_name, blk_cfg in block_presets.items():
                    add_recipe(
                        "guided_block_refine",
                        {**lat, **blk_cfg, **guided_sa, **guided_ref, **g_cfg, "guided_model": str(model_path)},
                        meta_extra={"guided_model": model_name, "guided": g_name, "block": blk_name, "lattice": lat},
                    )
                    if META_INIT_MODEL is not None:
                        add_recipe(
                            "guided_block_refine_meta",
                            {**lat, **blk_cfg, **guided_sa, **guided_ref, **g_cfg, "guided_model": str(model_path), "meta_init_model": META_INIT_MODEL},
                            meta_extra={"guided_model": model_name, "guided": g_name, "block": blk_name, "lattice": lat, "meta_init": True},
                        )
                    for lns_name, lns_cfg in lns_presets.items():
                        add_recipe(
                            "guided_block_refine_lns",
                            {**lat, **blk_cfg, **guided_sa, **guided_ref, **g_cfg, "guided_model": str(model_path), **lns_cfg},
                            meta_extra={"guided_model": model_name, "guided": g_name, "block": blk_name, "lns": lns_name, "lattice": lat},
                        )
                        if META_INIT_MODEL is not None:
                            add_recipe(
                                "guided_block_refine_lns_meta",
                                {**lat, **blk_cfg, **guided_sa, **guided_ref, **g_cfg, "guided_model": str(model_path), **lns_cfg, "meta_init_model": META_INIT_MODEL},
                                meta_extra={"guided_model": model_name, "guided": g_name, "block": blk_name, "lns": lns_name, "lattice": lat, "meta_init": True},
                            )

    # ---- L2O (n pequeno) + SA/refine ----
    l2o_lattice = lattice_variants[:2] if lattice_variants else [{"lattice_pattern": "hex", "lattice_margin": 0.02, "lattice_rotate": 0.0}]
    l2o_sa = sa_presets.get("sa50_prefix_mixed_log_ol0", next(iter(sa_presets.values())))
    l2o_ref = refine_presets.get("ref80_200_prefix_mixed_log_ol0", next(iter(refine_presets.values())))
    for model_name, model_path in CANDIDATE_L2O_MODELS.items():
        for lat in l2o_lattice:
            for l2o_name, l2o_cfg in l2o_presets.items():
                add_recipe(
                    "l2o_refine",
                    {**lat, **l2o_sa, **l2o_ref, **l2o_cfg, "l2o_model": str(model_path)},
                    meta_extra={"l2o_model": model_name, "l2o": l2o_name, "lattice": lat},
                )
                if META_INIT_MODEL is not None:
                    add_recipe(
                        "l2o_refine_meta",
                        {**lat, **l2o_sa, **l2o_ref, **l2o_cfg, "l2o_model": str(model_path), "meta_init_model": META_INIT_MODEL},
                        meta_extra={"l2o_model": model_name, "l2o": l2o_name, "lattice": lat, "meta_init": True},
                    )
                for lns_name, lns_cfg in lns_presets.items():
                    add_recipe(
                        "l2o_refine_lns",
                        {**lat, **l2o_sa, **l2o_ref, **l2o_cfg, "l2o_model": str(model_path), **lns_cfg},
                        meta_extra={"l2o_model": model_name, "l2o": l2o_name, "lns": lns_name, "lattice": lat},
                    )
                    if META_INIT_MODEL is not None:
                        add_recipe(
                            "l2o_refine_lns_meta",
                            {**lat, **l2o_sa, **l2o_ref, **l2o_cfg, "l2o_model": str(model_path), **lns_cfg, "meta_init_model": META_INIT_MODEL},
                            meta_extra={"l2o_model": model_name, "l2o": l2o_name, "lns": lns_name, "lattice": lat, "meta_init": True},
                        )

    # ---- Heatmap variants ----
    if HEATMAP_MODEL is not None:
        heat_lattice = lattice_variants[:2] if lattice_variants else [{"lattice_pattern": "hex", "lattice_margin": 0.02, "lattice_rotate": 0.0}]
        for lat in heat_lattice:
            for h_name, h_cfg in heatmap_presets.items():
                add_recipe(
                    "heatmap",
                    {**lat, **h_cfg, "heatmap_model": HEATMAP_MODEL},
                    meta_extra={"heatmap": h_name, "lattice": lat},
                )
                # heatmap + SA/refine
                add_recipe(
                    "heatmap_sa_refine",
                    {**lat, **h_cfg, "heatmap_model": HEATMAP_MODEL, **guided_sa, **guided_ref},
                    meta_extra={"heatmap": h_name, "lattice": lat, "sa": "sa50_prefix_mixed_log_ol0", "refine": "ref80_200_prefix_mixed_log_ol0"},
                )
                # heatmap + l2o (com o melhor modelo disponivel) + refine
                best_l2o_name = next(iter(CANDIDATE_L2O_MODELS))
                best_l2o_path = CANDIDATE_L2O_MODELS[best_l2o_name]
                best_l2o_cfg = l2o_presets.get("l2o10_det", next(iter(l2o_presets.values())))
                add_recipe(
                    "heatmap_l2o_refine",
                    {**lat, **h_cfg, "heatmap_model": HEATMAP_MODEL, **best_l2o_cfg, "l2o_model": str(best_l2o_path), **guided_sa, **guided_ref},
                    meta_extra={"heatmap": h_name, "l2o_model": best_l2o_name, "l2o": "l2o10_det", "lattice": lat},
                )

    # ---- Post-opt (hill-climb / GA) variants ----
    post_lattice = lattice_variants[:2] if lattice_variants else [{"lattice_pattern": "hex", "lattice_margin": 0.02, "lattice_rotate": 0.0, "lattice_rotations": "0,15,30"}]
    for lat in post_lattice:
        for hc_name, hc_cfg in hc_presets.items():
            add_recipe("hc", {**lat, **hc_cfg}, meta_extra={"lattice": lat, "hc": hc_name})
            add_recipe(
                "sa_refine_hc",
                {**lat, **guided_sa, **guided_ref, **hc_cfg},
                meta_extra={"lattice": lat, "sa": "sa50_prefix_mixed_log_ol0", "refine": "ref80_200_prefix_mixed_log_ol0", "hc": hc_name},
            )
            for lns_name, lns_cfg in lns_presets.items():
                add_recipe(
                    "sa_refine_lns_hc",
                    {**lat, **guided_sa, **guided_ref, **lns_cfg, **hc_cfg},
                    meta_extra={"lattice": lat, "sa": "sa50_prefix_mixed_log_ol0", "refine": "ref80_200_prefix_mixed_log_ol0", "lns": lns_name, "hc": hc_name},
                )

        for ga_name, ga_cfg in ga_presets.items():
            add_recipe("ga", {**lat, **ga_cfg}, meta_extra={"lattice": lat, "ga": ga_name})
            add_recipe(
                "sa_refine_ga",
                {**lat, **guided_sa, **guided_ref, **ga_cfg},
                meta_extra={"lattice": lat, "sa": "sa50_prefix_mixed_log_ol0", "refine": "ref80_200_prefix_mixed_log_ol0", "ga": ga_name},
            )
            for lns_name, lns_cfg in lns_presets.items():
                add_recipe(
                    "sa_refine_lns_ga",
                    {**lat, **guided_sa, **guided_ref, **lns_cfg, **ga_cfg},
                    meta_extra={"lattice": lat, "sa": "sa50_prefix_mixed_log_ol0", "refine": "ref80_200_prefix_mixed_log_ol0", "lns": lns_name, "ga": ga_name},
                )

    # ---- Mother-prefix variants (solve N once, emit prefixes) ----
    mother_lattice = lattice_variants[:2] if lattice_variants else [{"lattice_pattern": "hex", "lattice_margin": 0.02, "lattice_rotate": 0.0, "lattice_rotations": "0,15,30"}]
    for lat in mother_lattice:
        for m_name, m_cfg in mother_presets.items():
            add_recipe("mother", {**lat, **m_cfg}, meta_extra={"lattice": lat, "mother": m_name})
            # mother + guided refine (usa policy no refine/SA quando habilitado)
            for model_name, model_path in CANDIDATE_GUIDED_MODELS.items():
                for g_name, g_cfg in guided_presets.items():
                    add_recipe(
                        "mother_guided",
                        {**lat, **m_cfg, **g_cfg, "guided_model": str(model_path)},
                        meta_extra={"lattice": lat, "mother": m_name, "guided_model": model_name, "guided": g_name},
                    )

    # ---- Cap recipes per family ----
    if MAX_RECIPES_PER_FAMILY is not None:
        by_family: Dict[str, List[str]] = {}
        for name in recipes:
            fam = str(meta.get(name, {}).get("family", "misc"))
            by_family.setdefault(fam, []).append(name)
        keep: List[str] = []
        for fam, names in by_family.items():
            keep.extend(sorted(names)[:MAX_RECIPES_PER_FAMILY])
        recipes = {k: recipes[k] for k in keep}
        meta = {k: meta[k] for k in keep}

    settings: Dict[str, object] = {
        "max_recipes_per_family": MAX_RECIPES_PER_FAMILY,
        "max_lattice_variants": MAX_LATTICE_VARIANTS,
        "lattice_variants_total": len(lattice_variants_all),
        "lattice_variants_used": len(lattice_variants),
        "sa_presets": sorted(sa_presets.keys()),
        "refine_presets": sorted(refine_presets.keys()),
        "block_presets": sorted(block_presets.keys()),
        "lns_presets": sorted(lns_presets.keys()),
        "guided_presets": sorted(guided_presets.keys()),
        "l2o_presets": sorted(l2o_presets.keys()),
        "heatmap_presets": sorted(heatmap_presets.keys()),
        "hc_presets": sorted(hc_presets.keys()),
        "ga_presets": sorted(ga_presets.keys()),
        "mother_presets": sorted(mother_presets.keys()),
    }

    return recipes, meta, settings


# === Receitas (flags do python -m santa_packing.cli.generate_submission) ===
RECIPES, RECIPES_META, RECIPES_SETTINGS = _build_recipe_pool()
RECIPES_SETTINGS.update(
    {
        "meta_init_model": META_INIT_MODEL,
        "heatmap_model": HEATMAP_MODEL,
        "candidate_l2o_models": {k: str(v) for k, v in CANDIDATE_L2O_MODELS.items()},
        "candidate_guided_models": {k: str(v) for k, v in CANDIDATE_GUIDED_MODELS.items()},
    }
)
(RUN_DIR / "recipes.json").write_text(json.dumps(RECIPES, indent=2, sort_keys=True, default=str))
(RUN_DIR / "recipes_meta.json").write_text(json.dumps(RECIPES_META, indent=2, sort_keys=True, default=str))
(RUN_DIR / "recipes_settings.json").write_text(json.dumps(RECIPES_SETTINGS, indent=2, sort_keys=True, default=str))

# Remove receitas que dependem de modelos inexistentes
ACTIVE_RECIPES: Dict[str, Dict[str, object]] = {}
ACTIVE_META: Dict[str, Dict[str, object]] = {}
for name, recipe in RECIPES.items():
    needs = [
        ("l2o_model", recipe.get("l2o_model")),
        ("guided_model", recipe.get("guided_model")),
        ("meta_init_model", recipe.get("meta_init_model")),
        ("heatmap_model", recipe.get("heatmap_model")),
    ]
    missing = [k for k, v in needs if v is not None and not Path(str(v)).exists()]
    if missing:
        print(f"[skip] receita '{name}' (modelos ausentes: {missing})")
        continue
    ACTIVE_RECIPES[name] = recipe
    if name in RECIPES_META:
        ACTIVE_META[name] = RECIPES_META[name]
(RUN_DIR / "active_recipes.json").write_text(json.dumps(ACTIVE_RECIPES, indent=2, sort_keys=True, default=str))
(RUN_DIR / "active_recipes_meta.json").write_text(json.dumps(ACTIVE_META, indent=2, sort_keys=True, default=str))

SUB_DIR = RUN_DIR / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)

# Melhor receita "single" encontrada no sweep (para reutilizar no max_seed_sweep).
BEST_SEED_SWEEP_RECIPE_NAME: str | None = None
BEST_SEED_SWEEP_RECIPE: Dict[str, object] | None = None

# %% Experimento: sweep de submissions (2-stage)
if RUN_SUBMISSION_SWEEP:
    # Sweep em 2 estagios:
    # 1) ranking rapido em n pequeno (p/ cortar combinacoes)
    # 2) rerun somente top-K em nmax=200 (com overlap_check) + opcional ensemble por puzzle
    SWEEP_TOPK = 20  # quantos candidatos do estagio 1 vao para o estagio 2
    TWO_STAGE_SWEEP = True

    sweep_seeds = SWEEP_SEEDS if isinstance(SWEEP_SEEDS, list) else parse_int_list(str(SWEEP_SEEDS))
    if not sweep_seeds:
        raise ValueError("SWEEP_SEEDS is empty")

    jobs = int(SWEEP_JOBS)
    if jobs <= 0:
        jobs = max(1, int(os.cpu_count() or 1))

    (RUN_DIR / "submission_sweep_meta.json").write_text(
        json.dumps(
            {
                "two_stage": TWO_STAGE_SWEEP,
                "jobs": jobs,
                "reuse": bool(SWEEP_REUSE),
                "keep_going": bool(SWEEP_KEEP_GOING),
                "stage1": {
                    "nmax": int(SWEEP_NMAX),
                    "seeds": [int(s) for s in sweep_seeds],
                    "overlap_check": bool(SWEEP_SCORE_OVERLAP_CHECK),
                },
                "stage2": {
                    "nmax": int(SUBMISSION_NMAX),
                    "overlap_check": bool(SUBMISSION_OVERLAP_CHECK),
                    "topk": int(SWEEP_TOPK),
                },
            },
            indent=2,
        )
    )

    def _run_submission_candidate(
        stage: int,
        recipe_name: str,
        recipe: Dict[str, object],
        seed: int,
        *,
        nmax: int,
        check_overlap: bool,
    ) -> tuple[Dict[str, object], str, Path]:
        tag = f"{recipe_name}_seed{seed}"
        out_csv = SUB_DIR / f"stage{stage}_{tag}.csv"
        if not (SWEEP_REUSE and out_csv.exists()):
            generate_submission(out_csv, seed=seed, nmax=nmax, args=recipe)
        score = score_csv(out_csv, nmax=nmax, check_overlap=check_overlap)
        row: Dict[str, object] = {
            "tag": tag,
            "stage": int(stage),
            "recipe": recipe_name,
            "seed": int(seed),
            "nmax": int(nmax),
            "score": score.get("score"),
            "s_max": score.get("s_max"),
            "overlap_check": score.get("overlap_check"),
        }
        return row, tag, out_csv

    stage1_rows: List[Dict[str, object]] = []
    stage1_paths: Dict[str, Path] = {}
    planned_stage1: List[tuple[str, Dict[str, object], int]] = []
    for recipe_name, recipe in ACTIVE_RECIPES.items():
        for seed in sweep_seeds:
            planned_stage1.append((recipe_name, recipe, int(seed)))

    if jobs == 1:
        for recipe_name, recipe, seed in planned_stage1:
            try:
                row, tag, out_csv = _run_submission_candidate(
                    1,
                    recipe_name,
                    recipe,
                    seed,
                    nmax=SWEEP_NMAX,
                    check_overlap=SWEEP_SCORE_OVERLAP_CHECK,
                )
            except Exception as e:
                print(f"[sweep stage1] failed ({recipe_name}, seed={seed}): {e}")
                if not SWEEP_KEEP_GOING:
                    raise
                continue
            stage1_rows.append(row)
            stage1_paths[tag] = out_csv
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            fut_to_item = {
                ex.submit(
                    _run_submission_candidate,
                    1,
                    recipe_name,
                    recipe,
                    seed,
                    nmax=SWEEP_NMAX,
                    check_overlap=SWEEP_SCORE_OVERLAP_CHECK,
                ): (recipe_name, seed)
                for recipe_name, recipe, seed in planned_stage1
            }
            for fut in as_completed(fut_to_item):
                recipe_name, seed = fut_to_item[fut]
                try:
                    row, tag, out_csv = fut.result()
                except Exception as e:
                    print(f"[sweep stage1] failed ({recipe_name}, seed={seed}): {e}")
                    if not SWEEP_KEEP_GOING:
                        raise
                    continue
                stage1_rows.append(row)
                stage1_paths[tag] = out_csv

    stage1_rows = sorted(stage1_rows, key=lambda r: (float(r.get("score") or float("inf")), str(r["tag"])))
    write_csv(RUN_DIR / "submission_sweep_stage1.csv", stage1_rows)

    if not TWO_STAGE_SWEEP:
        # Comportamento antigo (1 estagio): promove o melhor do sweep rapido.
        best = stage1_rows[0] if stage1_rows else None
        if best is not None:
            best_path = stage1_paths[str(best["tag"])]
            shutil.copyfile(best_path, RUN_DIR / "submission_best.csv")
            (RUN_DIR / "submission_best.txt").write_text(json.dumps(best, indent=2))
            BEST_SEED_SWEEP_RECIPE_NAME = str(best.get("recipe"))
            BEST_SEED_SWEEP_RECIPE = ACTIVE_RECIPES.get(BEST_SEED_SWEEP_RECIPE_NAME) if BEST_SEED_SWEEP_RECIPE_NAME else None
            print("Best (sweep stage1):", best)
            print("Saved:", RUN_DIR / "submission_best.csv")

        if SWEEP_BUILD_ENSEMBLE and stage1_paths:
            ens_csv = RUN_DIR / "submission_ensemble.csv"
            ens_meta = _best_per_puzzle_ensemble(ens_csv, stage1_paths, nmax=SWEEP_NMAX, check_overlap=SWEEP_SCORE_OVERLAP_CHECK)
            (RUN_DIR / "submission_ensemble_meta.json").write_text(json.dumps(ens_meta, indent=2))
            ens_score = score_csv(ens_csv, nmax=SWEEP_NMAX, check_overlap=SWEEP_SCORE_OVERLAP_CHECK)
            (RUN_DIR / "submission_ensemble_score.json").write_text(json.dumps(ens_score, indent=2))
            print("Ensemble score (stage1):", ens_score.get("score"))
            print("Saved:", ens_csv)
    else:
        selected = stage1_rows[: int(SWEEP_TOPK)] if stage1_rows else []
        (RUN_DIR / "submission_sweep_selected.json").write_text(json.dumps(selected, indent=2, default=str))
        print(f"Stage1: {len(stage1_rows)} candidates; promoting top {len(selected)} to stage2")

        stage2_rows: List[Dict[str, object]] = []
        stage2_paths: Dict[str, Path] = {}
        planned_stage2: List[tuple[Dict[str, object], str, Dict[str, object], int]] = []
        for row in selected:
            recipe_name = str(row["recipe"])
            seed = int(row["seed"])
            recipe = ACTIVE_RECIPES[recipe_name]
            planned_stage2.append((row, recipe_name, recipe, seed))

        if jobs == 1:
            for row, recipe_name, recipe, seed in planned_stage2:
                try:
                    out_row, tag, out_csv = _run_submission_candidate(
                        2,
                        recipe_name,
                        recipe,
                        seed,
                        nmax=SUBMISSION_NMAX,
                        check_overlap=SUBMISSION_OVERLAP_CHECK,
                    )
                except Exception as e:
                    print(f"[sweep stage2] failed ({recipe_name}, seed={seed}): {e}")
                    if not SWEEP_KEEP_GOING:
                        raise
                    continue
                out_row["stage1_score"] = row.get("score")
                stage2_rows.append(out_row)
                stage2_paths[tag] = out_csv
        else:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                fut_to_item = {
                    ex.submit(
                        _run_submission_candidate,
                        2,
                        recipe_name,
                        recipe,
                        seed,
                        nmax=SUBMISSION_NMAX,
                        check_overlap=SUBMISSION_OVERLAP_CHECK,
                    ): (row, recipe_name, seed)
                    for row, recipe_name, recipe, seed in planned_stage2
                }
                for fut in as_completed(fut_to_item):
                    row, recipe_name, seed = fut_to_item[fut]
                    try:
                        out_row, tag, out_csv = fut.result()
                    except Exception as e:
                        print(f"[sweep stage2] failed ({recipe_name}, seed={seed}): {e}")
                        if not SWEEP_KEEP_GOING:
                            raise
                        continue
                    out_row["stage1_score"] = row.get("score")
                    stage2_rows.append(out_row)
                    stage2_paths[tag] = out_csv

        stage2_rows = sorted(stage2_rows, key=lambda r: (float(r.get("score") or float("inf")), str(r["tag"])))
        write_csv(RUN_DIR / "submission_sweep_stage2.csv", stage2_rows)

        best2 = stage2_rows[0] if stage2_rows else None
        best_score = float(best2.get("score")) if best2 is not None and best2.get("score") is not None else float("inf")
        best_csv: Path | None = None
        best_meta: Dict[str, object] | None = None
        if best2 is not None:
            best_csv = stage2_paths[str(best2["tag"])]
            best_meta = dict(best2)
            shutil.copyfile(best_csv, RUN_DIR / "submission_best.csv")
            (RUN_DIR / "submission_best.txt").write_text(json.dumps(best2, indent=2))
            BEST_SEED_SWEEP_RECIPE_NAME = str(best2.get("recipe"))
            BEST_SEED_SWEEP_RECIPE = ACTIVE_RECIPES.get(BEST_SEED_SWEEP_RECIPE_NAME) if BEST_SEED_SWEEP_RECIPE_NAME else None
            print("Best (stage2):", best2)
            print("Saved:", RUN_DIR / "submission_best.csv")

        if SWEEP_BUILD_ENSEMBLE and stage2_paths:
            ens_csv = RUN_DIR / "submission_ensemble.csv"
            ens_meta = _best_per_puzzle_ensemble(ens_csv, stage2_paths, nmax=SUBMISSION_NMAX, check_overlap=SUBMISSION_OVERLAP_CHECK)
            (RUN_DIR / "submission_ensemble_meta.json").write_text(json.dumps(ens_meta, indent=2))
            ens_score = score_csv(ens_csv, nmax=SUBMISSION_NMAX, check_overlap=SUBMISSION_OVERLAP_CHECK)
            (RUN_DIR / "submission_ensemble_score.json").write_text(json.dumps(ens_score, indent=2))
            print("Ensemble score (stage2):", ens_score.get("score"))
            print("Saved:", ens_csv)
            try:
                ens_total = float(ens_score.get("score")) if ens_score.get("score") is not None else float("inf")
            except Exception:
                ens_total = float("inf")
            if ens_total < best_score:
                shutil.copyfile(ens_csv, RUN_DIR / "submission_best.csv")
                (RUN_DIR / "submission_best.txt").write_text(
                    json.dumps(
                        {
                            "tag": "ensemble",
                            "stage": 2,
                            "score": ens_score.get("score"),
                            "s_max": ens_score.get("s_max"),
                            "overlap_check": ens_score.get("overlap_check"),
                            "selected_by_puzzle": ens_meta.get("selected_by_puzzle"),
                            "candidates": list(stage2_paths.keys()),
                        },
                        indent=2,
                    )
                )
                print("Best updated to ensemble.")
else:
    print("[submission_sweep] RUN_SUBMISSION_SWEEP=False; pule este cell.")


# %% Experimento: gerar submission.csv (single)
if not RUN_SUBMISSION_SWEEP:
    # Rodada unica (use nmax=200 + overlap_check=True para score final)
    def _pick_single_recipe() -> str:
        preferred_families = [
            "guided_block_refine_lns_meta",
            "guided_block_refine_lns",
            "guided_refine_lns_meta",
            "guided_refine_lns",
            "l2o_refine_lns_meta",
            "l2o_refine_lns",
            "sa_refine_lns",
            "refine_lns",
            "block_refine_lns",
            "lns",
            "guided_block_refine_meta",
            "guided_block_refine",
            "block_sa_refine_meta",
            "block_sa_refine",
            "block_refine_meta",
            "block_refine",
            "guided_refine_meta",
            "guided_refine",
            "l2o_refine_meta",
            "l2o_refine",
            "sa_refine_meta",
            "sa_refine",
            "sa",
            "lattice",
        ]
        for fam in preferred_families:
            cands = [k for k, m in ACTIVE_META.items() if str(m.get("family")) == fam]
            if cands:
                return sorted(cands)[0]
        return sorted(ACTIVE_RECIPES)[0]

    SINGLE_RECIPE = _pick_single_recipe()
    BEST_SEED_SWEEP_RECIPE_NAME = str(SINGLE_RECIPE)
    BEST_SEED_SWEEP_RECIPE = ACTIVE_RECIPES.get(BEST_SEED_SWEEP_RECIPE_NAME)
    out_csv = RUN_DIR / "submission.csv"
    generate_submission(out_csv, seed=SUBMISSION_SEED, nmax=SUBMISSION_NMAX, args=ACTIVE_RECIPES[SINGLE_RECIPE])
    score = score_csv(out_csv, nmax=SUBMISSION_NMAX, check_overlap=SUBMISSION_OVERLAP_CHECK)
    (RUN_DIR / "submission_score.json").write_text(json.dumps(score, indent=2))
    print("Submission saved to", out_csv)
    print("Score:", score.get("score"))
else:
    print("[single_submission] RUN_SUBMISSION_SWEEP=True; pule este cell.")

# %% Maximo de experimentos: multi-start + ensemble por n (via python -m santa_packing.cli.sweep_ensemble)
# Objetivo: rodar MUITAS seeds com uma receita forte (mother-prefix + refine no N=200) e
# deixar o ensemble escolher o melhor s_n por puzzle.
RUN_MAX_SEED_SWEEP = False
MAX_SEED_SWEEP_SEEDS = "1..200"  # aumente se quiser (ex.: 1..500)
MAX_SEED_SWEEP_JOBS = max(1, int(os.cpu_count() or 1))  # candidatos em paralelo (ajuste conforme CPU/RAM)
MAX_SEED_SWEEP_TAG = "max_seed_sweep"
MAX_SEED_SWEEP_OUT = RUN_DIR / "submission_max_ensemble.csv"

# Receita forte default (ajuste livremente). Se o sweep de receitas rodar, usamos a melhor receita "single".
DEFAULT_MAX_SEED_SWEEP_RECIPE: Dict[str, object] = {
    "mother_prefix": True,
    "sa_nmax": 0,
    "lattice_pattern": "hex",
    "lattice_margin": 0.005,
    "lattice_rotations": "0,5,10,15,20,25,30",
    "block_nmax": 200,
    "block_size": 2,
    "block_steps": 200,
    "block_batch": 32,
    "block_objective": "prefix",
    "block_init": "cluster",
    "lns_nmax": 200,
    "lns_passes": 10,
    "lns_destroy_k": 8,
    "lns_destroy_mode": "alns",
    "lns_tabu_tenure": 5,
    "lns_candidates": 64,
    "lns_angle_samples": 8,
    "lns_pad_scale": 2.0,
    "lns_group_moves": 4,
    "lns_group_size": 3,
    "lns_group_trans_sigma": 0.05,
    "lns_group_rot_sigma": 20.0,
    "lns_t_start": 0.0,
    "lns_t_end": 0.0,
    "refine_nmin": 200,
    "refine_steps": 5000,
    "refine_batch": 128,
    "refine_objective": "prefix",
    "refine_proposal": "mixed",
    "refine_cooling": "log",
    "refine_trans_sigma": 0.35,
    "refine_rot_sigma": 25.0,
    "refine_rot_prob": 0.4,
    "refine_rot_prob_end": 0.1,
}

if RUN_MAX_SEED_SWEEP:
    if MAX_SEED_SWEEP_OUT.exists():
        print("[max_seed_sweep] skipping: already exists:", MAX_SEED_SWEEP_OUT)
    else:
        max_seed_recipe_name = BEST_SEED_SWEEP_RECIPE_NAME or "default_refine"
        max_seed_recipe = BEST_SEED_SWEEP_RECIPE or DEFAULT_MAX_SEED_SWEEP_RECIPE
        print(f"[max_seed_sweep] base_recipe={max_seed_recipe_name}  seeds={MAX_SEED_SWEEP_SEEDS}  jobs={MAX_SEED_SWEEP_JOBS}")

        recipes_json = RUN_DIR / "max_seed_sweep_recipes.json"
        args_list: List[str] = []
        for key, value in max_seed_recipe.items():
            if value is None:
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args_list.append(flag)
            else:
                args_list += [flag, str(value)]
        recipes_json.write_text(json.dumps([{"name": str(max_seed_recipe_name), "args": args_list}], indent=2))

        run_cmd(
            [
                sys.executable,
                "-m",
                "santa_packing.cli.sweep_ensemble",
                "--repo",
                str(ROOT),
                "--runs-dir",
                str(RUN_DIR),
                "--tag",
                MAX_SEED_SWEEP_TAG,
                "--nmax",
                str(SUBMISSION_NMAX),
                "--seeds",
                str(MAX_SEED_SWEEP_SEEDS),
                "--jobs",
                str(MAX_SEED_SWEEP_JOBS),
                "--recipes-json",
                str(recipes_json),
                "--overlap-check",
                "selected",
                "--keep-going",
                "--out",
                str(MAX_SEED_SWEEP_OUT),
            ]
        )

        max_score = score_csv(MAX_SEED_SWEEP_OUT, nmax=SUBMISSION_NMAX, check_overlap=True)
        (RUN_DIR / "submission_max_ensemble_score.json").write_text(json.dumps(max_score, indent=2))
        print("[max_seed_sweep] score:", max_score.get("score"))
