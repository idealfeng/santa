"""Heatmap meta-optimizer (NumPy).

This module implements a simple "heatmap" policy that learns which tree indices
to perturb more often during a local search. It is intentionally lightweight
(NumPy-only) and is used via CLI tools for experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .geom_np import packing_score, polygon_bbox, shift_poses_to_origin, transform_polygon
from .scoring import polygons_intersect
from .tree_data import TREE_POINTS

Params = Dict[str, np.ndarray]


@dataclass(frozen=True)
class HeatmapConfig:
    """Configuration for the heatmap policy and the search loop."""

    hidden_size: int = 32
    policy: str = "gnn"  # "mlp" or "gnn"
    knn_k: int = 4
    heatmap_lr: float = 0.1
    trans_sigma: float = 0.2
    rot_sigma: float = 10.0
    t_start: float = 1.0
    t_end: float = 0.001


def init_params(rng: np.random.Generator, hidden_size: int = 32, policy: str = "gnn") -> Params:
    """Initialize heatmap network parameters.

    Args:
        rng: NumPy random generator.
        hidden_size: Hidden layer width.
        policy: `"mlp"` or `"gnn"`.

    Returns:
        Parameter dict of NumPy arrays.
    """

    def r(shape):
        return rng.normal(scale=0.1, size=shape)

    if policy == "gnn":
        return {
            "w_in": r((10, hidden_size)),
            "b_in": np.zeros((hidden_size,)),
            "w_msg": r((hidden_size, hidden_size)),
            "b_msg": np.zeros((hidden_size,)),
            "w_out": r((hidden_size, 1)),
            "b_out": np.zeros((1,)),
        }
    return {
        "w1": r((10, hidden_size)),
        "b1": np.zeros((hidden_size,)),
        "w2": r((hidden_size, 1)),
        "b2": np.zeros((1,)),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / (np.sum(exp) + 1e-9)


def _knn_indices(xy: np.ndarray, k: int) -> np.ndarray:
    n = xy.shape[0]
    dists = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    dists = dists + np.eye(n) * 1e9
    return np.argsort(dists, axis=1)[:, :k]


def _compute_features(
    poses: np.ndarray,
    last_collision: np.ndarray,
    step_frac: float,
) -> np.ndarray:
    xy = poses[:, :2]
    theta = np.deg2rad(poses[:, 2])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # bbox + normalization
    polys = [transform_polygon(np.array(TREE_POINTS, dtype=float), pose) for pose in poses]
    bboxes = np.array([polygon_bbox(p) for p in polys])
    min_x = float(np.min(bboxes[:, 0]))
    min_y = float(np.min(bboxes[:, 1]))
    max_x = float(np.max(bboxes[:, 2]))
    max_y = float(np.max(bboxes[:, 3]))
    width = max_x - min_x
    height = max_y - min_y
    max_side = max(width, height, 1e-6)
    center = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5])
    xy_norm = (xy - center) / max_side
    dist_center = np.linalg.norm(xy - center, axis=1) / max_side

    # nearest neighbor distance
    dists = np.sqrt(np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    nn_dist = np.min(dists, axis=1) / max_side

    node_feats = np.stack(
        [
            xy_norm[:, 0],
            xy_norm[:, 1],
            sin_t,
            cos_t,
            nn_dist,
            last_collision,
            dist_center,
        ],
        axis=1,
    )
    # global features
    globals_feat = np.array(
        [
            step_frac,
            width / max_side,
            height / max_side,
        ],
        dtype=float,
    )
    globals_rep = np.repeat(globals_feat[None, :], poses.shape[0], axis=0)
    return np.concatenate([node_feats, globals_rep], axis=1)


def _mlp_update(params: Params, feats: np.ndarray) -> np.ndarray:
    h = np.tanh(feats @ params["w1"] + params["b1"])
    out = h @ params["w2"] + params["b2"]
    return out.squeeze(-1)


def _gnn_update(params: Params, feats: np.ndarray, knn_k: int) -> np.ndarray:
    h0 = np.tanh(feats @ params["w_in"] + params["b_in"])
    idx = _knn_indices(feats[:, 0:2], knn_k)
    neigh = h0[idx]
    agg = np.mean(neigh, axis=1)
    h1 = np.tanh(h0 + agg @ params["w_msg"] + params["b_msg"])
    out = h1 @ params["w_out"] + params["b_out"]
    return out.squeeze(-1)


def update_heatmap(
    params: Params,
    poses: np.ndarray,
    theta: np.ndarray,
    last_collision: np.ndarray,
    step_frac: float,
    config: HeatmapConfig,
) -> np.ndarray:
    """Update heatmap logits based on current state and last-step feedback.

    Args:
        params: Heatmap network parameters.
        poses: Current poses `(N, 3)`.
        theta: Current heatmap logits `(N,)` (higher => sampled more often).
        last_collision: Per-index collision indicator `(N,)` from the last proposal.
        step_frac: Progress fraction in `[0, 1]`.
        config: Heatmap configuration.

    Returns:
        Updated heatmap logits `(N,)`.
    """
    feats = _compute_features(poses, last_collision, step_frac)
    if config.policy == "gnn":
        delta = _gnn_update(params, feats, config.knn_k)
    else:
        delta = _mlp_update(params, feats)
    return theta + config.heatmap_lr * delta


def _check_overlaps(points: np.ndarray, poses: np.ndarray) -> bool:
    polys = [transform_polygon(points, pose) for pose in poses]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_intersect(polys[i], polys[j]):
                return True
    return False


def heatmap_search(
    params: Params,
    poses: np.ndarray,
    config: HeatmapConfig,
    steps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Run a heatmap-guided local search starting from `poses`.

    Args:
        params: Heatmap network parameters.
        poses: Initial poses `(N, 3)`.
        config: Search configuration.
        steps: Number of search steps.
        rng: NumPy random generator.

    Returns:
        Tuple `(best_poses, best_score)`.
    """
    points = np.array(TREE_POINTS, dtype=float)
    poses = shift_poses_to_origin(points, poses)
    theta = np.zeros((poses.shape[0],), dtype=float)
    current_score = packing_score(points, poses)
    best_score = current_score
    best_poses = poses.copy()
    last_collision = np.zeros((poses.shape[0],), dtype=float)

    for step in range(steps):
        frac = step / max(steps, 1)
        temp = config.t_start * (config.t_end / config.t_start) ** frac
        probs = _softmax(theta)
        idx = rng.choice(poses.shape[0], p=probs)

        delta = rng.normal(size=(3,))
        delta[0:2] *= config.trans_sigma * temp
        delta[2] *= config.rot_sigma * temp

        candidate = poses.copy()
        candidate[idx] = candidate[idx] + delta
        candidate[idx, 2] = np.mod(candidate[idx, 2], 360.0)

        collided = _check_overlaps(points, candidate)
        accept = False
        cand_score = current_score
        if not collided:
            cand_score = packing_score(points, candidate)
            dscore = cand_score - current_score
            if dscore < 0 or rng.random() < math.exp(-dscore / max(temp, 1e-9)):
                accept = True

        if accept:
            poses = candidate
            current_score = cand_score
            if current_score < best_score:
                best_score = current_score
                best_poses = poses.copy()

        last_collision = np.zeros_like(last_collision)
        last_collision[idx] = 1.0 if collided else 0.0
        theta = update_heatmap(params, poses, theta, last_collision, frac, config)

    return best_poses, float(best_score)


def save_params(path, params: Params, meta: Dict[str, object] | None = None) -> None:
    """Save heatmap parameters to a `.npz` file."""
    payload = {k: np.array(v) for k, v in params.items()}
    if meta:
        for key, value in meta.items():
            payload[f"meta/{key}"] = np.array(value)
    np.savez(path, **payload)


def load_params(path) -> Tuple[Params, Dict[str, object]]:
    """Load heatmap parameters from a `.npz` file."""
    data = np.load(path)
    params: Params = {}
    meta: Dict[str, object] = {}
    for key in data.files:
        if key.startswith("meta/"):
            meta[key.split("/", 1)[1]] = data[key].item() if data[key].shape == () else data[key]
        else:
            params[key] = np.array(data[key])
    return params, meta
