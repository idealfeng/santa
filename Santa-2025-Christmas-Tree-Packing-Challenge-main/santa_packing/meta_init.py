"""Meta-initialization model for SA/L2O pipelines.

The goal of meta-init is to provide a better starting configuration for an
optimizer (typically SA) by applying a learned, bounded delta to the current
poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp

Params = Dict[str, jnp.ndarray]


@dataclass(frozen=True)
class MetaInitConfig:
    """Configuration for the meta-initialization update."""

    hidden_size: int = 32
    delta_xy: float = 0.2
    delta_theta: float = 10.0


def init_meta_params(key: jax.Array, hidden_size: int = 32) -> Params:
    """Initialize meta-init MLP parameters.

    Args:
        key: JAX PRNGKey.
        hidden_size: Hidden layer width.

    Returns:
        Parameter dict containing weights/biases.
    """
    key1, key2 = jax.random.split(key)
    w1 = jax.random.normal(key1, (4, hidden_size)) * 0.1
    b1 = jnp.zeros((hidden_size,))
    w2 = jax.random.normal(key2, (hidden_size, 3)) * 0.1
    b2 = jnp.zeros((3,))
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def _features(poses: jax.Array) -> jax.Array:
    x = poses[:, 0:1]
    y = poses[:, 1:2]
    theta = jnp.deg2rad(poses[:, 2:3])
    return jnp.concatenate([x, y, jnp.sin(theta), jnp.cos(theta)], axis=1)


def apply_meta_init(params: Params, poses: jax.Array, config: MetaInitConfig) -> jax.Array:
    """Apply a bounded learned delta to poses.

    Args:
        params: Parameter dict produced by `init_meta_params`.
        poses: Array `(N, 3)` of `[x, y, theta_deg]`.
        config: Scale limits for the update.

    Returns:
        Updated poses `(N, 3)` with angles wrapped into `[0, 360)`.
    """
    feats = _features(poses)
    h = jnp.tanh(feats @ params["w1"] + params["b1"])
    raw = h @ params["w2"] + params["b2"]
    scale = jnp.array([config.delta_xy, config.delta_xy, config.delta_theta])
    delta = jnp.tanh(raw) * scale
    out = poses + delta
    out = out.at[:, 2].set(jnp.mod(out[:, 2], 360.0))
    return out


def save_meta_params(path, params: Params, meta: Dict[str, object] | None = None) -> None:
    """Save meta-init parameters to a `.npz` file.

    Args:
        path: Output path (string or Path-like).
        params: Parameter dict.
        meta: Optional metadata dict stored under `meta/<key>` entries.
    """
    import numpy as np

    payload = {k: np.array(v) for k, v in params.items()}
    if meta:
        for key, value in meta.items():
            payload[f"meta/{key}"] = np.array(value)
    np.savez(path, **payload)


def load_meta_params(path) -> tuple[Params, Dict[str, object]]:
    """Load meta-init parameters from a `.npz` file.

    Args:
        path: Input path (string or Path-like).

    Returns:
        Tuple `(params, meta)` where `params` is a JAX array dict and `meta` is a
        plain-Python metadata dict.
    """
    import numpy as np

    data = np.load(path)
    params: Params = {}
    meta: Dict[str, object] = {}
    for key in data.files:
        if key.startswith("meta/"):
            meta[key.split("/", 1)[1]] = data[key].item() if data[key].shape == () else data[key]
        else:
            params[key] = jnp.array(data[key])
    return params, meta
