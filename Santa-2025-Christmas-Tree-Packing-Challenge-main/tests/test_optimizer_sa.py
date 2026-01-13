import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp

    from santa_packing.collisions import check_any_collisions
    from santa_packing.lattice import lattice_poses
    from santa_packing.optimizer import run_sa_batch
    from santa_packing.packing import packing_score
    from santa_packing.tree import get_tree_polygon
except Exception:  # pragma: no cover
    pytest.skip("JAX not available", allow_module_level=True)


def test_run_sa_batch_shapes_and_feasibility() -> None:
    n_trees = 8
    batch_size = 2

    poses_np = lattice_poses(n_trees, pattern="hex", margin=0.02, rotate_deg=0.0)
    initial_single = jnp.array(poses_np)
    initial = jnp.tile(initial_single[None, :, :], (batch_size, 1, 1))

    base_poly = get_tree_polygon()
    assert bool(check_any_collisions(initial[0], base_poly).item()) is False

    init_score = packing_score(initial[0])

    best_poses, best_scores = run_sa_batch(
        jax.random.PRNGKey(0),
        10,
        n_trees,
        initial,
        trans_sigma=0.05,
        rot_sigma=15.0,
        rot_prob=0.2,
        cooling="geom",
        proposal="random",
        allow_collisions=False,
        objective="packing",
    )

    assert best_poses.shape == initial.shape
    assert best_scores.shape == (batch_size,)
    assert np.all(np.asarray(best_scores) <= float(init_score) + 1e-6)

    collided = jax.vmap(lambda p: check_any_collisions(p, base_poly))(best_poses)
    assert bool(jnp.any(collided).item()) is False

    deg = best_poses[..., 2]
    assert bool(jnp.all((deg >= 0.0) & (deg < 360.0)).item()) is True
