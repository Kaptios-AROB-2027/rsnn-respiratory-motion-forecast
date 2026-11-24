from __future__ import annotations

import jax
import jax.numpy as jnp


def clip_by_global_norm(tree, max_norm: float):
    leaves = jax.tree.leaves(tree)
    norm = jnp.sqrt(sum(jnp.sum(x * x) for x in leaves))
    scale = jnp.where(norm > max_norm, max_norm / jnp.maximum(norm, 1e-30), 1.0)
    return jax.tree.map(lambda x: x * scale.astype(x.dtype), tree), norm


def analog_window(z, i, k: int):
    """u_i = [1, z(i), ..., z(i+k-1)] time-major, size 9k+1."""
    seg = jax.lax.dynamic_slice(z, (i, 0), (k, z.shape[1])).reshape(-1)
    return jnp.concatenate([jnp.ones((1,), z.dtype), seg])
