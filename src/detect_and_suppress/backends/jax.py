"""JAX operations for the Pi0.5/OpenPI integration."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..intervention import _validate_shapes


def suppress_features(
    hidden: jax.Array,
    direction: jax.Array,
    *,
    lambda_: float,
    active: jax.Array | bool = True,
) -> jax.Array:
    """Apply feature suppression inside an OpenPI forward.

    ``direction`` identifies the feature and ``lambda_`` controls the
    intervention strength. ``hidden`` is [batch, tokens, hidden_dim];
    ``active`` is a boolean scalar or [batch] gate. Under jax.jit, lambda_
    must be static, as in the existing OpenPI steering methods.
    """
    direction = jnp.asarray(direction, dtype=hidden.dtype)
    active = jnp.asarray(active, dtype=jnp.bool_)
    _validate_shapes(hidden.shape, direction.shape, active.shape, lambda_)
    if not jnp.issubdtype(hidden.dtype, jnp.floating):
        raise TypeError("hidden must have a floating-point dtype")
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-8)
    norms = jnp.linalg.norm(hidden.astype(jnp.float32), axis=-1, keepdims=True).astype(hidden.dtype)
    gate = active[:, None, None] if active.ndim else active
    return jnp.where(gate, hidden - lambda_ * norms * direction, hidden)
