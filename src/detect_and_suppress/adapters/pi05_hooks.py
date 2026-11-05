"""JAX hooks called by the small OpenPI integration patch."""

from __future__ import annotations

import jax.numpy as jnp

from ..backends.jax import suppress_features


def apply_defense_hook(xs, layer_idx, config):
    """Apply a prepared defense while preserving the original capture location."""
    expert = int(config.get("expert", 0))
    hidden = xs[expert]
    if hidden is None:
        raise ValueError("defense requires the prefix residual stream")
    pooled = jnp.mean(hidden, axis=1)
    gate = jnp.asarray(config.get("intervention_active", False), dtype=jnp.bool_)
    if gate.ndim == 3:
        gate = gate[:, 0, 0]
    if config.get("detector_mean") is not None:
        score = ((pooled.astype(jnp.float32) - config["detector_mean"]) / config["detector_scale"]
                 ) @ config["detector_coef"] + config["detector_intercept"]
        gate = jnp.where(layer_idx == config["detector_layer"], score > config["detector_threshold"], gate)
    mode = config["defense_mode"]
    if mode == "always":
        gate = jnp.ones((hidden.shape[0],), dtype=jnp.bool_)
    elif mode == "disabled":
        gate = jnp.zeros((hidden.shape[0],), dtype=jnp.bool_)
    elif mode == "oracle":
        gate = jnp.asarray(config["defense_active"], dtype=jnp.bool_)
    elif mode != "probe":
        raise ValueError("unknown defense mode")
    applied = jnp.broadcast_to(gate, (hidden.shape[0],)) & (layer_idx == config["layer"])
    updated = suppress_features(hidden, jnp.asarray(config["direction"]), lambda_=config["alpha"], active=applied)
    outputs = list(xs)
    outputs[expert] = updated
    capture = bool(config.get("capture_defense", False))
    return outputs, {"pooled": pooled, "applied": applied,
                     "before": hidden if capture else hidden[:, :0, :],
                     "after": updated if capture else updated[:, :0, :]}
