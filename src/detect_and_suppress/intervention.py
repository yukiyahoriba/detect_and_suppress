"""Feature directions and a NumPy reference for residual-stream suppression."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


def feature_direction(decoder: ArrayLike, feature_id: int) -> NDArray[np.float32]:
    """Read a unit direction from a decoder shaped [hidden_dim, n_features]."""
    decoder = np.asarray(decoder)
    if decoder.ndim != 2 or decoder.shape[0] == 0:
        raise ValueError("decoder must have shape [hidden_dim, n_features]")
    if not isinstance(feature_id, (int, np.integer)) or isinstance(feature_id, bool):
        raise TypeError("feature_id must be an integer")
    if not 0 <= feature_id < decoder.shape[1]:
        raise ValueError("feature_id is outside the SAE dictionary")
    direction = np.array(decoder[:, feature_id], dtype=np.float32, copy=True)
    if not np.isfinite(direction).all():
        raise ValueError("feature direction must be finite")
    return direction / np.maximum(np.linalg.norm(direction), np.float32(1e-8))


def load_feature_direction(checkpoint: str | Path, feature_id: int) -> NDArray[np.float32]:
    """Load a decoder direction from an existing SAE NPZ checkpoint."""
    with np.load(checkpoint, allow_pickle=False) as artifact:
        return feature_direction(artifact["decoder"], feature_id)


def _validate_shapes(
    hidden_shape: tuple[int, ...],
    direction_shape: tuple[int, ...],
    active_shape: tuple[int, ...],
    lambda_: float,
) -> None:
    if len(hidden_shape) != 3 or hidden_shape[-1] == 0:
        raise ValueError("hidden must have shape [batch, tokens, hidden_dim]")
    if direction_shape != (hidden_shape[-1],):
        raise ValueError("direction must have shape [hidden_dim]")
    if active_shape not in ((), (hidden_shape[0],)):
        raise ValueError("active must be a scalar or have shape [batch]")
    if not math.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be finite and non-negative")


def suppress_features(
    hidden: NDArray,
    direction: ArrayLike,
    *,
    lambda_: float,
    active: ArrayLike = True,
) -> NDArray:
    """Suppress one SAE direction in a batch of residual tokens.

    ``direction`` identifies the feature to suppress and ``lambda_`` controls
    the intervention strength. ``active`` is a boolean scalar or [batch]
    gate, supplied by a detector or experiment condition. Inputs are not
    modified; inactive samples retain their original values.
    """
    hidden = np.asarray(hidden)
    direction = np.asarray(direction, dtype=np.float32)
    active = np.asarray(active, dtype=bool)
    _validate_shapes(hidden.shape, direction.shape, active.shape, lambda_)
    if not np.issubdtype(hidden.dtype, np.floating):
        raise TypeError("hidden must have a floating-point dtype")
    direction = direction / np.maximum(np.linalg.norm(direction), np.float32(1e-8))
    direction = direction.astype(hidden.dtype)
    norms = np.linalg.norm(hidden.astype(np.float32), axis=-1, keepdims=True).astype(hidden.dtype)
    gate = active.reshape(-1, 1, 1) if active.ndim else active
    return np.where(gate, hidden - lambda_ * norms * direction, hidden)
