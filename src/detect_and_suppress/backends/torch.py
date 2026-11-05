"""PyTorch operations for the SmolVLA integration."""

from __future__ import annotations

import torch
from torch import Tensor

from ..intervention import _validate_shapes


def suppress_features(
    hidden: Tensor,
    direction: Tensor,
    *,
    lambda_: float,
    active: Tensor | bool = True,
    normalize_direction: bool = True,
) -> Tensor:
    """Apply feature suppression on device, preserving the residual dtype.

    ``direction`` identifies the feature and ``lambda_`` controls the
    intervention strength. ``hidden`` is [batch, tokens, hidden_dim];
    ``active`` is a boolean scalar or [batch] gate. Model integrations may
    disable direction preparation when supplying a previously prepared vector.
    No input is mutated.
    """
    scalar_gate = float(active) if isinstance(active, bool) else None
    direction = torch.as_tensor(direction, device=hidden.device, dtype=torch.float32)
    active = torch.as_tensor(active, device=hidden.device, dtype=torch.bool)
    _validate_shapes(tuple(hidden.shape), tuple(direction.shape), tuple(active.shape), lambda_)
    if not hidden.is_floating_point():
        raise TypeError("hidden must have a floating-point dtype")
    if normalize_direction:
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)
    direction = direction.to(dtype=hidden.dtype)
    norms = torch.linalg.vector_norm(hidden.float(), dim=-1, keepdim=True).to(hidden.dtype)
    if scalar_gate is not None:
        return hidden - scalar_gate * lambda_ * norms * direction
    gate = active.to(hidden.dtype)
    gate = gate[:, None, None] if gate.ndim else gate
    return hidden - gate * lambda_ * norms * direction
