"""Encoding with existing research SAE checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SAEEncoder:
    """Parameters required to encode activations; no training state."""

    encoder: NDArray[np.float32]
    pre_bias: NDArray[np.float32]
    top_k: int = 100

    def __post_init__(self) -> None:
        if self.encoder.ndim != 2 or min(self.encoder.shape) < 1:
            raise ValueError("encoder must have shape [n_features, hidden_dim]")
        if self.pre_bias.shape != (self.encoder.shape[1],):
            raise ValueError("pre_bias must have shape [hidden_dim]")
        if not isinstance(self.top_k, (int, np.integer)) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")


def load_sae_encoder(path: str | Path, *, top_k: int | None = None) -> SAEEncoder:
    """Read encoder parameters and the training TopK, defaulting to 100.

    An explicit ``top_k`` overrides the checkpoint configuration. The
    checkpoint remains unchanged and is never opened with pickle support.
    """
    with np.load(path, allow_pickle=False) as artifact:
        encoder = np.asarray(artifact["encoder"], dtype=np.float32)
        pre_bias = np.asarray(artifact["pre_bias"], dtype=np.float32)
        if top_k is None:
            config = json.loads(str(artifact["train_config"])) if "train_config" in artifact else {}
            top_k = int(config.get("top_k", 100))
    return SAEEncoder(encoder, pre_bias, top_k)


def encode_activations(activations: ArrayLike, sae: SAEEncoder) -> NDArray[np.float32]:
    """Encode [samples, hidden_dim] activations using the legacy NumPy path.

    Preserves the research viewer's preprocessing, float32 arithmetic,
    and TopK selection, including its handling of tied activations.
    """
    values = np.asarray(activations, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != sae.encoder.shape[1]:
        raise ValueError("activations must have shape [samples, hidden_dim] matching the SAE")
    encoder = np.asarray(sae.encoder, dtype=np.float32)
    centered = values - np.asarray(sae.pre_bias, dtype=np.float32)
    centered = centered - centered.mean(axis=-1, keepdims=True)
    l2 = np.linalg.norm(centered, axis=-1, keepdims=True)
    normalized = centered / np.maximum(l2, 1e-6) * math.sqrt(centered.shape[-1])
    pre = np.maximum(normalized @ encoder.T, 0.0)
    k = min(sae.top_k, pre.shape[-1])
    if k >= pre.shape[-1]:
        return pre
    indices = np.argpartition(pre, -k, axis=-1)[:, -k:]
    encoded = np.zeros_like(pre, dtype=np.float32)
    rows = np.arange(pre.shape[0])[:, None]
    encoded[rows, indices] = pre[rows, indices]
    return encoded
