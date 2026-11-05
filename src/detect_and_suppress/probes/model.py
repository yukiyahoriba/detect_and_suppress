"""Residual representation contracts and NumPy Probe inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ResidualSpec:
    model: str
    family: str
    layer: int
    capture_point: str
    pooling: str
    hidden_dim: int
    image_keys: tuple[str, ...] = ()
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        if not all((self.model, self.family, self.capture_point, self.pooling)):
            raise ValueError("model, family, capture_point and pooling must be specified")
        if self.hidden_dim < 1 or self.layer < 0:
            raise ValueError("invalid residual dimension or layer")


@dataclass(frozen=True)
class Standardizer:
    mean: NDArray
    scale: NDArray

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.scale.shape != self.mean.shape:
            raise ValueError("mean and scale must be matching vectors")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("standardization statistics must be finite, with positive scale")

    def transform(self, values: ArrayLike) -> NDArray:
        """Standardize without changing the input's precision or modifying it."""
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1:] != self.mean.shape:
            raise ValueError("expected [samples, hidden_dim] residual activations")
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError("residual activations must be floating point")
        return (values - self.mean) / self.scale


@dataclass(frozen=True)
class LogisticProbe:
    representation: ResidualSpec
    mean: NDArray
    scale: NDArray
    coef: NDArray
    intercept: np.floating
    threshold: np.floating
    metadata: Mapping[str, object] = field(default_factory=dict)
    extra_arrays: Mapping[str, NDArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Standardizer(self.mean, self.scale)
        if self.mean.shape != (self.representation.hidden_dim,) or self.coef.shape != self.mean.shape:
            raise ValueError("Probe dimension does not match its representation")
        if not np.isfinite(self.coef).all() or not np.isfinite(self.intercept) or not np.isfinite(self.threshold):
            raise ValueError("Probe parameters must be finite")

    def validate_representation(self, representation: ResidualSpec) -> None:
        if representation != self.representation:
            raise ValueError("residual representation does not match the Probe")

    def standardize(self, values: ArrayLike, *, representation: ResidualSpec | None = None) -> NDArray:
        """Optionally verify provenance; dimensions are always checked."""
        if representation is not None:
            self.validate_representation(representation)
        return Standardizer(self.mean, self.scale).transform(values)

    def decision_function(self, values: ArrayLike, *, representation: ResidualSpec | None = None) -> NDArray:
        return self.standardize(values, representation=representation) @ self.coef + self.intercept

    def predict(self, values: ArrayLike, *, representation: ResidualSpec | None = None) -> NDArray[np.bool_]:
        return self.decision_function(values, representation=representation) > self.threshold
