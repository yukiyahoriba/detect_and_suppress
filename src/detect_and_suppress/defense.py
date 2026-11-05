"""Defense artifacts, configuration, and inference records."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .intervention import load_feature_direction
from .probes import LogisticProbe, ResidualSpec, load_probe
from .selection import FeatureSummary, load_selection


DefenseMode = Literal["probe", "always", "disabled", "oracle"]


@dataclass(frozen=True)
class DefenseConfig:
    detector: ResidualSpec
    intervention: ResidualSpec
    lambda_: float
    mode: DefenseMode = "probe"

    def __post_init__(self) -> None:
        validate_mode(self.mode)
        if not math.isfinite(self.lambda_) or self.lambda_ < 0:
            raise ValueError("lambda_ must be finite and non-negative")
        if self.detector.model != self.intervention.model:
            raise ValueError("detector and intervention must use the same VLA model")
        a, b = self.detector.checkpoint, self.intervention.checkpoint
        if a is not None and b is not None and a != b:
            raise ValueError("detector and intervention checkpoint identities differ")


def validate_mode(mode: str) -> None:
    if mode not in ("probe", "always", "disabled", "oracle"):
        raise ValueError("mode must be probe, always, disabled, or oracle")


@dataclass(frozen=True)
class DefenseBundle:
    config: DefenseConfig
    probe: LogisticProbe
    feature: FeatureSummary
    direction: NDArray[np.float32]
    decoder_direction: NDArray[np.float32]
    sae_checkpoint: str

    def __post_init__(self) -> None:
        self.probe.validate_representation(self.config.detector)
        spec = self.config.intervention
        expert = "pg" if spec.family == "llm" and spec.model.startswith("pi05") else spec.family
        if self.feature.layer != spec.layer or self.feature.expert != expert:
            raise ValueError("selected feature does not belong to the intervention stream/layer")
        for value in (self.direction, self.decoder_direction):
            if value.shape != (spec.hidden_dim,) or not np.isfinite(value).all():
                raise ValueError("SAE direction does not match the intervention dimension")


def load_defense(
    *, probe: str | Path, selection: str | Path, sae_checkpoint: str | Path,
    config: DefenseConfig, feature_id: int | None = None,
) -> DefenseBundle:
    """Bind selected features and a Probe to explicitly specified VLA residuals.

    Select one feature explicitly when a ranking contains multiple entries.
    Recorded SAE identities are checked; older metadata without an identity
    are explicitly bound to the caller-supplied checkpoint and representation.
    """
    selected = load_selection(selection)
    expert = "pg" if config.intervention.family == "llm" and config.intervention.model.startswith("pi05") else config.intervention.family
    candidates = [item for item in selected.features if item.layer == config.intervention.layer and item.expert == expert
                  and (feature_id is None or item.feature_id == feature_id)]
    if len(candidates) != 1:
        raise ValueError("select exactly one feature using feature_id and its intervention layer")
    feature = candidates[0]
    path = Path(sae_checkpoint).resolve()
    metadata = selected.metadata
    recorded_hash = metadata.get("sae_sha256")
    if recorded_hash is not None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != recorded_hash:
            raise ValueError("selected feature's SAE checkpoint hash differs")
    recorded_path = metadata.get("sae_checkpoint", metadata.get("sae"))
    if recorded_path is not None and recorded_hash is None:
        recorded_path = Path(recorded_path)
        if not recorded_path.is_absolute():
            recorded_path = Path(selection).resolve().parent / recorded_path
        if recorded_path.resolve() != path:
            raise ValueError("selected feature's SAE checkpoint path differs")
    with np.load(path, allow_pickle=False) as artifact:
        decoder = artifact["decoder"]
        if decoder.ndim != 2 or not 0 <= feature.feature_id < decoder.shape[1]:
            raise ValueError("selected feature is outside the SAE dictionary")
        # Preserve the decoder column layout: Torch's CPU norm reduction can
        # differ between strided columns and contiguous copies in float32.
        raw = np.asarray(decoder, dtype=np.float32)[:, feature.feature_id]
    direction = load_feature_direction(path, feature.feature_id)
    return DefenseBundle(config, load_probe(probe, representation=config.detector), feature,
                         direction, raw, str(path))


@dataclass(frozen=True)
class DefenseRecord:
    episode_id: str | int | None
    forward_id: int
    mode: DefenseMode
    detector_layer: int
    intervention_layer: int
    scores: NDArray
    detected: NDArray
    applied: NDArray
    intervention_calls: int
    timing: str
    before: object | None = None
    after: object | None = None


class DefenseHistory:
    """Bounded host records; only last_record retains optional device captures."""

    def __init__(self, size: int = 128):
        if size < 1:
            raise ValueError("history_size must be positive")
        self._records = deque(maxlen=size)
        self.last_record: DefenseRecord | None = None
        self.episode_id: str | int | None = None
        self.forward_id = 0

    @property
    def history(self) -> tuple[DefenseRecord, ...]:
        return tuple(self._records)

    def append(self, record: DefenseRecord) -> None:
        self.last_record = record
        self._records.append(replace(record, before=None, after=None))
        self.forward_id += 1

    def reset(self, episode_id: str | int | None) -> None:
        self._records.clear()
        self.last_record = None
        self.episode_id, self.forward_id = episode_id, 0


def check_checkpoint(bundle: DefenseBundle, checkpoint: str | Path | None) -> None:
    for spec in (bundle.config.detector, bundle.config.intervention):
        if spec.checkpoint is not None and (checkpoint is None or str(checkpoint) != spec.checkpoint):
            raise ValueError("supply the checkpoint identity declared by the defense configuration")
