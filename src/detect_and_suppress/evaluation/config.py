"""Explicit LIBERO-10 experiment identity, independent of worker partitioning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from ..defense import DefenseConfig
from ..probes import ResidualSpec
from .patches import OverlaySpec


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str

    @classmethod
    def identify(cls, path: str | Path) -> Artifact:
        """Hash file contents, or a directory's sorted relative paths and contents."""
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            if path.is_dir():
                digest.update(str(file.relative_to(path)).encode() + b"\0")
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
        return cls(str(path), digest.hexdigest())

    def verify(self) -> None:
        if self != self.identify(self.path):
            raise ValueError(f"artifact contents changed: {self.path}")


@dataclass(frozen=True)
class EvaluationConfig:
    model: str
    condition: str
    defense: DefenseConfig
    artifacts: dict[str, Artifact]
    feature_id: int
    seed: int = 1
    overlay_seed: int = 1
    patch_mode: str = "bursts"
    patch_windows: int = 3
    patch_window_frames: int = 10
    replan_steps: int = 5
    denoising_steps: int = 10
    num_steps_wait: int = 10
    max_steps: int = 520
    resize_size: int = 224
    config_name: str = "pi05_libero"
    timing: str = "prefix"
    task_suite: str = "libero_10"
    episodes_per_task: int = 50
    # Revisions, simulator versions and old condition/config references belong here.
    provenance: dict | None = None

    def __post_init__(self):
        counts = (self.feature_id, self.seed, self.overlay_seed, self.replan_steps, self.denoising_steps,
                  self.num_steps_wait, self.max_steps, self.resize_size, self.episodes_per_task,
                  self.patch_windows, self.patch_window_frames)
        if any(type(value) is not int for value in counts) or min(self.feature_id, self.seed, self.overlay_seed) < 0:
            raise ValueError("provide integer counts and non-negative feature/seed identifiers")
        if self.model not in ("pi05", "smolvla") or self.task_suite != "libero_10" or self.episodes_per_task != 50:
            raise ValueError("evaluation supports the paper's LIBERO-10, 50 initial states/task")
        if self.defense.detector.model != self.model or not self.condition:
            raise ValueError("provide matching defense model and a named paper condition")
        if not {"checkpoint", "probe", "sae", "selection"} <= self.artifacts.keys():
            raise ValueError("checkpoint, probe, sae and selection identities are required")
        if self.patch_mode not in ("none", "always", "bursts"):
            raise ValueError("invalid patch mode")
        if self.patch_mode != "none" and "patch" not in self.artifacts:
            raise ValueError("patch identity is required")
        if self.patch_windows not in (3, 10) or self.patch_window_frames != 10:
            raise ValueError("paper schedules use 3/10 bursts of 10 frames")
        if min(self.replan_steps, self.denoising_steps, self.max_steps, self.resize_size) < 1 or self.num_steps_wait < 0:
            raise ValueError("invalid rollout dimensions/counts")
        if self.timing not in ("prefix", "denoising"):
            raise ValueError("unsupported intervention timing")
        if self.model == "smolvla" and (self.replan_steps != 1 or self.timing != "prefix"):
            raise ValueError("SmolVLA paper evaluation requires replan_steps=1 and prefix timing")
        if self.model == "pi05" and self.overlay_seed != self.seed:
            raise ValueError("Pi0.5 legacy rendering and schedule use the policy seed")
        if self.timing == "denoising" and (self.defense.mode == "probe" or self.defense.detector.family == "vit"):
            raise ValueError("repeated-prefix timing supports PaliGemma Oracle/always/disabled")

    @property
    def schedule_horizon(self) -> int:
        return self.max_steps if self.model == "pi05" else (90 if self.patch_windows == 3 else 109)

    @property
    def required_patch_forwards(self) -> int | None:
        return self.patch_windows * self.patch_window_frames if self.model == "smolvla" and self.patch_mode == "bursts" else None

    def overlay(self) -> OverlaySpec:
        return OverlaySpec(mode=self.patch_mode,
                           patch_path=Path(self.artifacts["patch"].path) if "patch" in self.artifacts else Path("."),
                           seed=self.overlay_seed, short_windows=self.patch_windows,
                           short_window_frames=self.patch_window_frames, non_overlapping_windows=True,
                           **self.protocol["patch_geometry"])

    @property
    def protocol(self) -> dict:
        """Fixed schema-1 paper semantics, expanded for review in saved configs."""
        return {
            "version": 1, "environment_resolution": 256,
            "patch_geometry": {"min_scale": .61, "max_scale": 1.39, "max_rotation_deg": 30.,
                               "max_shear": .2, "no_geometry_prob": .2},
            "patch_geometry_scope": "episode", "shared_camera_geometry": True,
            "non_overlapping_windows": True,
            "patch_cameras": ["agent"] if self.model == "pi05" else ["agent", "wrist"],
            "schedule_horizon": self.schedule_horizon, "schedule_unit": "environment_step",
            "queue_flush_on_visibility_change": self.model == "pi05",
            "success_rule": "done" if self.model == "pi05" else "info.is_success",
            "noise_scheme": ("jax.fold_in(seed, rollout_id); split once per policy forward" if self.model == "pi05" else
                             "torch CPU/CUDA seed + rollout_id before reset; CUDA randn once per policy forward"),
        }

    def to_dict(self) -> dict:
        return {"schema_version": 1, **asdict(self), "protocol": self.protocol}

    @classmethod
    def from_dict(cls, value: dict) -> EvaluationConfig:
        value = dict(value)
        protocol = value.pop("protocol", None)
        if value.pop("schema_version") != 1:
            raise ValueError("unsupported evaluation configuration schema")
        defense = dict(value["defense"])
        for name in ("detector", "intervention"):
            spec = dict(defense[name])
            spec["image_keys"] = tuple(spec.get("image_keys", ()))
            defense[name] = ResidualSpec(**spec)
        value["defense"] = DefenseConfig(**defense)
        value["artifacts"] = {name: Artifact(**artifact) for name, artifact in value["artifacts"].items()}
        result = cls(**value)
        if protocol is not None and canonical(protocol) != canonical(result.protocol):
            raise ValueError("unsupported or modified schema-1 evaluation protocol")
        return result

    @property
    def fingerprint(self) -> str:
        # Protocol is a validated expansion of schema/model/settings, not an
        # independent option. Preserve identities of earlier schema-1 records.
        return hashlib.sha256(canonical({"schema_version": 1, **asdict(self)}).encode()).hexdigest()


@dataclass(frozen=True)
class Shard:
    index: int = 0
    count: int = 1

    def __post_init__(self):
        if type(self.count) is not int or type(self.index) is not int or self.count < 1 or not 0 <= self.index < self.count:
            raise ValueError("invalid shard index/count")

    @property
    def rollout_ids(self) -> tuple[int, ...]:
        return tuple(i for i in range(500) if i % self.count == self.index)
