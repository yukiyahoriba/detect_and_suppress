"""Residual datasets and the three legacy collection layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from .model import ResidualSpec


@dataclass(frozen=True)
class ProbeDataset:
    clean: NDArray
    attack: NDArray
    clean_task_ids: NDArray | None = None
    attack_task_ids: NDArray | None = None
    clean_episode_ids: NDArray | None = None
    attack_episode_ids: NDArray | None = None

    def __post_init__(self) -> None:
        if self.clean.ndim != 2 or self.attack.ndim != 2 or self.clean.shape[1] != self.attack.shape[1]:
            raise ValueError("clean and attack must be [samples, hidden_dim] arrays")
        for condition in ("clean", "attack"):
            values = getattr(self, condition)
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("legacy training datasets require finite float32 residuals; convert inputs explicitly")
            for suffix in ("task_ids", "episode_ids"):
                ids = getattr(self, f"{condition}_{suffix}")
                if ids is not None and ids.shape != (len(values),):
                    raise ValueError(f"{condition}_{suffix} must align with its activation rows")

    def training_arrays(self) -> tuple[NDArray, NDArray]:
        return (np.concatenate([self.clean, self.attack]).astype(np.float64),
                np.r_[np.zeros(len(self.clean)), np.ones(len(self.attack))])


@dataclass(frozen=True)
class ProbeSplits:
    representation: ResidualSpec
    train: ProbeDataset
    dev: ProbeDataset
    test: ProbeDataset | None = None
    split_unit: Literal["task", "episode", "prespecified"] = "prespecified"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        splits = [self.train, self.dev] + ([self.test] if self.test is not None else [])
        if self.split_unit not in ("task", "episode", "prespecified"):
            raise ValueError("unknown split_unit")
        for split in splits:
            if split.clean.shape[1] != self.representation.hidden_dim:
                raise ValueError("dataset dimension does not match the residual representation")
        # Paired clean/attack rows may share IDs within a split, never across splits.
        for unit in ("task", "episode"):
            ids = [[getattr(split, f"{condition}_{unit}_ids") for condition in ("clean", "attack")]
                   for split in splits]
            available = all(item is not None for pair in ids for item in pair)
            if unit == self.split_unit and not available:
                raise ValueError(f"{unit} split requires IDs for both conditions in each split")
            if available and (unit == self.split_unit or unit == "episode"):
                groups = [set(np.concatenate(pair).tolist()) for pair in ids]
                if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i)):
                    raise ValueError(f"{unit} IDs overlap across data splits")


def task_fold(fold: int = 0) -> dict[str, set[int]]:
    if not isinstance(fold, int) or not 0 <= fold < 5:
        raise ValueError("fold must be in [0, 4]")
    groups = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    test, dev = set(groups[fold]), set(groups[(fold + 1) % 5])
    return {"train": set(range(10)) - test - dev, "dev": dev, "test": test}


def _dataset(chunks: dict[str, list], dim: int) -> ProbeDataset:
    def join(name):
        return np.concatenate(chunks[name]) if chunks[name] else np.empty((0, dim), dtype=np.float32)
    return ProbeDataset(join("clean"), join("attack"),
                        *(np.concatenate(chunks[name]).astype(np.int64) if chunks[name] else np.empty(0, np.int64)
                          for name in ("clean_task_ids", "attack_task_ids", "clean_episode_ids", "attack_episode_ids")))


def _chunks():
    return {name: [] for name in ("clean", "attack", "clean_task_ids", "attack_task_ids",
                                   "clean_episode_ids", "attack_episode_ids")}


def _append(chunks, condition, values, task, episode):
    chunks[condition].append(values)
    chunks[f"{condition}_task_ids"].append(np.full(len(values), task))
    chunks[f"{condition}_episode_ids"].append(np.full(len(values), episode))


def load_dense_collection(root: str | Path, *, representation: ResidualSpec) -> ProbeSplits:
    """Read manifest activation offsets and the recorded attack windows."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if representation.model not in ("pi05", manifest["model"]) or representation.family not in ("llm", "pg"):
        raise ValueError("dense PaliGemma collection does not match the requested model/family")
    if manifest["activation_dtype"] != "float32" or manifest["sae"]["layer"] != representation.layer:
        raise ValueError("dense collection dtype/layer does not match the requested representation")
    values = np.load(root / manifest["activation_file"], mmap_mode="r", allow_pickle=False)
    dim = representation.hidden_dim
    if values.ndim != 2 or values.shape[1] != dim:
        raise ValueError("activation dimension mismatch")
    window = int(manifest["patch"]["frames_per_window"])
    data = {name: _chunks() for name in ("train", "dev", "test")}
    for row in manifest["rollouts"]:
        condition = row["condition"]
        start = int(row["activation_start"])
        if condition == "clean":
            indices = np.arange(start, int(row["activation_end"]))
        elif condition == "attack":
            indices = np.concatenate([np.arange(start + int(t), start + int(t) + window)
                                      for t in row["attack_starts"]])
        else:
            raise ValueError(f"unknown collection condition: {condition}")
        _append(data[row["split"]], condition, np.asarray(values[indices], dtype=np.float32),
                int(row["task_id"]), int(row["pair_id"]))
    splits = {name: _dataset(chunks, dim) for name, chunks in data.items()}
    return ProbeSplits(representation, **splits, split_unit="task",
                       metadata={"source_collection": str(root), "layout": "dense_manifest",
                                 "collection_model": manifest["model"]})


def load_paired_collection(root: str | Path, *, layer: int) -> ProbeSplits:
    """Read SmolVLA clean rollouts and only patched frames from their attack pairs."""
    root = Path(root)
    cfg = json.loads((root / "collection_config.json").read_text())
    if layer not in cfg["layers"]:
        raise ValueError("layer is absent from the collection")
    family = "vit" if cfg.get("representation") == "vit" else "vlm"
    spec = ResidualSpec("smolvla", family, layer, cfg["capture_point"], cfg["pooling"], cfg["hidden_dim"],
                        tuple(cfg.get("image_keys", ())), cfg.get("checkpoint"))
    data = {name: _chunks() for name in ("train", "dev", "test")}
    for path in sorted((root / "parts").glob("pair*_clean.json")):
        row = json.loads(path.read_text())
        attack_path = path.with_name(path.name.replace("_clean.json", "_attack.json"))
        if not attack_path.is_file():
            raise ValueError(f"incomplete pair: {path}")
        other = json.loads(attack_path.read_text())
        if any(row[key] != other[key] for key in ("pair_id", "task_id", "split")):
            raise ValueError("clean and attack pair identities differ")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as artifact:
            clean = np.asarray(artifact[f"layer{layer}"], dtype=np.float32)
        with np.load(attack_path.with_suffix(".npz"), allow_pickle=False) as artifact:
            signal = np.asarray(artifact["patch_signal"], dtype=bool)
            if signal.sum() != 30:
                raise ValueError("legacy paired collection requires 30 patch samples per rollout")
            attack = np.asarray(artifact[f"layer{layer}"], dtype=np.float32)[signal]
        for condition, values in (("clean", clean), ("attack", attack)):
            _append(data[row["split"]], condition, values, int(row["task_id"]), int(row["pair_id"]))
    splits = {name: _dataset(chunks, spec.hidden_dim) for name, chunks in data.items()}
    return ProbeSplits(spec, **splits, split_unit="task",
                       metadata={"source_collection": str(root), "layout": "paired_rollouts"})


def load_episode_collection(root: str | Path, *, family: str, layer: int, fold: int = 0) -> ProbeSplits:
    """Read Pi0.5 multimodal or ViT-only paired episode arrays with task folds."""
    root = Path(root)
    cfg = json.loads((root / "collection_config.json").read_text())
    multimodal = cfg["representation"] == "pi05_llm_vit_residuals"
    if not multimodal and cfg["representation"] != "siglip_post_block_residual":
        raise ValueError("unsupported episode collection")
    if (not multimodal and family != "vit") or family not in ("llm", "vit"):
        raise ValueError("family is absent from the collection")
    prefix = f"{family}_" if multimodal else ""
    if layer not in cfg[prefix + "layers"]:
        raise ValueError("layer is absent from the collection")
    spec = ResidualSpec(cfg["model"], family, layer, cfg[prefix + "capture_point"], cfg[prefix + "pooling"],
                        cfg[prefix + "hidden_dim"], tuple(cfg["image_keys"]), cfg.get("checkpoint_dir"))
    groups = task_fold(fold)
    data = {name: _chunks() for name in groups}
    for path in sorted((root / "parts").glob("episode_*.json")):
        row = json.loads(path.read_text())
        task = int(row["task_id"])
        split = next((name for name, tasks in groups.items() if task in tasks), None)
        if split is None:
            continue
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as artifact:
            clean = np.asarray(artifact[f"clean_{prefix}layer{layer}"], dtype=np.float32)
            attack = np.asarray(artifact[f"patch_{prefix}layer{layer}"], dtype=np.float32)
        if clean.shape != attack.shape:
            raise ValueError(f"unpaired episode arrays: {path}")
        for condition, values in (("clean", clean), ("attack", attack)):
            _append(data[split], condition, values, task, int(row["episode_id"]))
    splits = {name: _dataset(chunks, spec.hidden_dim) for name, chunks in data.items()}
    if any(not len(split.clean) for split in splits.values()):
        raise ValueError("each task split must contain examples")
    return ProbeSplits(spec, **splits, split_unit="task",
                       metadata={"source_collection": str(root), "layout": "paired_episodes", "fold": fold})
