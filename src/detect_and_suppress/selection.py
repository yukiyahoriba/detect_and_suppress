"""Episode correlations, feature rankings, and selection artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
import tempfile
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


RankingCriterion = Literal["mean", "max", "count"]


@dataclass(frozen=True)
class EpisodeLatents:
    """Latents and rollout signals indexed by environment step."""

    expert: str
    layer: int
    episode_id: int
    task_id: int
    latents: NDArray
    patch_signal: NDArray
    policy_forward_signal: NDArray


@dataclass(frozen=True)
class EpisodeCorrelation:
    expert: str
    layer: int
    episode_id: int
    task_id: int
    feature_id: int
    pearson_r: float | None
    forward_samples: int
    patch_on_samples: int
    patch_off_samples: int
    fire_count: int
    mean_activation_on: float | None
    mean_activation_off: float | None


@dataclass(frozen=True)
class FeatureSummary:
    expert: str
    layer: int
    feature_id: int
    num_episodes_fired: int
    num_episodes_valid_r: int
    num_episodes_r_gt_threshold: int
    threshold: float
    mean_pearson_r: float | None
    median_pearson_r: float | None
    std_pearson_r: float | None
    min_pearson_r: float | None
    max_pearson_r: float | None
    total_fire_count: int
    episode_ids: tuple[int, ...]


@dataclass(frozen=True)
class FeatureRanking:
    criterion: RankingCriterion
    correlation_threshold: float
    features: tuple[FeatureSummary, ...]


@dataclass(frozen=True)
class FeatureSelection:
    criterion: RankingCriterion
    correlation_threshold: float
    features: tuple[FeatureSummary, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


def pearson_correlation(values: ArrayLike, patch_presence: ArrayLike) -> float | None:
    """Return Pearson r, or None when there is no defined finite correlation."""
    values = np.asarray(values, dtype=np.float64)
    patch = np.asarray(patch_presence, dtype=np.float64)
    if values.ndim != 1 or values.shape != patch.shape:
        raise ValueError("values and patch_presence must be matching one-dimensional arrays")
    if values.size < 2:
        return None
    centered_values = values - values.mean()
    centered_patch = patch - patch.mean()
    denominator = float(np.linalg.norm(centered_values) * np.linalg.norm(centered_patch))
    if denominator <= 0:
        return None
    result = float(np.dot(centered_values, centered_patch) / denominator)
    return result if math.isfinite(result) else None


def compute_correlations(episodes: Iterable[EpisodeLatents]) -> list[EpisodeCorrelation]:
    """Compute fired-feature correlations at policy forwards, per episode.

    As in the legacy rollout writers, signals are truncated to their common
    length, and forward/patch presence uses a strict 0.5 cutoff. Features
    that never fire at a forward have no row. Undefined r is stored as None.
    """
    rows = []
    for episode in episodes:
        latents = np.asarray(episode.latents)
        patch_signal = np.asarray(episode.patch_signal)
        forward_signal = np.asarray(episode.policy_forward_signal)
        if latents.ndim != 2 or patch_signal.ndim != 1 or forward_signal.ndim != 1:
            raise ValueError("expected [steps, features] latents and one-dimensional signals")
        n = min(latents.shape[0], patch_signal.size, forward_signal.size)
        indices = np.flatnonzero(forward_signal[:n] > 0.5)
        if indices.size == 0:
            continue
        patch = patch_signal[indices].astype(np.float64)
        values = latents[indices].astype(np.float64)
        on = patch > 0.5
        off = ~on
        for feature_id in np.flatnonzero(np.any(values > 0, axis=0)):
            feature = values[:, feature_id]
            rows.append(EpisodeCorrelation(
                expert=episode.expert, layer=episode.layer, episode_id=episode.episode_id,
                task_id=episode.task_id, feature_id=int(feature_id),
                pearson_r=pearson_correlation(feature, patch), forward_samples=int(indices.size),
                patch_on_samples=int(on.sum()), patch_off_samples=int(off.sum()),
                fire_count=int(np.count_nonzero(feature > 0)),
                mean_activation_on=float(feature[on].mean()) if on.any() else None,
                mean_activation_off=float(feature[off].mean()) if off.any() else None,
            ))
    rows.sort(key=lambda row: (row.episode_id,
                              -(row.pearson_r if row.pearson_r is not None else -math.inf), row.feature_id))
    return rows


def mean_feature_scores(
    correlations: Iterable[EpisodeCorrelation], *, expert: str, layer: int, n_features: int,
) -> NDArray[np.float64]:
    """Return unweighted valid-episode means; absent/undefined features score -inf."""
    if n_features < 0:
        raise ValueError("n_features must be non-negative")
    values: list[list[float]] = [[] for _ in range(n_features)]
    for row in correlations:
        if row.expert != expert or row.layer != layer:
            continue
        if not 0 <= row.feature_id < n_features:
            raise ValueError("feature_id is outside the dictionary")
        if row.pearson_r is not None:
            values[row.feature_id].append(row.pearson_r)
    result = np.full(n_features, -np.inf, dtype=np.float64)
    for feature_id, correlations_for_feature in enumerate(values):
        if correlations_for_feature:
            result[feature_id] = float(np.mean(correlations_for_feature))
    return result


def aggregate_correlations(
    correlations: Iterable[EpisodeCorrelation], *, threshold: float = 0.6,
) -> list[FeatureSummary]:
    """Summarize episodes where a feature fired, excluding undefined r from statistics."""
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    grouped: dict[tuple[str, int, int], list[EpisodeCorrelation]] = {}
    for row in sorted(correlations, key=lambda r: (r.layer, r.episode_id, r.feature_id)):
        grouped.setdefault((row.expert, row.layer, row.feature_id), []).append(row)
    summaries = []
    for (expert, layer, feature_id), rows in grouped.items():
        values = np.asarray([row.pearson_r for row in rows if row.pearson_r is not None], dtype=np.float64)
        episode_ids = tuple(sorted({row.episode_id for row in rows}))
        summaries.append(FeatureSummary(
            expert=expert, layer=layer, feature_id=feature_id,
            num_episodes_fired=len(episode_ids), num_episodes_valid_r=len(values),
            num_episodes_r_gt_threshold=int(np.count_nonzero(values > threshold)), threshold=threshold,
            mean_pearson_r=float(values.mean()) if values.size else None,
            median_pearson_r=float(np.median(values)) if values.size else None,
            std_pearson_r=float(values.std()) if values.size else None,
            min_pearson_r=float(values.min()) if values.size else None,
            max_pearson_r=float(values.max()) if values.size else None,
            total_fire_count=sum(row.fire_count for row in rows), episode_ids=episode_ids,
        ))
    return summaries


def rank_features(
    summaries: Iterable[FeatureSummary], *, criterion: RankingCriterion = "mean",
    correlation_threshold: float | None = None,
) -> FeatureRanking:
    """Rank by mean r, maximum r, or episode count above the aggregation threshold.

    Count ties use mean r; remaining ties use layer and feature ID, as in
    the research summary script. Undefined scores sort after finite scores.
    """
    if criterion not in ("mean", "max", "count"):
        raise ValueError("criterion must be mean, max, or count")
    rows = list(summaries)
    if correlation_threshold is None:
        correlation_threshold = rows[0].threshold if rows else 0.6
    if not math.isfinite(correlation_threshold) or any(row.threshold != correlation_threshold for row in rows):
        raise ValueError("correlation_threshold must match the aggregation threshold")

    def key(row: FeatureSummary) -> tuple:
        score = row.max_pearson_r if criterion == "max" else row.mean_pearson_r
        tail = (-(score if score is not None else -math.inf), row.layer, row.feature_id)
        return (-row.num_episodes_r_gt_threshold, *tail) if criterion == "count" else tail

    return FeatureRanking(criterion, correlation_threshold, tuple(sorted(rows, key=key)))


def select_features(
    ranking: FeatureRanking, *, top_n: int = 20, metadata: Mapping[str, object] | None = None,
) -> FeatureSelection:
    """Select the leading ranked features and retain caller-supplied provenance."""
    if not isinstance(top_n, int) or top_n < 0:
        raise ValueError("top_n must be a non-negative integer")
    return FeatureSelection(ranking.criterion, ranking.correlation_threshold, ranking.features[:top_n],
                            dict(metadata) if metadata is not None else {})


def select_positive_features(scores: ArrayLike, *, top_n: int = 20) -> tuple[int, ...]:
    """Select positive finite scores using the legacy viewer's NumPy ordering.

    This is separate from the summary rankings, which retain nonpositive
    and undefined scores. In particular, preserve the viewer's tie ordering.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not isinstance(top_n, int) or top_n < 0:
        raise ValueError("expected one-dimensional scores and a non-negative integer top_n")
    positive = np.flatnonzero(np.isfinite(scores) & (scores > 0))
    ranked = positive[np.argsort(scores[positive])[::-1]]
    return tuple(int(value) for value in ranked[:top_n])


def read_correlations(path: str | Path) -> list[EpisodeCorrelation]:
    """Read a legacy all-feature episode Pearson CSV, mapping empty fields to None."""
    integer_fields = {"layer", "episode_id", "task_id", "feature_id", "forward_samples",
                      "patch_on_samples", "patch_off_samples", "fire_count"}
    optional_fields = {"pearson_r", "mean_activation_on", "mean_activation_off"}
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            converted = {name: int(record[name]) for name in integer_fields}
            converted.update({name: float(record[name]) if record[name] != "" else None for name in optional_fields})
            rows.append(EpisodeCorrelation(expert=record["expert"], **converted))
    return rows


def write_correlations(path: str | Path, correlations: Iterable[EpisodeCorrelation]) -> None:
    """Write the legacy CSV column layout, with empty fields for undefined values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[item.name for item in fields(EpisodeCorrelation)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in correlations)


def save_selection(path: str | Path, selection: FeatureSelection) -> None:
    """Save selected feature identities, scores, selection rule, and metadata as JSON."""
    payload = {"schema_version": 1, **asdict(selection)}
    contents = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_selection(path: str | Path) -> FeatureSelection:
    """Read a versioned selection artifact without reranking its features."""
    def reject_constant(value: str):
        raise ValueError(f"Nonfinite JSON value: {value}")

    payload = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)
    if payload.pop("schema_version") != 1:
        raise ValueError("unsupported selection schema_version")
    rows = tuple(FeatureSummary(**{**row, "episode_ids": tuple(row["episode_ids"])})
                 for row in payload.pop("features"))
    ranking = rank_features(rows, criterion=payload["criterion"],
                            correlation_threshold=payload["correlation_threshold"])
    if ranking.features != rows:
        raise ValueError("selection features are not in the declared ranking order")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("metadata must be a JSON object")
    return FeatureSelection(features=rows, **payload)
