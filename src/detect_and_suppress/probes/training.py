"""Legacy logistic training profiles with explicit reference and runtime results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .data import ProbeDataset, ProbeSplits
from .metrics import calibrate_threshold, detection_rates, ranking_metrics, task_macro_auroc, window_detection_rate
from .model import LogisticProbe, Standardizer


@dataclass(frozen=True)
class ProbeTrainingConfig:
    profile: Literal["sklearn_dense", "smolvla", "pi05"]
    c_values: tuple[float, ...] | None = None
    target_fpr: float = 0.02
    max_iter: int | None = None
    expected_reference_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.profile not in ("sklearn_dense", "smolvla", "pi05"):
            raise ValueError("unknown Probe training profile")
        if not 0 <= self.target_fpr <= 1:
            raise ValueError("target_fpr must lie in [0, 1]")
        if not self.c_grid or any(not math.isfinite(c) or c <= 0 for c in self.c_grid):
            raise ValueError("C values must be positive and finite")
        if self.profile == "sklearn_dense" and len(self.c_grid) != 1:
            raise ValueError("sklearn_dense uses one fixed C")
        if self.iterations < 1:
            raise ValueError("max_iter must be positive")

    @property
    def c_grid(self) -> tuple[float, ...]:
        if self.c_values is not None:
            return self.c_values
        if self.profile == "sklearn_dense":
            return (.01,)
        if self.profile == "smolvla":
            return (.01, .1, 1., 10., 100.)
        return (.001, .01, .1, 1., 10., 100.)

    @property
    def iterations(self) -> int:
        return self.max_iter if self.max_iter is not None else (300 if self.profile == "smolvla" else 2000)


@dataclass(frozen=True)
class CandidateResult:
    c: float
    selection_score: float | None
    reference_threshold: float
    train_loss: float | None


@dataclass(frozen=True)
class ProbeTrainingResult:
    probe: LogisticProbe
    reference_probe: LogisticProbe
    selected_c: float
    candidates: tuple[CandidateResult, ...]
    dev_reference: dict
    dev_runtime: dict
    test_reference: dict | None
    test_runtime: dict | None


def fit_standardizer(values: NDArray, *, method: Literal["numpy", "sklearn"]) -> Standardizer:
    """Fit train-only statistics using the specified legacy implementation."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("expected a nonempty finite training matrix")
    if method == "sklearn":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(values)
        return Standardizer(scaler.mean_, scaler.scale_)
    if method != "numpy":
        raise ValueError("method must be numpy or sklearn")
    mean, scale = values.mean(0), values.std(0)
    scale[scale == 0] = 1.
    return Standardizer(mean, scale)


def _fit_torch(values: NDArray, labels: NDArray, c: float, max_iter: int):
    import torch

    xt = torch.from_numpy(np.asarray(values, np.float64))
    yt = torch.from_numpy(np.asarray(labels, np.float64))
    n, d = xt.shape
    w = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    npos, nneg = float(yt.sum()), n - float(yt.sum())
    sample_w = torch.where(yt > 0, torch.tensor(n / (2 * npos), dtype=torch.float64),
                           torch.tensor(n / (2 * nneg), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([w, b], lr=1., max_iter=max_iter, max_eval=max_iter * 2,
                                 tolerance_grad=1e-8, tolerance_change=1e-10,
                                 history_size=20, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = xt @ w + b
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, reduction="none")
        loss = (sample_w * bce).sum() / sample_w.sum() + .5 * w.square().sum() / (c * n)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        logits = xt @ w + b
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, reduction="none")
        loss = (sample_w * bce).sum() / sample_w.sum() + .5 * w.square().sum() / (c * n)
        return w.numpy().copy(), float(b), float(loss)


def evaluate_probe(probe: LogisticProbe, dataset: ProbeDataset, *, window: int | None = None) -> dict:
    clean_scores = probe.decision_function(dataset.clean)
    attack_scores = probe.decision_function(dataset.attack)
    scores = np.r_[clean_scores, attack_scores]
    labels = np.r_[np.zeros(len(clean_scores)), np.ones(len(attack_scores))]
    result = detection_rates(scores, labels, float(probe.threshold))
    result["auroc"], result["auprc"] = ranking_metrics(scores, labels)
    if window is not None:
        result["window_detection_rate"] = window_detection_rate(attack_scores, float(probe.threshold), window=window)
    if dataset.clean_task_ids is not None and dataset.attack_task_ids is not None and len(attack_scores):
        if set(dataset.clean_task_ids.tolist()) == set(dataset.attack_task_ids.tolist()):
            result["task_macro_auroc"] = task_macro_auroc(clean_scores, attack_scores,
                                                        dataset.clean_task_ids, dataset.attack_task_ids)
        result["attack_tpr_by_task"] = {
            str(task): float(np.mean(attack_scores[dataset.attack_task_ids == task] > probe.threshold))
            for task in sorted(set(dataset.attack_task_ids.tolist()))
        }
    return result


def train_probe(splits: ProbeSplits, *, config: ProbeTrainingConfig) -> ProbeTrainingResult:
    """Fit a research Probe, keeping reference precision separate from its artifact.

    The profile fixes standardization, optimizer, candidate selection, and
    runtime calibration. Test data never participate in fitting or selection.
    """
    if not len(splits.train.clean) or not len(splits.train.attack) or not len(splits.dev.clean):
        raise ValueError("training needs both classes and calibration needs clean dev samples")
    if config.profile != "sklearn_dense" and not len(splits.dev.attack):
        raise ValueError("candidate selection requires attack dev samples")
    values, labels = splits.train.training_arrays()
    stats = fit_standardizer(values, method="sklearn" if config.profile == "sklearn_dense" else "numpy")
    standardized = stats.transform(values)
    candidates, best = [], None
    optimizer_version = None
    for c in config.c_grid:
        if config.profile == "sklearn_dense":
            import sklearn
            from sklearn.linear_model import LogisticRegression

            optimizer_version = sklearn.__version__
            model = LogisticRegression(C=c, class_weight="balanced", max_iter=config.iterations).fit(standardized, labels)
            coef, intercept, loss = model.coef_[0], float(model.intercept_[0]), None
        else:
            import torch

            optimizer_version = torch.__version__
            coef, intercept, loss = _fit_torch(standardized, labels, c, config.iterations)
        clean_scores = stats.transform(splits.dev.clean.astype(np.float64)) @ coef + intercept
        threshold = calibrate_threshold(clean_scores, target_fpr=config.target_fpr)
        score = None
        if config.profile != "sklearn_dense":
            attack_scores = stats.transform(splits.dev.attack.astype(np.float64)) @ coef + intercept
            if config.profile == "smolvla":
                score = window_detection_rate(attack_scores, threshold)
            else:
                if splits.dev.clean_task_ids is None or splits.dev.attack_task_ids is None:
                    raise ValueError("pi05 selection requires task IDs in dev")
                score = task_macro_auroc(clean_scores, attack_scores,
                                         splits.dev.clean_task_ids, splits.dev.attack_task_ids)
        candidates.append(CandidateResult(c, score, threshold, loss))
        key = (score if score is not None else 0., -c)
        if best is None or key > best[0]:
            best = key, c, coef, intercept, threshold, loss
    _, c, coef, intercept, threshold, loss = best
    if config.expected_reference_threshold is not None and not np.isclose(
        threshold, config.expected_reference_threshold, rtol=0., atol=1e-8,
    ):
        raise RuntimeError(f"reference threshold mismatch: {threshold} != {config.expected_reference_threshold}")
    metadata = {
        **dict(splits.metadata), "profile": config.profile, "C": c, "C_grid": list(config.c_grid),
        "max_iter": config.iterations, "target_fpr": config.target_fpr, "class_weight": "balanced",
        "standardization": "sklearn StandardScaler" if config.profile == "sklearn_dense" else "numpy mean/std ddof=0",
        "optimizer": "sklearn LogisticRegression" if config.profile == "sklearn_dense" else "torch.float64 LBFGS",
        "optimizer_version": str(optimizer_version), "numpy_version": np.__version__,
        "selection": {"sklearn_dense": "fixed C", "smolvla": "dev 10-step window detection; ties smaller C",
                      "pi05": "dev task-macro AUROC; ties smaller C"}[config.profile],
        "runtime_calibration": "cast reference threshold" if config.profile == "pi05" else "recalibrate float32 clean dev",
        "reference_float64_threshold": threshold, "train_loss": loss,
        "split_unit": splits.split_unit, "candidates": [asdict(item) for item in candidates],
        "threshold_quantile_method": "higher", "decision_rule": "score > threshold",
        "training_config": {**asdict(config), "c_values": list(config.c_grid), "max_iter": config.iterations},
        "optimizer_settings": ({"solver": "lbfgs", "tol": 1e-4, "max_iter": config.iterations}
                               if config.profile == "sklearn_dense" else {
                                   "lr": 1., "max_iter": config.iterations, "max_eval": config.iterations * 2,
                                   "tolerance_grad": 1e-8, "tolerance_change": 1e-10,
                                   "history_size": 20, "line_search_fn": "strong_wolfe",
                               }),
    }
    for name in ("train", "dev", "test"):
        split = getattr(splits, name)
        if split is not None:
            metadata[f"{name}_clean_steps"], metadata[f"{name}_attack_steps"] = len(split.clean), len(split.attack)
            metadata[f"{name}_tasks"] = (sorted(set(split.clean_task_ids.tolist()) | set(split.attack_task_ids.tolist()))
                                         if split.clean_task_ids is not None and split.attack_task_ids is not None else None)
            metadata[f"{name}_episodes"] = (sorted(set(split.clean_episode_ids.tolist()) | set(split.attack_episode_ids.tolist()))
                                            if split.clean_episode_ids is not None and split.attack_episode_ids is not None else None)
    reference = LogisticProbe(splits.representation, stats.mean, stats.scale, coef, np.float64(intercept),
                              np.float64(threshold), metadata)
    extras = {"C": np.asarray(c, dtype=np.float32), "target_fpr": np.asarray(config.target_fpr, dtype=np.float32),
              "layer": np.asarray(splits.representation.layer, dtype=np.int32),
              "reference_float64_threshold": np.asarray(threshold, dtype=np.float64)}
    if splits.representation.image_keys:
        extras["image_keys"] = np.asarray(splits.representation.image_keys)
    runtime = LogisticProbe(splits.representation, stats.mean.astype("f4"), stats.scale.astype("f4"),
                            coef.astype("f4"), np.float32(intercept), np.float32(threshold), metadata, extras)
    if config.profile != "pi05":
        runtime = replace(runtime, threshold=np.float32(calibrate_threshold(
            runtime.decision_function(splits.dev.clean), target_fpr=config.target_fpr)))
    metadata["runtime_float32_threshold"] = float(runtime.threshold)
    metadata["float32_vs_float64_clean_dev_prediction_disagreements"] = int(np.count_nonzero(
        runtime.predict(splits.dev.clean) != reference.predict(splits.dev.clean.astype(np.float64))))
    window = 10 if config.profile == "smolvla" else None
    evaluation = dict(
        dev_reference=evaluate_probe(reference, splits.dev, window=window),
        dev_runtime=evaluate_probe(runtime, splits.dev, window=window),
        test_reference=evaluate_probe(reference, splits.test, window=window) if splits.test is not None else None,
        test_runtime=evaluate_probe(runtime, splits.test, window=window) if splits.test is not None else None,
    )
    metadata["evaluation"] = evaluation
    return ProbeTrainingResult(runtime, reference, c, tuple(candidates), **evaluation)
