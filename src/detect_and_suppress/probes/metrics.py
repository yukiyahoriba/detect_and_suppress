"""Metrics and threshold calibration used by the research Probe scripts."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def calibrate_threshold(clean_scores: ArrayLike, *, target_fpr: float = 0.02) -> float:
    scores = np.asarray(clean_scores)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("clean_scores must be a nonempty finite vector")
    if not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must lie in [0, 1]")
    return float(np.quantile(scores, 1.0 - target_fpr, method="higher"))


def _scores_labels(scores: ArrayLike, labels: ArrayLike):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape or not np.isin(labels, (0, 1)).all():
        raise ValueError("expected matching score and binary-label vectors")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return scores, labels


def detection_rates(scores: ArrayLike, labels: ArrayLike, threshold: float) -> dict[str, float | int]:
    scores, labels = _scores_labels(scores, labels)
    y, p = labels.astype(bool), scores > threshold
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    tn, fn = int((~p & ~y).sum()), int((~p & y).sum())
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                tpr=tp / (tp + fn) if tp + fn else 0.,
                fpr=fp / (fp + tn) if fp + tn else 0.,
                precision=tp / (tp + fp) if tp + fp else 0.,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)


def ranking_metrics(scores: ArrayLike, labels: ArrayLike) -> tuple[float, float]:
    """Return research AUROC and AUPRC, retaining stable sample-level tie order.

    This deliberately preserves the legacy metric's treatment of tied scores
    rather than replacing it with a different ROC/AP implementation.
    """
    scores, labels = _scores_labels(scores, labels)
    y = labels.astype(np.int8)
    order = np.argsort(-scores, kind="stable")
    ys = y[order]
    pos, neg = int(y.sum()), len(y) - int(y.sum())
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auroc = float(integrate(tp / max(pos, 1), fp / max(neg, 1)))
    precision = tp / np.arange(1, len(y) + 1)
    auprc = float((precision * ys).sum() / max(pos, 1))
    return auroc, auprc


def window_detection_rate(scores: ArrayLike, threshold: float, *, window: int = 10) -> float:
    """Measure detection in consecutive windows, preserving the input order."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not isinstance(window, int) or window < 1 or not len(scores) or len(scores) % window:
        raise ValueError("patch scores must form nonempty complete windows")
    return float(np.mean(np.any(scores.reshape(-1, window) > threshold, axis=1)))


def task_macro_auroc(clean_scores: ArrayLike, attack_scores: ArrayLike,
                     clean_tasks: ArrayLike, attack_tasks: ArrayLike) -> float:
    clean, attack = np.asarray(clean_scores), np.asarray(attack_scores)
    ct, at = np.asarray(clean_tasks), np.asarray(attack_tasks)
    if ct.shape != clean.shape or at.shape != attack.shape or set(ct.tolist()) != set(at.tolist()) or not len(ct):
        raise ValueError("both conditions must provide scores for the same tasks")
    values = []
    for task in sorted(set(ct.tolist())):
        c, a = clean[ct == task], attack[at == task]
        values.append(ranking_metrics(np.r_[c, a], np.r_[np.zeros(len(c)), np.ones(len(a))])[0])
    return float(np.mean(values))
