"""Independent residual-activation Probe training and inference."""

from .artifacts import load_probe, save_probe
from .data import ProbeDataset, ProbeSplits, load_dense_collection, load_episode_collection, load_paired_collection, task_fold
from .metrics import calibrate_threshold, detection_rates, ranking_metrics, task_macro_auroc, window_detection_rate
from .model import LogisticProbe, ResidualSpec, Standardizer
from .training import ProbeTrainingConfig, ProbeTrainingResult, evaluate_probe, fit_standardizer, train_probe

__all__ = [
    "LogisticProbe", "ProbeDataset", "ProbeSplits", "ProbeTrainingConfig", "ProbeTrainingResult", "ResidualSpec",
    "Standardizer", "calibrate_threshold", "detection_rates", "evaluate_probe", "fit_standardizer",
    "load_dense_collection", "load_episode_collection", "load_paired_collection", "load_probe", "ranking_metrics",
    "save_probe", "task_fold", "task_macro_auroc", "train_probe", "window_detection_rate",
]
