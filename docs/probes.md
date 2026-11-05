# Residual Probe APIs

`detect_and_suppress.probes` is independent of SAE feature selection and
does not import model, hook, rollout, Torch or sklearn code at import time.
The selected optimizer is imported only when needed for training. Inference,
calibration, metrics and artifact I/O use NumPy.

## Training profiles

| Profile | Train statistics / optimizer | C selection | Runtime threshold |
| --- | --- | --- | --- |
| `sklearn_dense` | sklearn StandardScaler / balanced LogisticRegression, float64, max_iter 2000 | Fixed C, default 0.01 | Recalibrate on float32 clean dev scores |
| `smolvla` | NumPy mean/std / balanced Torch float64 LBFGS, max_iter 300 | Maximize dev 10-step patch-window detection; ties choose smaller C | Recalibrate on float32 clean dev scores |
| `pi05` | NumPy mean/std / balanced Torch float64 LBFGS, max_iter 2000 | Maximize dev task-macro AUROC; ties choose smaller C | Cast the float64 clean-dev threshold |

The SmolVLA C grid is `(0.01, 0.1, 1, 10, 100)`. The Pi0.5 grid adds 0.001.
`ProbeTrainingConfig` supports explicit C values and iteration limits to
reproduce a recorded run. Its optional `expected_reference_threshold` uses
the original dense script's absolute tolerance of 1e-8, with no relative
tolerance. The dense profile requires one C rather than introducing a new
candidate-selection rule.

Torch preserves zero initialization, balanced sample weights, the original
regularized loss, and LBFGS settings: lr 1, max_eval twice max_iter,
tolerance_grad 1e-8, tolerance_change 1e-10, history_size 20 and strong_wolfe
line search. These settings and dependency versions are recorded. The
sklearn path uses the original estimator and its defaults, not the Torch
optimizer. StandardScaler's near-constant feature handling is retained;
the NumPy path only replaces exactly zero standard deviations with one.

`calibrate_threshold` uses the `higher` clean-score quantile. Detection is
strictly `score > threshold`, so equality does not trigger detection. The
calibrated false-positive rate need not equal the requested rate exactly.

## Data and representation

`ResidualSpec` records model, family, layer, capture point, pooling, hidden
dimension, camera order when known, and checkpoint identity when available.
`ProbeDataset` contains separate clean/attack arrays and optional per-row
task and episode IDs. All legacy training collections contain float32
residuals; the dataset API requires that dtype explicitly, then training
converts the concatenated clean-first/attack-second matrix to float64.
This avoids silently changing runtime calibration when callers supply a
different input dtype. `Standardizer.transform` and Probe inference retain
NumPy's normal dtype promotion rather than forcing all scoring to float32.

`ProbeSplits` holds train/dev/test and the representation. Task-disjoint or
episode-disjoint splits require the relevant IDs and reject overlaps. When
episode IDs are supplied for all splits, their overlap is checked even for
prespecified splits. Episode IDs must identify an episode/pair globally,
not a state index reused across tasks. Clean and attack may share pair IDs
within a split. Prespecified arrays without IDs are supported, but do not
provide an identity-level leakage guarantee. Test is optional and is never
used to fit statistics, choose C, or calibrate a threshold.

Three readers preserve file ordering and the original sample selection:

- `load_dense_collection(root, representation=spec)` reads manifest offsets;
  clean uses all recorded rows and attack uses the recorded burst starts.
- `load_paired_collection(root, layer=...)` reads paired SmolVLA rollout
  files; clean uses all rows and attack uses the 30 patch-marked rows per
  rollout, in their saved order. No random split is introduced.
- `load_episode_collection(root, family=..., layer=..., fold=...)` reads Pi0.5
  multimodal or ViT-only episode files. `task_fold` implements the existing
  five task groups and train/dev/test rotation.

Manual datasets use the same API:

```python
from detect_and_suppress.probes import (
    ProbeDataset, ProbeSplits, ProbeTrainingConfig, ResidualSpec, train_probe,
)

spec = ResidualSpec(
    model="smolvla", family="vlm", layer=0,
    capture_point="VLM block input_layernorm input (pre-RMSNorm residual)",
    pooling="FP32 masked mean over all valid prefix tokens", hidden_dim=960,
)
splits = ProbeSplits(
    representation=spec,
    train=ProbeDataset(train_clean, train_attack),
    dev=ProbeDataset(dev_clean, dev_attack),
    test=ProbeDataset(test_clean, test_attack),
    metadata={"source_collection": "my-saved-residuals"},
)
result = train_probe(splits, config=ProbeTrainingConfig("smolvla"))
```

The SmolVLA profile requires consecutive complete 10-sample patch windows
in dev/test attack arrays; its collection reader preserves that ordering.
Pi0.5 macro selection requires task IDs for both dev conditions. Clean-only
dev data are sufficient for the fixed-C sklearn profile.

## Scoring, evaluation, and artifacts

`LogisticProbe.standardize`, `.decision_function`, and `.predict` share the
same input shape `[samples, hidden_dim]`. Pass `representation=actual_spec`
when scoring a new source to validate all representation fields; omitting
it assumes the caller already bound the data to this Probe. Dimension is
always checked. The API does not acquire or pool model activations.

`evaluate_probe` reports confusion counts, FPR, TPR, precision, F1, AUROC,
and AUPRC. Task macro and per-task attack TPR are available with task IDs;
window detection is opt-in. AUROC/AUPRC intentionally retain the original
stable per-sample tie ordering, rather than substituting sklearn metrics
with different tie handling.

`ProbeTrainingResult` includes the float32 `probe`, float64 `reference_probe`,
all candidate summaries, and separate reference/runtime dev/test metrics.
For Pi0.5, the legacy reported test metrics used the float64 reference
model even though the saved runtime arrays were float32; both results are
retained explicitly. Artifact metadata include both evaluations, split IDs,
training settings and dependency versions.

`save_probe` writes the legacy keys `mean`, `scale`, `coef`, `intercept`,
`threshold`, along with C/layer/etc. and embedded versioned representation
and metadata. A JSON sidecar is provided for readability. NPZ is
self-contained; each file is replaced atomically. The NPZ/JSON pair is not
a multi-file transaction, and the embedded NPZ metadata are authoritative.

`load_probe` preserves stored arrays and thresholds and never recalibrates.
For an old NPZ, supply `representation=spec` explicitly: legacy metadata
cannot fully identify model/capture/pooling by themselves. Available
layer, dimension, family, pooling, capture point and camera metadata are
checked, and extra NPZ arrays and sidecar metadata are retained. Old runtime
loaders can read the required keys of the new artifacts.

## Reproducibility

Training profiles preserve their distinct precision, optimization and calibration
procedures. Historical environments are not fully archived. See
[reproducibility and limitations](reproducibility.md).
