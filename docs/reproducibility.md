# Reproducibility and limitations

The package preserves the recorded algorithms, model boundaries and LIBERO-10
evaluation semantics. The two portable templates identify current artifacts
by SHA256 and record the chosen feature, layers, strength, gate, patch, seeds,
cameras, preprocessing and rollout settings. No experiment setting or numerical
tolerance was changed to make results match historical runs.

## Representative LIBERO-10 results

Each condition uses ten tasks and the first fifty initial states per task,
seed 1, three ten-frame patch bursts, ten warmup/denoising steps and a maximum
of 520 control steps.

| Condition | Historical result | Package result | Difference |
|---|---:|---:|---:|
| Pi0.5 PG0 Probe → PG5/F902, strength 0.3, replan 5 | 237/500 (47.4%) | 224/500 (44.8%) | −2.6 pp |
| SmolVLA VLM0 Probe → VLM9/F123, strength 0.2, replan 1 | 98/500 (19.6%) | 106/500 (21.2%) | +1.6 pp |

These are completed evaluations, not claims of exact historical reproduction.
Success labels differed in 145 Pi0.5 episodes and 22 SmolVLA episodes. Pi0.5
changes occurred across all tasks; SmolVLA changes occurred in tasks 2, 3, 8
and 9. Equal aggregate Success Rate can hide different episode outcomes.

Fixed-input comparisons established numerical agreement for the main detector,
gate, intervention and action-generation paths in the evaluated environments.
Independent executions can still diverge: XLA autotuning can choose different
kernels, and native rendering can produce pixel differences with identical
measured simulator states. Small same-episode repeats also showed changing
scores and, for SmolVLA, changing success. These observations do not quantify
full-benchmark SR variance or explain every historical difference.

Historical configurations did not archive artifact content hashes, complete
runtime fingerprints or compiled/autotune caches. Current artifacts are pinned,
but historical byte identity cannot be established retrospectively. The paper
version/table-to-condition mapping has not been confirmed; `paper.table` is
intentionally null in the templates.

## Paper/code description

The implementation scales subtraction using the residual norm. The paper
description instead scales it using the SAE activation. This remains an
unresolved method-description discrepancy. The implementation is preserved;
no decision about correcting the paper is made here.

## Supported scope

- The 500-rollout evidence covers the two listed three-burst Probe conditions.
  Other modes, layers, checkpoints and schedules have narrower coverage.
- Evaluation supports LIBERO-10, fifty initial states per task and one selected
  feature. Pi0.5 uses the agent-only observation adapter. SmolVLA requires
  replan interval 1 and prefix timing; RTC and unsupported combinations are
  rejected.
- Optimizer profiles keep their original float64 training/float32 runtime,
  candidate-selection and calibration rules. Results can depend on numerical
  library versions. Original sklearn artifact metadata do not fully establish
  its historical optimizer environment.
- New environment installation was checked at the package/CLI boundary with
  existing numerical dependencies. Building every CUDA/model/simulator
  dependency from zero and fresh upstream artifact downloads remain unverified.

## Artifact availability

Author checkpoint/patch hosting and redistribution permissions, the package
license and relevant upstream model/asset terms require confirmation. No public
download is invented for an unavailable artifact. Full selection/Probe
retraining additionally requires activation collections whose distribution
is pending. Evaluation with saved artifacts does not require those collections.

See the [artifact inventory](../artifacts/README.md),
[environment setup](../environments/README.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md).
