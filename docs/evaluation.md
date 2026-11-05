# LIBERO-10 evaluation

The evaluation package runs the research conditions against installed LIBERO,
OpenPI and LeRobot. It uses the existing defense adapters. It does not contain
a simulator, VLA inference engine, training code or patch optimizer.

## Public API

- `Artifact.identify(path)` hashes a file or a directory's sorted relative
  filenames and contents. `verify()` rejects changed contents.
- `EvaluationConfig` describes the condition, defense configuration, selected
  feature, artifact identities, policy/schedule seeds and rollout settings.
  `to_dict()` / `from_dict()` preserve its fingerprint across JSON round trips.
  Saved `protocol` expands the fixed schema-1 geometry, clock, noise and
  success rules. Geometry is the legacy paper profile, not another tunable
  intervention/attack option. Modified protocol metadata are rejected.
- `Shard(index, count)` assigns global rollout IDs by modulo. Task and initial
  state are always `rollout_id // 50` and `rollout_id % 50`.
- `run_evaluation(config, adapter, output, shard=..., max_new_episodes=...)`
  resumes validated episodes and runs missing ones. `on_episode(row, summary)`
  is an optional progress callback. `run_episode()` supports a `trace` callback
  for inspecting observations, full action chunks and executed actions.
- `ResultStore` commits episode results and events. `summarize()` computes
  Success Rate, per-task rates, forward counts, TPR and FPR.
- `merge_shards(config, sources, output)` validates identities and disjoint
  results, and requires all 500 IDs by default. `require_complete=False`
  explicitly permits a partial aggregate.
- `repartition_shards(config, sources, output, count)` moves a snapshot of
  completed episodes into a new modulo partition. Stop input workers first;
  locks reject active writers. The output must be new. Existing committed
  data are retained, and new workers resume the remaining IDs.
Model-specific public adapters are `Pi05LiberoAdapter` and
`SmolVLALiberoAdapter` in `detect_and_suppress.adapters`. Both are context
managers and load artifacts through `load_defense`.

For a newly trained SmolVLA residual Probe with complete representation metadata:

```python
from dataclasses import replace
import json
from pathlib import Path

from detect_and_suppress.probes import load_probe
from detect_and_suppress.defense import DefenseConfig
from detect_and_suppress.evaluation import (
    Artifact, EvaluationConfig, Shard, run_evaluation, merge_shards,
)
from detect_and_suppress.adapters.libero_support import libero_runtime_identity
from detect_and_suppress.adapters.smolvla_libero import SmolVLALiberoAdapter

probe = load_probe("probe.npz")
paths = dict(checkpoint="/path/to/checkpoint", probe="probe.npz",
             sae="sae.npz", selection="selection.json", patch="patch.png")
config = EvaluationConfig(
    model="smolvla", condition="paper/patch3/probe-layer0/feature123",
    defense=DefenseConfig(probe.representation,
                          replace(probe.representation, layer=9), lambda_=0.2),
    artifacts={name: Artifact.identify(path) for name, path in paths.items()},
    feature_id=123, replan_steps=1,
    provenance=libero_runtime_identity("smolvla"),
)
Path("experiment.json").write_text(json.dumps(config.to_dict(), indent=2))

with SmolVLALiberoAdapter(config) as adapter:
    summary = run_evaluation(config, adapter, "results/shard0",
                             shard=Shard(0, 4), max_new_episodes=25)
```

Restart with the same configuration/output to continue. Other workers load
the **same** `experiment.json` with `EvaluationConfig.from_dict()` and use
their own shard directories. Worker-local selection paths would produce
different experiment identities; share the same artifact references.
Use a separate process and adapter per worker; SmolVLA retains the legacy
process-global Torch RNG. Threaded concurrent rollouts are not supported.
After all four finish:

```python
summary = merge_shards(config,
    [f"results/shard{i}" for i in range(4)], "results/merged")
```

Legacy Probes without representation metadata require an explicit
`ResidualSpec`, as described in [defense integration](defense.md). ViT Probes
require a separate language-stream intervention specification.

## Preserved model-specific behavior

| Setting | Pi0.5 | SmolVLA |
|---|---|---|
| Suite | LIBERO-10 task order 0–9, first 50 initial states each | Same assignment through LeRobot's existing environment |
| Environment | Native `OffScreenRenderEnv`, 256 pixels | Native LeRobot `LiberoEnv`, 256 pixels, relative control |
| Warmup | Ten dummy steps by default | Existing environment's ten settling steps |
| Input cameras | Agent flipped in both axes, padded resize to 224, zero/masked wrists | Agent and wrist, existing LeRobot preprocessing |
| Patch | After agent resize | Flip both raw cameras, render patch, restore raw orientation before preprocessing |
| Geometry | Episode-fixed, shared across cameras | Same legacy renderer and geometry |
| Burst clock | Environment steps over horizon 520 | Horizon 90/109; replan interval **must** be 1, so environment steps equal policy forwards |
| Queue | Clear on either patch visibility transition | Replan every step; other intervals rejected |
| Policy RNG | Fold seed by global rollout ID, split once per actual forward | Seed Torch CPU/CUDA with seed + rollout ID before reset; draw noise directly on CUDA once per forward |
| Default denoising | Ten steps in native OpenPI sampler | Ten steps in native SmolVLA sampler |
| Success | Native `done` | `info.get("is_success", False)`; termination alone is insufficient |
| Limit | 520 control steps, excluding warmup | Same control-step limit; exact burst forward counts are enforced |

There is no extra noise draw on cached Pi0.5 actions or visibility changes.
SmolVLA uses `cudnn.benchmark=True` and TF32 matmul as in the legacy runner.
Native SmolVLA sampling has no additional read-only diagnostic forward.
OpenPI output transforms still run before unbatching, followed by truncation
to seven action components, exactly as in the old evaluator. The tested
Pi0.5 boundary produces a `[7,7]` converted chunk from native `[1,10,32]`
actions; moving unbatching earlier would change this legacy behavior.

Modes are the existing `probe`, `oracle`, `always` and `disabled` modes.
Oracle receives the patch visibility for the observation being replanned.
Pi0.5's legacy Oracle/always repeated-prefix conditions require
`timing="denoising"`; Probe conditions use `timing="prefix"`. Unsupported
combinations are rejected by the configuration or defense adapter.
Multiple selected features, RTC, non-LIBERO-10 suites and SmolVLA replan
intervals greater than one are outside this evaluator.

## Identity, resume and records

`libero_runtime_identity(model)` records ordered task names, BDDL and
initial-state content hashes, active numerical dependency versions, Python,
and pinned model-source revisions/hashes. Adapters verify these fields and
all artifact content hashes before loading. The supported external revisions
are unchanged from [defense integration](defense.md).
`adapters/patches/libero_sources.json` additionally pins LIBERO Python
sources/assets and the evaluation transformations. SmolVLA requires the
already modified LeRobot environment (`envs/libero.py`, `envs/configs.py`)
at revision `5865170d36442b907bb35f946e837eee18aafdf1` plus the existing
research changes recorded in `smolvla_libero_research.patch`. Apply it after
`smolvla_research.patch` when reconstructing that checkout. No external
source was changed for this integration. Native environment patching is
disabled; the package applies the recorded image overlay once.

Each output directory contains:

- `config.json`: experiment settings, artifacts and provenance.
- `partition.json`: worker index/count, independent of experiment identity.
- `results.jsonl`: completed episodes, Success, lengths, detection/intervention
  counts and visibility/intervention action steps.
- `events/NNN.json`: per-forward score, detection, applied gate, intervention
  count and action boundaries.
- `summary.json`: aggregate Success Rate, per-task results and detector rates.

`trace(kind, timestep, data)` includes the rollout ID and optionally exposes
observations, native normalized actions (`Prediction.normalized_actions`),
converted chunks and executed actions without retaining them in the default results.
Large activation captures remain the defense adapter's optional facility.
`ended` records environment termination; reaching `max_steps` can leave it
false. `success` always follows the model-specific rule above.

A directory lock rejects simultaneous writers. Events are atomically written
and synced before an episode completion record is appended and synced.
Interrupted, uncommitted episodes run again. Orphan events are overwritten.
Malformed/truncated JSON, duplicate IDs, missing events, inconsistent counts,
different configurations and out-of-shard IDs are rejected. Nothing silently
repairs or skips corrupt results. Shard merge uses the same validation.

## Reproducibility

Representative results, environment limitations and the unresolved paper/code
description are summarized in [reproducibility](reproducibility.md).
