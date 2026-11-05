# Defense integration

`defense.py` binds existing Probe, feature-selection and SAE artifacts.
Model-specific adapters invoke the existing VLA inference engine and apply
the prepared defense inside its prefix processing. No action integration
or simulator loop is implemented by this package.

## Configuration and usage

`DefenseConfig(detector, intervention, lambda_, mode="probe")` takes two
`ResidualSpec` objects declaring layers, capture locations, pooling,
dimensions, camera order and checkpoint identity. `load_defense` reuses
`load_probe`, `load_selection` and the existing direction loader. It checks
the feature's expert/layer, dimensions and recorded SAE identity. A recorded
SAE hash supports relocated weights; relative SAE paths are resolved against
the selection JSON. Older selections without an identity are explicitly
bound to the checkpoint/specification supplied by the caller.

One selected direction is applied. Specify `feature_id` when needed to
disambiguate entries; the configured intervention layer/expert also constrain
the choice. If the specifications declare a checkpoint identity, supply that
identity when attaching the already loaded model.

For a trained SmolVLA VLM-residual Probe:

```python
from dataclasses import replace
from detect_and_suppress.probes import load_probe
from detect_and_suppress.defense import DefenseConfig, load_defense
from detect_and_suppress.adapters.smolvla import SmolVLADefense

probe = load_probe("probe.npz")
config = DefenseConfig(probe.representation,
                       replace(probe.representation, layer=9), lambda_=0.2)
bundle = load_defense(probe="probe.npz", selection="selection.json",
                      sae_checkpoint="sae.npz", config=config, feature_id=123)
with SmolVLADefense(policy.model, bundle, checkpoint=checkpoint_identity,
                    capture_activations=True) as adapter:
    adapter.reset_episode("episode-0")
    actions = adapter.sample_actions(
        images, image_masks, language_tokens, language_masks, state, noise=noise,
    )
    record = adapter.last_record
    scores, detected, applied = record.scores, record.detected, record.applied
    before, after = record.before, record.after
```

For legacy Probes, bind their missing fields explicitly using
`load_probe(path, representation=spec)`, or supply that detector spec to
`DefenseConfig`. A ViT Probe needs a separate VLM/PaliGemma intervention
specification, rather than a copy of its ViT capture/pooling fields.

## Modes and state

Modes are `probe`, `always`, `disabled`, and `oracle`, with optional per-call
overrides. Oracle mode requires explicit `oracle_active` for the observation
being sampled. The adapter does not infer patch presence or manage action
queues: burst scheduling and patch-switch replanning remain with the caller.

`record.detected` is the learned Probe's strict threshold decision;
`record.applied` is the gate used by the selected mode. Records include
episode/forward IDs, layers, scores, gates, timing and intervention-call
counts. `capture_activations=True` retains device-native before/after tensors
on `last_record`; the default omits them. Bounded `history` retains host
records without those large tensors. Repeated-prefix timing captures the
final iteration and counts applications across denoising iterations.

`reset_episode` clears state/history without reseeding the model. Supply
native noise/RNG arguments for reproducibility. Nested/overlapping managed
calls are rejected. `close` removes owned hooks/wrappers, clears state and
is idempotent. Use context managers for exception cleanup. Attaching the
SmolVLA adapter alone does not enable intervention in arbitrary raw calls:
use `adapter.sample_actions`, or manage a fresh sampler invocation explicitly:

```python
with adapter.forward(mode="oracle", oracle_active=patch_active):
    actions = existing_instrumented_sampler(...)
```

Do not wrap an environment step that only returns a cached action.

## Model connections

SmolVLA uses pre-hooks on VLM `input_layernorm` inputs. VLM scores use the
original FP32 masked mean. ViT scores use post-block FP32 token means and
agent-then-wrist concatenation. Intervention covers every prefix token,
including padding, as in the original hooks. A same-layer detector observes
the original residual before applying the intervention.

Instance-local wrappers observe the existing `embed_prefix` result and scope
the existing `vlm_with_expert.forward`; their inference computation is not
copied. Suffix and additional diagnostic forwards are excluded. Use the
model's canonical camera order and put the model in eval mode first.
Directions are prepared once on CPU as in the old hooks. The shared Torch
kernel preserves the old scalar/batch gate multiplication order and accepts
a prepared direction, removing the earlier integration precision differences.
Saved decoder-column strides are retained during CPU preparation without
mutating the weights.

Pi0.5 requires a source overlay prepared before importing/loading OpenPI:

```python
from detect_and_suppress.adapters.openpi_overlay import prepare_openpi_overlay
src = prepare_openpi_overlay("/path/to/supported/openpi", "/path/to/new/overlay")
```

Put the returned `src` first on `PYTHONPATH` in a new inference process.
The helper copies source files, applies pinned patches, and verifies hashes.
It never edits its input checkout; output must not exist. The system `patch`
executable is required. Upstream, research and integrated source sets are
recognized by exact model-file hashes.

```python
from detect_and_suppress.adapters.pi05 import Pi05Defense

with Pi05Defense(model, pi05_bundle, checkpoint=checkpoint_identity,
                  capture_activations=True) as adapter:
    actions = adapter.sample_actions(rng, observation, noise=noise, num_steps=10)
    record = adapter.last_record
```

PaliGemma pooling remains unmasked. The native layer-0 prefix-embedding fast
path, including its dtype order, is retained. Later layers use the original
Gemma scan gate carry. SigLIP uses native `+mlp` block outputs and FP32 spatial
means in the Probe's explicit camera order. The native action loop, cache,
positions and attention masks remain in OpenPI.

Default `timing="prefix"` reproduces Probe-gated prefix prefill.
`timing="denoising"` delegates to the existing repeated-prefix path used by
Oracle/always experiments:

```python
with Pi05Defense(model, pi05_bundle, checkpoint=checkpoint_identity,
                  timing="denoising") as adapter:
    actions = adapter.sample_actions(rng, observation, noise=noise,
                                     mode="oracle", oracle_active=patch_active)
```

Repeated-prefix timing supports a PaliGemma detector with Oracle, always or
disabled mode. Probe gating uses prefix timing. ViT plus repeated-prefix
timing is explicitly rejected. The adapter never silently changes timing.

The OpenPI patch adds optional controls/capture outputs to existing methods
and invokes the package's JAX kernel in the Gemma hook. Its return annotation
is extended for capture dictionaries, retaining runtime type checks. Calls
without integration arguments use the original code path.

## External revisions

`adapters/patches/revisions.json` records file hashes and revisions:

- OpenPI `981483dca0fd9acba698fea00aa6e52d56a66c58`:
  `openpi_research.patch` reconstructs the three existing research model
  files (`gemma.py`, `pi0.py`, `siglip.py`); `openpi_defense.patch` is the
  additional change for this integration.
- LeRobot `5865170d36442b907bb35f946e837eee18aafdf1`:
  `smolvla_research.patch` records the already modified `modeling_smolvla.py`
  and `smolvlm_with_expert.py`. LIBERO evaluation additionally uses `smolvla_libero_research.patch`.

These recipes cover raw model inference, not existing evaluation/serving
modifications. The supported SmolVLA environment uses Python 3.10.

## Limits

One selected feature is supported. Supported timing/mode combinations are
validated explicitly; arbitrary concurrent model calls are not supported.
See [reproducibility and limitations](reproducibility.md).
