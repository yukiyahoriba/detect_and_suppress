# Running experiments

Detailed setup and execution instructions. Run commands from the repository root.

A Python package for reproducing SAE-feature-based defenses against adversarial
patches in VLA policies. The research pipeline selects features using saved
activations, trains residual attack-detection Probes, then combines detection
and feature suppression during native Pi0.5 or SmolVLA inference in LIBERO-10.
VLA inference, SAE training, patch optimization and the simulator remain in
their existing external implementations.

## Installation

Use Python >=3.10. Install this package inside the environment for the intended
stage; Pi0.5 and SmolVLA require separate external environments.

```bash
python -m pip install -e '.[probe-torch,evaluation]'
das --help
python -m detect_and_suppress --help
```

The base package and Probe inference require NumPy. `probe-torch` supports the
Pi0.5/SmolVLA training profiles; use `probe-sklearn` for the separate
`sklearn_dense` profile. `evaluation` adds Pillow. These extras do not install
OpenPI, LeRobot, LIBERO, CUDA, or all of their dependencies.

Offline installation into an environment with dependencies already installed
was verified from an independent source copy:

```bash
python -m pip wheel . --no-deps --no-build-isolation --no-index -w wheels
python -m pip install --no-deps --no-index wheels/detect_and_suppress-0.1.0-py3-none-any.whl
```

New venvs used preinstalled numerical dependencies for validation. Installing
all external dependencies from zero has **not** been validated; the observed
inventories are not fresh-install lockfiles.

## External Dependencies

See [model-specific environments](../environments/README.md) for exact versions,
upstream revisions, LIBERO assets, required source patches and validation limits.
Use the pinned external checkouts and their upstream setup instructions.
Set `OPENPI_SOURCE`, `LEROBOT_SOURCE`, and `LIBERO_SOURCE` to your own locations.
They are repository roots, not this research project's old directory.

In the Pi0.5 environment:

```bash
das prepare-external --model pi05 --source "$OPENPI_SOURCE" --output runtime/openpi
export PYTHONPATH="$PWD/runtime/openpi/src:$OPENPI_SOURCE/packages/openpi-client/src:$LIBERO_SOURCE"
```

In the SmolVLA environment (installed `hf_libero==0.1.3`):

```bash
das prepare-external --model smolvla --source "$LEROBOT_SOURCE" --output runtime/lerobot
export PYTHONPATH="$PWD/runtime/lerobot/src"
```

Prepare native LIBERO path configuration once per checkout; export the path
in each environment. Both use the
original pinned LIBERO checkout's assets, BDDL and initial states:

```bash
python examples/configure_libero.py --source "$LIBERO_SOURCE" --output runtime/libero-config
export LIBERO_CONFIG_PATH="$PWD/runtime/libero-config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Use a GPU visible to your process. No GPU index is embedded in configs or
commands. Overlays and native config generation were checked against clean
upstream revisions; input checkouts are left unchanged. Run one policy
adapter per process, with separate output directories for concurrent shards.

## Downloading Checkpoints

Place the approved artifacts in the layout in [artifacts/README.md](../artifacts/README.md).
The manifest includes sizes, SHA256, exact checkpoint file sets, corresponding
conditions and known upstream model revisions. Unpublished artifacts have
`url: null`; there is currently no working public download for the full paper
condition. Evaluation can use previously obtained matching artifacts without
re-running feature selection or Probe training.

Copy the supplied feature metadata into your artifact root:

```bash
mkdir -p weights/pi05 weights/smolvla
cp artifacts/selections/pi05.json weights/pi05/selection.json
cp artifacts/selections/smolvla.json weights/smolvla/selection.json
das verify-artifacts --manifest artifacts/manifest.json --artifact-root weights --model pi05
```

The native models also require auxiliary tokenizer/backbone files; do not rely
on an undocumented user cache. After provisioning the manifest's runtime
artifacts, stage an isolated cache. In Pi0.5:

```bash
das prepare-runtime --model pi05 --manifest artifacts/manifest.json --artifact-root weights --output runtime/pi05-assets
export OPENPI_DATA_HOME="$PWD/runtime/pi05-assets/openpi"
```

In SmolVLA:

```bash
das verify-artifacts --manifest artifacts/manifest.json --artifact-root weights --model smolvla
das prepare-runtime --model smolvla --manifest artifacts/manifest.json --artifact-root weights --output runtime/smolvla-assets
export HF_HOME="$PWD/runtime/smolvla-assets/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

This preserves native loading behavior while fixing the HF cache revision.
Artifact checks never download, change weights or recalibrate thresholds.

## Feature Selection

Run a small synthetic example, independent of VLA inference and LIBERO:

```bash
python examples/make_sample_data.py --output sample-data
das select --config sample-data/selection.json --output sample-data/selected.json --correlations sample-data/correlations.csv
```

For real data, supply the trained SAE and episode NPZ files with residual
activations, patch presence and policy-forward signals. The JSON input schema
is described in [configs/README.md](../configs/README.md). Data paths are relative
to that config. The CLI calls the existing encoding, per-episode Pearson,
unweighted aggregation and ranking APIs; undefined correlations and nonfiring
features retain the legacy behavior. See [reproducibility](reproducibility.md).

## Probe Training

Using the same generated sample files:

```bash
das train-probe --config sample-data/probe.json --output sample-data/trained.npz
```

This creates a self-contained Probe NPZ, JSON metadata and a `.metrics.json`
report with reference/runtime dev/test metrics. The output stem must differ
from the input config stem because Probe metadata uses the `.json` sidecar.
Outputs must be new; nothing is silently overwritten. The synthetic sample
checks packaging and API wiring, not paper-level performance.

Real-data configs select `dense`, `paired`, `episodes` or explicit `arrays`
collection layouts and a `sklearn_dense`, `pi05` or `smolvla` training profile.
Train/dev/test membership stays under the collection API, with no automatic
resplitting. Optimizers, class weights, precision, C search, candidate metric,
clean-dev calibration and strict decision rule remain profile-specific.
See [Probe API and formats](probes.md). Training uses residuals directly
and does not depend on SAE feature selection.

## LIBERO Evaluation

After provisioning and verifying artifacts and preparing the appropriate
external environment, bind a portable template to a local experiment
config. It verifies all primary artifacts and source identities, binds the
checkpoint location and records the active runtime. Keep this config for
all shards and resume attempts. Preparation outputs must be new.

Pi0.5, one complete episode including visibility transitions:

```bash
das prepare-evaluation --template configs/evaluation/pi05_libero10_patch3_probe.json --artifact-root weights --output runtime/pi05.json
das evaluate --config runtime/pi05.json --output results/pi05-small --shard-index 0 --shard-count 500 --max-new-episodes 1
```

SmolVLA, a separately validated episode with visibility transitions:

```bash
das prepare-evaluation --template configs/evaluation/smolvla_libero10_patch3_probe.json --artifact-root weights --output runtime/smolvla.json
das evaluate --config runtime/smolvla.json --output results/smolvla-small --shard-index 145 --shard-count 500 --max-new-episodes 1
```

`shard-index/count` partitions global IDs 0–499 by modulo. Count 500 isolates
one episode; it does not mean that the small run performs 500 rollouts.
For all 500, omit shard arguments and the episode limit. For multiple workers,
use one shared config and a fixed count, each with its own index/output.
To merge four completed shards, for example:

```bash
das merge-evaluation --config runtime/pi05.json --sources results/shard0 results/shard1 results/shard2 results/shard3 --output results/merged
```

Partial merging requires `--allow-partial`. Re-run an evaluation command with
the same config/output/partition to resume. Completed episodes are validated
and skipped. Config mismatch, duplicate IDs, corrupt rows and missing event
logs are rejected. An uncommitted episode is rerun after interruption.
Outputs include `config.json`, `partition.json`, `results.jsonl`, per-forward
`events/`, and `summary.json`. See [evaluation semantics and API](evaluation.md).

## Reproducibility and limitations

See [reproducibility](reproducibility.md) for reported results, current
limitations and the paper/code description discrepancy.
