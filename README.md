# Detect and Suppress

SAE-feature-based defenses against adversarial patches in Pi0.5 and SmolVLA.

This repository contains code to:

1. Select SAE features correlated with patch presence.
2. Train residual Probes for attack detection.
3. Apply detection-gated feature suppression during VLA inference.
4. Evaluate defense performance on LIBERO-10.

```text
src/detect_and_suppress/   Python library and CLI
  selection.py            Feature correlations and ranking
  probes/                 Probe training and inference
  defense.py              Defense configuration and artifacts
  adapters/               Pi0.5 / SmolVLA integration
  evaluation/             LIBERO rollouts, results and shards
configs/                  Experiment configurations
artifacts/                Artifact inventory and selected features
examples/                 Small runnable examples
```

## Installation

Python >=3.10. Install in the environment for your model:

```bash
python -m pip install -e '.[probe-torch,evaluation]'
das --help
```

Pi0.5 and SmolVLA use separate OpenPI / LeRobot environments. See
[environment setup](environments/README.md) for pinned revisions and
[setup commands](docs/running_experiments.md#external-dependencies) for the
required source patches and LIBERO configuration.

### Checkpoints

Evaluation uses pretrained VLA, SAE and Probe checkpoints, selected features,
and an existing adversarial patch. Public downloads for the author artifacts
are pending. Available identities, checksums and the expected directory layout
are listed in the [artifact guide](artifacts/README.md).

## Pipeline

### 1. Feature selection

Generate a small synthetic dataset and select features:

```bash
python examples/make_sample_data.py --output sample-data
das select --config sample-data/selection.json --output sample-data/selected.json
```

For research data, supply your SAE and saved episode activations using the
[configuration format](configs/README.md) and [Python API](docs/selection.md).

### 2. Probe training

```bash
das train-probe --config sample-data/probe.json --output sample-data/trained.npz
```

The example uses synthetic residuals. Real-data training supports the original
Pi0.5, SmolVLA and sklearn profiles; see [Probe training](docs/probes.md).

### 3. LIBERO evaluation

With the model environment and artifacts prepared, run a recorded condition:

```bash
das prepare-evaluation \
  --template configs/evaluation/pi05_libero10_patch3_probe.json \
  --artifact-root weights --output runtime/pi05.json
das evaluate --config runtime/pi05.json --output results/pi05
```

This evaluates all 500 episodes and resumes completed results on restart.
