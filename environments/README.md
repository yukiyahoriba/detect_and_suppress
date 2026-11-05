# External environments

Use separate environments for Pi0.5 and SmolVLA. The package requires Python
>=3.10; the evaluated environments used the versions below. These are recorded
working versions, not complete dependency lockfiles. Installing the full
CUDA/model/simulator stacks from zero has not yet been validated.

| Dependency | Pi0.5 | SmolVLA |
|---|---|---|
| Python | 3.11.14 | 3.10.19 |
| OpenPI revision | `981483dca0fd9acba698fea00aa6e52d56a66c58` | — |
| LeRobot revision | OpenPI's pinned dependency | `5865170d36442b907bb35f946e837eee18aafdf1` |
| LIBERO code | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` | `hf_libero==0.1.3` |
| NumPy | 1.26.4 | 2.2.6 |
| Torch | 2.7.1 | 2.10.0 |
| JAX | 0.5.3 | — |
| MuJoCo | 3.4.0 | 3.5.0 |
| robosuite | 1.4.1 | 1.4.0 |
| Pillow | 12.1.0 | 12.1.1 |
| Transformers | 4.53.2 | 4.57.6 |

Install the external stack using the pinned revision's upstream instructions.
OpenPI has its own uv workspace and `openpi-client` package; apply its pinned
Transformers replacement files as instructed upstream. Pi0.5 LIBERO additionally
needs `gym==0.26.2` and `easydict`. System requirements include compatible CUDA
drivers, EGL/OpenGL, `git` and `patch`. Do not silently upgrade numerical
packages to work around dependency conflicts.

Use `das prepare-external` to create a source overlay outside the input
checkout. It applies the packaged patches and verifies model/evaluation file
hashes. Put the returned `src` path first on `PYTHONPATH` in a new process.

- Pi0.5: `openpi_research.patch`, `openpi_defense.patch`, and
  `openpi_libero_research.patch`.
- SmolVLA: `smolvla_research.patch`, then `smolvla_libero_research.patch`.

The recipes use the pinned OpenPI/LeRobot revisions above and do not modify
input checkouts. Unsupported source hashes are rejected. Required hashes are
packaged under `detect_and_suppress.adapters/patches/`.

Both environments use the original pinned LIBERO checkout's assets, BDDL and
initial states. SmolVLA uses HF LIBERO Python code with those original assets
and task data. `examples/configure_libero.py` writes native `config.yaml` for
an explicit checkout; export `LIBERO_CONFIG_PATH` before importing LIBERO.
Task order, BDDL and initial-state hashes are recorded in the prepared config.
`libero-assets.json` identifies the 585 asset files by relative name, size and
SHA256. The directory checksum is enforced by the evaluation source manifest.

Model tokenizers and auxiliary backbone files must also be provisioned.
`das prepare-runtime` verifies the inventoried inputs and stages isolated
native caches. SmolVLA's cache ref is pinned to the recorded revision; use
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to prevent silent updates.

See [execution commands](../docs/running_experiments.md) and
[reproducibility limits](../docs/reproducibility.md). Package/CLI and model
integration were checked with preinstalled numerical dependencies. A complete
fresh dependency installation and current upstream downloads remain unverified.
