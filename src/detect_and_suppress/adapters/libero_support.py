"""Artifact binding and data identity shared by the two LIBERO adapters."""

from __future__ import annotations

import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import platform

from ..defense import load_defense
from ..evaluation.config import Artifact


def load_evaluation_defense(config):
    source_identity(config.model)
    verify_evaluation_sources(config.model)
    for artifact in config.artifacts.values():
        artifact.verify()
    return load_defense(probe=config.artifacts["probe"].path, selection=config.artifacts["selection"].path,
                        sae_checkpoint=config.artifacts["sae"].path, config=config.defense,
                        feature_id=config.feature_id)


def source_identity(model: str) -> dict:
    """Reject incompatible model source sets and expose their exact identities."""
    manifest = json.loads((Path(__file__).with_name("patches") / "revisions.json").read_text())
    family = "openpi" if model == "pi05" else "smolvla"
    hashes = {}
    for relative, expected in manifest[family].items():
        module = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError(f"missing model module: {module}")
        digest = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
        if digest != expected["integrated" if family == "openpi" else "research"]:
            raise ValueError(f"unsupported model source revision: {spec.origin}")
        hashes[module] = digest
    return {"revision": manifest[f"{family}_revision"], "files": hashes}


def evaluation_source_identity(model: str) -> dict:
    """Hash installed LIBERO Python sources and the model's evaluation boundary."""
    spec = importlib.util.find_spec("libero.libero")
    if spec is None or spec.origin is None:
        raise ValueError("LIBERO is not installed")
    root = Path(spec.origin).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    modules = (("openpi.transforms", "openpi.policies.libero_policy", "openpi.training.config",
                "openpi.models.pi0_config", "openpi_client.image_tools") if model == "pi05" else
               ("lerobot.envs.libero", "lerobot.envs.configs"))
    hashes = {}
    for name in modules:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"missing evaluation module: {name}")
        hashes[name] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    from libero.libero import get_libero_path
    return {"libero_python_sha256": digest.hexdigest(), "modules": hashes,
            "assets_sha256": Artifact.identify(get_libero_path("assets")).sha256}


def verify_evaluation_sources(model: str) -> None:
    expected = json.loads((Path(__file__).with_name("patches") / "libero_sources.json").read_text())[model]
    if evaluation_source_identity(model) != expected:
        raise ValueError("unsupported LIBERO/evaluation source set; use the recorded research revisions and patches")


def load_suite():
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    if suite.n_tasks != 10:
        raise ValueError("expected ten LIBERO-10 tasks")
    return suite


def initial_states(suite, task_id):
    import torch
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    states = torch.load(path, weights_only=False)
    if len(states) < 50:
        raise ValueError("LIBERO task has fewer than fifty initial states")
    return states


def libero_data_identity() -> dict:
    """Record task order, BDDL and initial-state hashes for the installed suite."""
    from libero.libero import get_libero_path

    suite = load_suite()
    rows = []
    for i in range(10):
        task = suite.get_task(i)
        paths = {
            "bddl": Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
            "initial_states": Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
        }
        rows.append(dict(task_id=i, task=str(task.language), **{
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}))
    return {"libero_10": rows}


def validate_libero_identity(config):
    actual = libero_runtime_identity(config.model)
    if any((config.provenance or {}).get(key) != value for key, value in actual.items()):
        raise ValueError("record the active environment using libero_runtime_identity(model)")


def libero_runtime_identity(model: str) -> dict:
    """Identify task order/data, pinned model sources and active numerical dependencies."""
    versions = {}
    for name in ("numpy", "mujoco", "robosuite", "torch", "jax", "Pillow", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"versions": versions, "python": platform.python_version(),
            "model_sources": source_identity(model), **libero_data_identity()}
