"""Versioned Probe artifacts and explicit binding of legacy NPZ files."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import numpy as np

from .model import LogisticProbe, ResidualSpec


def _atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_probe(path: str | Path, probe: LogisticProbe) -> None:
    """Save a self-contained NPZ and a readable JSON sidecar with legacy keys."""
    import io

    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Probe artifacts use the .npz extension")
    spec = json.dumps(asdict(probe.representation), allow_nan=False)
    metadata = json.dumps(dict(probe.metadata), indent=2, allow_nan=False)
    reserved = {"mean", "scale", "coef", "intercept", "threshold", "schema_version",
                "representation_json", "metadata_json"}
    if reserved & probe.extra_arrays.keys():
        raise ValueError("extra_arrays must not overwrite Probe fields")
    buffer = io.BytesIO()
    np.savez(buffer, **probe.extra_arrays, mean=probe.mean, scale=probe.scale, coef=probe.coef,
             intercept=probe.intercept, threshold=probe.threshold, schema_version=np.int32(1),
             representation_json=np.asarray(spec), metadata_json=np.asarray(metadata))
    _atomic_write(path, buffer.getvalue())
    sidecar = {**dict(probe.metadata), "residual_spec": asdict(probe.representation)}
    _atomic_write(path.with_suffix(".json"), (json.dumps(sidecar, indent=2, allow_nan=False) + "\n").encode())


def load_probe(path: str | Path, *, representation: ResidualSpec | None = None) -> LogisticProbe:
    """Read a Probe without changing its arrays or recalibrating its threshold.

    Legacy artifacts must be bound to an explicit representation because
    their original metadata do not fully identify the residual input.
    Available legacy metadata are checked against this binding.
    """
    path = Path(path)
    required = ("mean", "scale", "coef", "intercept", "threshold")
    with np.load(path, allow_pickle=False) as artifact:
        arrays = {key: np.asarray(artifact[key]) for key in required}
        if "schema_version" in artifact:
            if int(artifact["schema_version"]) != 1:
                raise ValueError("unsupported Probe schema_version")
            raw_spec = json.loads(str(artifact["representation_json"]))
            raw_spec["image_keys"] = tuple(raw_spec["image_keys"])
            saved_spec = ResidualSpec(**raw_spec)
            if representation is not None and saved_spec != representation:
                raise ValueError("residual representation does not match the artifact")
            representation = saved_spec
            metadata = json.loads(str(artifact["metadata_json"]))
        else:
            if representation is None:
                raise ValueError("legacy Probe requires an explicit residual representation")
            sidecar = path.with_suffix(".json")
            metadata = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            for key, expected in (("layer", representation.layer), ("hidden_dim", representation.hidden_dim),
                                  ("pooling", representation.pooling), ("capture_point", representation.capture_point),
                                  ("model", representation.model), ("family", representation.family)):
                if key in metadata and metadata[key] != expected:
                    raise ValueError(f"legacy Probe {key} conflicts with the requested representation")
        if "layer" in artifact and int(artifact["layer"]) != representation.layer:
            raise ValueError("Probe layer conflicts with the representation")
        if "image_keys" in artifact and tuple(artifact["image_keys"].tolist()) != representation.image_keys:
            raise ValueError("Probe camera order conflicts with the representation")
        extra = {key: np.asarray(artifact[key]) for key in artifact.files
                 if key not in (*required, "schema_version", "representation_json", "metadata_json")}
    for key in ("intercept", "threshold"):
        if arrays[key].shape != ():
            raise ValueError(f"{key} must be a scalar")
        arrays[key] = arrays[key][()]
    return LogisticProbe(representation=representation, metadata=metadata, extra_arrays=extra, **arrays)
