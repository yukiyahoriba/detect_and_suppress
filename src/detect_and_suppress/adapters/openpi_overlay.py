"""Prepare a pinned OpenPI source overlay without modifying its input checkout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess


PATCHES = Path(__file__).with_name("patches")


def prepare_openpi_overlay(source: str | Path, output: str | Path) -> Path:
    """Copy supported sources, apply required patches, and record exact hashes.

    Accepts the pinned upstream model files or the pinned research overlay.
    Put the returned ``src`` directory before the installed OpenPI sources.
    The output must not exist; checkpoints and environments are not copied.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output == source or source in output.parents:
        raise ValueError("overlay output must be outside the source repository")
    manifest = json.loads((PATCHES / "revisions.json").read_text())
    hashes = {relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
              for relative in manifest["openpi"]}
    stage = next((stage for stage in ("upstream", "research", "integrated")
                  if all(hashes[path] == values[stage] for path, values in manifest["openpi"].items())), None)
    if stage is None:
        raise ValueError("OpenPI model files do not match a supported revision/hash set")
    if shutil.which("patch") is None:
        raise RuntimeError("preparing the OpenPI overlay requires the patch executable")
    try:
        output.mkdir(parents=True)
        shutil.copytree(source / "src", output / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        names = (["openpi_research.patch"] if stage == "upstream" else [])
        if stage != "integrated":
            names.append("openpi_defense.patch")
        for name in names:
            completed = subprocess.run(["patch", "--batch", "--forward", "-p1"], input=(PATCHES / name).read_text(),
                                       cwd=output, text=True, capture_output=True)
            if completed.returncode:
                raise RuntimeError(f"failed to apply {name}: {completed.stdout}{completed.stderr}")
        for relative, expected in manifest["openpi"].items():
            if hashlib.sha256((output / relative).read_bytes()).hexdigest() != expected["integrated"]:
                raise RuntimeError(f"patched source hash mismatch: {relative}")
        (output / "defense_overlay.json").write_text(json.dumps({
            "source": str(source), "input_stage": stage, "patches": names, **manifest,
        }, indent=2) + "\n")
        return output / "src"
    except BaseException:
        if output.exists():
            shutil.rmtree(output)
        raise
