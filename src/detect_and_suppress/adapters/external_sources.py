"""Reconstruct pinned source overlays without mutating external checkouts."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

from .openpi_overlay import PATCHES, prepare_openpi_overlay


def prepare_evaluation_overlay(model: str, source: str | Path, output: str | Path) -> Path:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if source == output or source in output.parents:
        raise ValueError('overlay output must be outside the source checkout')
    if model not in ('pi05', 'smolvla'):
        raise ValueError('unsupported model')
    manifest = json.loads((PATCHES/'revisions.json').read_text())
    boundary = json.loads((PATCHES/'libero_sources.json').read_text())[model]['modules']
    if shutil.which('patch') is None:
        raise RuntimeError('patch executable is required')
    try:
        if model == 'pi05':
            result = prepare_openpi_overlay(source, output)
            expected = boundary['openpi.policies.libero_policy']
            path = output/'src/openpi/policies/libero_policy.py'
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                _apply(output, 'openpi_libero_research.patch')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('unsupported OpenPI LIBERO output transform')
        else:
            output.mkdir(parents=True)
            shutil.copytree(source/'src', output/'src', symlinks=True, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            for relative, hashes in manifest['smolvla'].items():
                actual = hashlib.sha256((source/relative).read_bytes()).hexdigest()
                if actual not in (hashes['upstream'], hashes['research']):
                    raise ValueError(f'unsupported LeRobot model source: {relative}')
            stages = {hashlib.sha256((source/p).read_bytes()).hexdigest() == v['research'] for p,v in manifest['smolvla'].items()}
            if stages == {False}:
                _apply(output, 'smolvla_research.patch')
            elif stages != {True}:
                raise ValueError('mixed LeRobot model source stages')
            if any(hashlib.sha256((output/('src/'+name.replace('.','/')+'.py')).read_bytes()).hexdigest() != sha
                   for name,sha in boundary.items()):
                _apply(output, 'smolvla_libero_research.patch')
            for relative, hashes in manifest['smolvla'].items():
                if hashlib.sha256((output/relative).read_bytes()).hexdigest()!=hashes['research']:
                    raise ValueError(f'patched model hash mismatch: {relative}')
            result=output/'src'
        for name,sha in boundary.items():
            # openpi_client is a separately installed workspace package.
            relative=output/('src/'+name.replace('.','/')+'.py')
            if relative.is_file() and hashlib.sha256(relative.read_bytes()).hexdigest()!=sha:
                raise ValueError(f'evaluation boundary hash mismatch: {name}')
        return result
    except BaseException:
        if output.exists():
            shutil.rmtree(output)
        raise


def _apply(output, name):
    result=subprocess.run(['patch','--batch','--forward','-p1'],cwd=output,
                          input=(PATCHES/name).read_text(),text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{name}: {result.stdout}{result.stderr}')
