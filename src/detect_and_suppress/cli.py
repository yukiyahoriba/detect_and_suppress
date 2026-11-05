"""Thin, file-oriented entry points for the research pipeline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resolve_path(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def spec_from_dict(value):
    from .probes import ResidualSpec
    value = dict(value)
    value['image_keys'] = tuple(value.get('image_keys', ()))
    return ResidualSpec(**value)


def selection_command(args):
    import numpy as np
    from .sae_checkpoint import load_sae_encoder, encode_activations
    from .selection import (EpisodeLatents, compute_correlations, aggregate_correlations,
                            rank_features, select_features, save_selection, write_correlations)
    from .evaluation import Artifact
    cfg = read_json(args.config)
    if cfg.pop('schema_version') != 1:
        raise ValueError('unsupported selection configuration')
    base = Path(args.config).resolve().parent
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.correlations and (args.correlations.exists() or args.correlations.resolve() == output.resolve()):
        raise ValueError('correlation output must be a new path distinct from selection')
    sae_path = resolve_path(base, cfg.pop('sae'))
    sae = load_sae_encoder(sae_path, top_k=cfg.pop('top_k', None))
    rows = []
    inputs = []
    for item in cfg.pop('episodes'):
        item = dict(item)
        path = resolve_path(base, item.pop('path'))
        with np.load(path, allow_pickle=False) as values:
            latents = encode_activations(values['activations'], sae)
            rows.extend(compute_correlations([EpisodeLatents(**item, latents=latents,
                patch_signal=values['patch_signal'], policy_forward_signal=values['policy_forward_signal'])]))
        inputs.append(asdict(Artifact.identify(path)))
    threshold = cfg.pop('threshold', .6)
    criterion = cfg.pop('criterion', 'mean')
    top_n = cfg.pop('top_n', 20)
    metadata = dict(cfg.pop('metadata', {}))
    if cfg:
        raise ValueError(f'unknown selection settings: {sorted(cfg)}')
    metadata.update(sae_sha256=Artifact.identify(sae_path).sha256, top_k=sae.top_k,
                    source_episodes=inputs)
    ranking = rank_features(aggregate_correlations(rows, threshold=threshold), criterion=criterion)
    save_selection(output, select_features(ranking, top_n=top_n, metadata=metadata))
    if args.correlations:
        write_correlations(args.correlations, rows)
    return {'selection': str(output), 'features': len(ranking.features[:top_n])}


def training_command(args):
    import numpy as np
    from .probes import (ProbeDataset, ProbeSplits, ProbeTrainingConfig, load_dense_collection,
        load_paired_collection, load_episode_collection, train_probe, save_probe)
    cfg = read_json(args.config)
    if cfg.pop('schema_version') != 1:
        raise ValueError('unsupported training configuration')
    base = Path(args.config).resolve().parent
    output = Path(args.output)
    if output.suffix != '.npz':
        raise ValueError('Probe output must use .npz')
    for path in (output, output.with_suffix('.json'), output.with_suffix('.metrics.json')):
        if path.exists():
            raise FileExistsError(path)
    data = dict(cfg.pop('data'))
    layout = data.pop('layout')
    if layout == 'arrays':
        representation = spec_from_dict(data.pop('representation'))
        split_unit = data.pop('split_unit', 'prespecified')
        datasets = {}
        for split in ('train', 'dev', 'test'):
            if split not in data:
                if split != 'test':
                    raise ValueError(f'missing {split} split')
                continue
            with np.load(resolve_path(base, data.pop(split)), allow_pickle=False) as archive:
                datasets[split] = ProbeDataset(**{key: archive[key] for key in archive.files})
        splits = ProbeSplits(representation, **datasets, split_unit=split_unit)
    else:
        root = resolve_path(base, data.pop('root'))
        if layout == 'dense':
            splits = load_dense_collection(root, representation=spec_from_dict(data.pop('representation')))
        elif layout == 'paired':
            splits = load_paired_collection(root, layer=data.pop('layer'))
        elif layout == 'episodes':
            splits = load_episode_collection(root, family=data.pop('family'), layer=data.pop('layer'), fold=data.pop('fold', 0))
        else:
            raise ValueError(f'unsupported collection layout: {layout}')
    if data:
        raise ValueError(f'unknown data settings: {sorted(data)}')
    training = dict(cfg.pop('training'))
    if training.get('c_values') is not None:
        training['c_values'] = tuple(training['c_values'])
    if cfg:
        raise ValueError(f'unknown training settings: {sorted(cfg)}')
    result = train_probe(splits, config=ProbeTrainingConfig(**training))
    save_probe(output, result.probe)
    metrics = {name: getattr(result, name) for name in ('selected_c', 'dev_reference', 'dev_runtime', 'test_reference', 'test_runtime')}
    write_json(output.with_suffix('.metrics.json'), metrics)
    return metrics


def bind_probe_checkpoint(defense, probe_path, checkpoint):
    """Keep versioned residual metadata intact; bind only legacy/missing identities."""
    import numpy as np
    from .probes import load_probe
    with np.load(probe_path, allow_pickle=False) as artifact:
        versioned = 'schema_version' in artifact
    saved_checkpoint = checkpoint
    if versioned:
        probe = load_probe(probe_path)
        saved_checkpoint = probe.representation.checkpoint
        expected = {**defense['detector'], 'checkpoint': saved_checkpoint}
        probe.validate_representation(spec_from_dict(expected))
        if saved_checkpoint is not None and saved_checkpoint != checkpoint:
            raise ValueError('versioned Probe checkpoint identity differs; explicit verified relocation through the Python API is required')
    defense['detector']['checkpoint'] = saved_checkpoint
    defense['intervention']['checkpoint'] = checkpoint


def prepare_command(args):
    """Bind a portable template to verified local artifacts and active runtime."""
    from .evaluation import Artifact, EvaluationConfig
    from .defense import load_defense
    from .adapters.libero_support import libero_runtime_identity, verify_evaluation_sources
    template = read_json(args.template)
    if template.pop('template_version') != 1:
        raise ValueError('unsupported evaluation template')
    paper = template.pop('paper')
    config = template.pop('evaluation')
    if template:
        raise ValueError(f'unknown template fields: {sorted(template)}')
    for name, value in config['artifacts'].items():
        path = resolve_path(args.artifact_root, value['path'])
        identified = Artifact.identify(path)
        if identified.sha256 != value['sha256']:
            raise ValueError(f'{name} checksum mismatch: {path}')
        config['artifacts'][name] = asdict(identified)
    checkpoint = config['artifacts']['checkpoint']['path']
    bind_probe_checkpoint(config['defense'], config['artifacts']['probe']['path'], checkpoint)
    verify_evaluation_sources(config['model'])
    config['provenance'] = {**libero_runtime_identity(config['model']), 'paper': paper}
    bound = EvaluationConfig.from_dict(config)
    load_defense(probe=bound.artifacts['probe'].path, selection=bound.artifacts['selection'].path,
                 sae_checkpoint=bound.artifacts['sae'].path, config=bound.defense, feature_id=bound.feature_id)
    write_json(args.output, bound.to_dict())
    return {'config_id': bound.fingerprint, 'config': str(args.output)}


def evaluation_command(args):
    from .evaluation import EvaluationConfig, Shard, run_evaluation
    config = EvaluationConfig.from_dict(read_json(args.config))
    if config.model == 'pi05':
        from .adapters.pi05_libero import Pi05LiberoAdapter as Adapter
    else:
        from .adapters.smolvla_libero import SmolVLALiberoAdapter as Adapter
    with Adapter(config) as adapter:
        return run_evaluation(config, adapter, args.output, shard=Shard(args.shard_index, args.shard_count),
                              max_new_episodes=args.max_new_episodes)


def merge_command(args):
    from .evaluation import EvaluationConfig, merge_shards
    return merge_shards(EvaluationConfig.from_dict(read_json(args.config)), args.sources, args.output,
                        require_complete=not args.allow_partial)


def artifacts_command(args):
    from .evaluation import Artifact
    manifest = read_json(args.manifest)
    if manifest['schema_version'] != 1:
        raise ValueError('unsupported artifact manifest')
    checked = []
    for item in manifest['artifacts']:
        if args.model and item['model'] != args.model:
            continue
        artifact = Artifact.identify(resolve_path(args.artifact_root, item['path']))
        if artifact.sha256 != item['sha256']:
            raise ValueError(f"checksum mismatch: {item['id']}")
        checked.append(item['id'])
    return {'verified': checked}


def external_command(args):
    from .adapters.external_sources import prepare_evaluation_overlay
    return {'pythonpath': str(prepare_evaluation_overlay(args.model, args.source, args.output))}


def runtime_command(args):
    from .runtime_assets import prepare_runtime_assets
    return prepare_runtime_assets(read_json(args.manifest), args.artifact_root, args.output, model=args.model)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='das', description='Detect and Suppress experiment stages')
    sub = parser.add_subparsers(dest='command', required=True)
    for name, function in [('select', selection_command), ('train-probe', training_command), ('evaluate', evaluation_command)]:
        p = sub.add_parser(name)
        p.add_argument('--config', required=True, type=Path)
        p.add_argument('--output', required=True, type=Path)
        p.set_defaults(function=function)
        if name == 'select':
            p.add_argument('--correlations', type=Path)
        if name == 'evaluate':
            p.add_argument('--shard-index', type=int, default=0)
            p.add_argument('--shard-count', type=int, default=1)
            p.add_argument('--max-new-episodes', type=int)
    p = sub.add_parser('prepare-evaluation', help='verify artifacts/sources and freeze a local evaluation config')
    p.add_argument('--template', required=True, type=Path)
    p.add_argument('--artifact-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.set_defaults(function=prepare_command)
    p = sub.add_parser('merge-evaluation')
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--sources', nargs='+', required=True, type=Path)
    p.add_argument('--allow-partial', action='store_true')
    p.set_defaults(function=merge_command)
    p = sub.add_parser('verify-artifacts')
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--artifact-root', required=True, type=Path)
    p.add_argument('--model', choices=['pi05', 'smolvla'])
    p.set_defaults(function=artifacts_command)
    p = sub.add_parser('prepare-external', help='reconstruct pinned OpenPI or LeRobot source overlay')
    p.add_argument('--model', required=True, choices=['pi05', 'smolvla'])
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.set_defaults(function=external_command)
    p = sub.add_parser('prepare-runtime', help='verify and stage tokenizer/backbone files into an isolated native cache')
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--artifact-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model', required=True, choices=['pi05', 'smolvla'])
    p.set_defaults(function=runtime_command)
    args = parser.parse_args(argv)
    try:
        result = args.function(args)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f'error: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
