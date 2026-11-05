"""Stage explicitly verified auxiliary files into native model cache layouts."""
from pathlib import Path
import shutil

from .evaluation import Artifact


def prepare_runtime_assets(manifest: dict, artifact_root, output, *, model: str) -> dict[str, str]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if manifest['schema_version'] != 1 or model not in ('pi05','smolvla'):
        raise ValueError('unsupported manifest/model')
    items = [a for a in manifest['artifacts'] if a['model']==model and a['role']=='runtime']
    if not items:
        raise ValueError('manifest has no runtime assets for this model')
    sources=[]
    for item in items:
        path=(Path(artifact_root)/item['path']).resolve()
        if Artifact.identify(path).sha256!=item['sha256']:
            raise ValueError(f"runtime asset checksum mismatch: {item['id']}")
        if path==output or path in output.parents:
            raise ValueError('cache output must be outside runtime input')
        sources.append((item,path))
    try:
        output.mkdir(parents=True)
        for item,source in sources:
            if model=='pi05':
                target=output/'openpi/big_vision/paligemma_tokenizer.model'
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
            else:
                upstream=item['upstream']
                repo=output/'hf/hub'/('models--'+upstream['repo_id'].replace('/','--'))
                target=repo/'snapshots'/upstream['revision']
                shutil.copytree(source,target)
                (repo/'refs').mkdir()
                (repo/'refs/main').write_text(upstream['revision'])
        return ({'OPENPI_DATA_HOME':str(output/'openpi')} if model=='pi05' else
                {'HF_HOME':str(output/'hf'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
    except BaseException:
        shutil.rmtree(output)
        raise
