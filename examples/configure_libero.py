"""Write LIBERO's native path config for an explicit pinned checkout."""
import argparse
from pathlib import Path
import json


def configure(source, output):
    source=Path(source).resolve()/'libero/libero'
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    for name in ('assets','bddl_files','init_files'):
        if not (source/name).is_dir(): raise FileNotFoundError(source/name)
    values={'benchmark_root':str(source),'bddl_files':str(source/'bddl_files'),
            'init_states':str(source/'init_files'),'assets':str(source/'assets'),
            'datasets':str(output/'datasets')}
    output.mkdir(parents=True)
    # JSON string quoting is also valid YAML for paths with spaces/non-ASCII.
    (output/'config.yaml').write_text(''.join(f'{k}: {json.dumps(v)}\n' for k,v in values.items()))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path); p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(configure(a.source,a.output))
