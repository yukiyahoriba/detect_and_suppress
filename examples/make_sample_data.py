"""Generate small synthetic inputs for the CLI usage examples; no research measurements."""
import argparse
import json
from pathlib import Path
import numpy as np


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(17)
    np.savez(output / 'sae.npz', encoder=rng.normal(size=(6,4)).astype('f4'),
             decoder=rng.normal(size=(4,6)).astype('f4'), pre_bias=np.zeros(4,'f4'),
             train_config=np.asarray(json.dumps({'top_k':2})))
    episodes=[]
    for episode_id in range(2):
        name=f'episode{episode_id}.npz'
        patch=np.tile([0.,1.],12)
        activation=rng.normal(size=(24,4)).astype('f4'); activation[:,0]+=patch*2
        np.savez(output/name, activations=activation, patch_signal=patch,
                 policy_forward_signal=np.tile([1.,1.,0.],8))
        episodes.append(dict(path=name,expert='pg',layer=5,episode_id=episode_id,task_id=0))
    selection=dict(schema_version=1,sae='sae.npz',episodes=episodes,threshold=.6,criterion='mean',top_n=2)
    representation=dict(model='sample',family='residual',layer=0,capture_point='synthetic',pooling='none',hidden_dim=4)
    data=dict(layout='arrays',representation=representation,split_unit='episode')
    for idx,split in enumerate(('train','dev','test')):
        clean=rng.normal(-.2,1,size=(40,4)).astype('f4')
        attack=rng.normal(.2,1,size=(40,4)).astype('f4')
        np.savez(output/f'{split}.npz',clean=clean,attack=attack,
                 clean_episode_ids=np.full(40,idx),attack_episode_ids=np.full(40,idx))
        data[split]=f'{split}.npz'
    probe=dict(schema_version=1,data=data,training=dict(profile='smolvla',c_values=[.01,.1],max_iter=30))
    for name,value in [('selection.json',selection),('probe.json',probe)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    generate(parser.parse_args().output)
