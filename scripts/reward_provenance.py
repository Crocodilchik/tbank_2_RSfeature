"""Record reward-label availability boundaries and frozen reward-model provenance."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.frozen_run import frozen_data
from Data_loader.pipeline import load
from SaTP.common import dump,digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    parent=(ROOT/args.run).resolve();c,folder=frozen_data(parent)
    manifest=json.loads((folder/'manifest.json').read_text());stages={}
    for split,cutoff in [('F',manifest['cuts'][0]),('C',manifest['cuts'][1])]:
        d=load(folder,split);mask=d['reward_eligible']&(d['target']>0)&np.isfinite(d['reward'])
        if not mask.any():raise ValueError('No eligible reward labels')
        max_available=int(d['reward_available_at'][mask].max())
        if max_available>=cutoff:raise ValueError('Reward label crosses stage boundary')
        stages[split]=dict(used_labels=int(mask.sum()),all_query_rows=len(mask),max_label_available_at=max_available,exclusive_stage_end=cutoff,passed=True)
    files={p.name:digest(p) for p in sorted((parent/'reward_models').glob('*')) if p.is_file()}
    dump(parent/'reward_provenance.json',dict(stages=stages,guard_source_units=c['data']['reward_guard_units'],RM_fit_population='All F listen anchors with mature supported labels; organic/recommendation events are mixed in fit. C calibration and V/T reward evaluation use recommendation-driven targets.',delay_assumption=manifest['availability_assumption'],files=files,script_sha256=digest(__file__)))
    print(json.dumps(stages),flush=True)
if __name__=='__main__':main()
