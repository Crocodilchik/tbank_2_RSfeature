"""Recompute ranking metrics from saved recommendations; verify completed matrix."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import polars as pl
import nbformat
from SaTP.common import dump
from SaTP.evaluation import list_metrics,summarize

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    parent=(ROOT/args.run).resolve()
    if not parent.is_relative_to(ROOT):raise ValueError('Outside project')
    status=json.loads((parent/'status.json').read_text())
    if status['state']!='completed':raise ValueError('Matrix is not completed')
    expected=['B0','A1','A2','A3','A4','control_sft'];checks=[];registry=None
    seeds=status['seeds']
    for seed in seeds:
        out=parent/f'seed_{seed}';metrics=json.loads((out/'metrics.json').read_text())
        for name in expected:
            frame=pl.read_parquet(out/f'{name}_predictions.parquet')
            keys=frame.select('query','user','timestamp','target','original_target')
            if registry is None:registry=keys
            if not keys.equals(registry):raise ValueError('Different query registry')
            if frame['query'].n_unique()!=frame.height:raise ValueError('Duplicated query')
            arrays=np.asarray(frame['items'].to_list());target=frame['target'].to_numpy()
            # Independent rank reconstruction rather than calling the evaluator's metric function.
            hits=arrays==target[:,None]
            ranks=np.where(hits.any(axis=1),hits.argmax(axis=1)+1,1_000_000)
            reference=dict(ndcg10=np.where(ranks<=10,1/np.log2(ranks+1),0.),recall5=(ranks<=5).astype(float),recall10=(ranks<=10).astype(float))
            for metric,values in reference.items():
                if not np.allclose(values,frame[metric].to_numpy(),rtol=1e-7,atol=1e-8):raise ValueError(f'Wrong {name}/{metric}')
            users,inverse=np.unique(frame['user'].to_numpy(),return_inverse=True)
            user_counts=np.bincount(inverse)
            independent={**reference,'reward':frame['reward'].to_numpy(),'coverage':(target>0).astype(float)}
            result={key:float(np.mean(np.bincount(inverse,weights=values)/user_counts)) for key,values in independent.items()}
            for metric in ['ndcg10','recall5','recall10','reward','coverage']:
                if not np.isclose(result[metric],metrics[name][metric],rtol=1e-6,atol=1e-8):raise ValueError('Summary mismatch')
            checks.append(dict(seed=seed,solution=name,queries=frame.height,passed=True))
    report=ROOT/'reports'/f'{parent.name}_comparison.ipynb';notebook=nbformat.read(report,as_version=4)
    code=[cell for cell in notebook.cells if cell.cell_type=='code']
    errors=[output for cell in code for output in cell.outputs if output.output_type=='error']
    if errors or any(cell.execution_count is None for cell in code):raise ValueError('Notebook incomplete or has errors')
    dump(parent/'verification.json',dict(completed_cells=len(checks),seeds=seeds,expected_primary_cells=5*len(seeds),control_cells=len(seeds),queries_per_cell=registry.height,all_ranking_metrics_recomputed=True,notebook_code_cells=len(code),notebook_errors=0,checks=checks))
    print(json.dumps(dict(cells=len(checks),seeds=seeds,queries=registry.height,notebook_errors=0)),flush=True)
if __name__=='__main__':main()
