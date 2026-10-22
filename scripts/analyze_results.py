"""Behavioral slices and common-decoding comparison, without refitting any model."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import polars as pl
from SaTP.common import config,dump,digest
from Data_loader.pipeline import prepare,load
from scripts.frozen_run import frozen_data

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    parent=(ROOT/args.run).resolve()
    if not parent.is_relative_to(ROOT):raise ValueError('Outside project')
    if json.loads((parent/'status.json').read_text())['state']!='completed':raise ValueError('Matrix is incomplete')
    c,folder=frozen_data(parent)
    manifest=json.loads((folder/'manifest.json').read_text());data=load(folder,'T');fit=load(folder,'F')
    ids=np.load(folder/'item_ids.npy');vocab=len(ids);mapping={int(x):i+1 for i,x in enumerate(ids)}
    counts=pl.scan_parquet(ROOT/'data/raw/flat/50m/multi_event.parquet').filter(pl.col('uid').is_in(manifest['user_ids'])&(pl.col('timestamp')<manifest['cuts'][0])&(pl.col('event_type').cast(pl.String)=='listen')).group_by('item_id').len().collect()
    pop=np.zeros(vocab+2,np.int64)
    for row in counts.iter_rows(named=True):
        index=mapping.get(int(row['item_id']),0)
        if index:pop[index]=row['len']
    popularity_order=np.lexsort((np.arange(1,vocab+1),-pop[1:vocab+1]))+1
    popular=set(popularity_order[:max(1,int(vocab*.01))].tolist())
    tail_threshold=float(np.quantile(pop[1:vocab+1],.8))
    target=data['target'];h=data['history'];length=(h!=0).sum(1)
    train_users=set(fit['user'].tolist())
    masks={
        'all':np.ones(len(target),bool),
        'catalog_supported':target>0,
        'catalog_oov':target==0,
        'train_cold_user':np.array([int(u) not in train_users for u in data['user']]),
        'empty_history':length==0,
        'short_history_le10':length<=10,
        'recent_repeat_target':(h==target[:,None]).any(1)&(target>0),
        'not_in_recent_history':~(h==target[:,None]).any(1)&(target>0),
        'train_long_tail':(target>0)&(pop[target]<=tail_threshold),
    }
    rows=[];behavior=[];common=[]
    for seedpath in sorted(parent.glob('seed_*')):
        seed=int(seedpath.name.split('_')[-1]);baseline=pl.read_parquet(seedpath/'B0_predictions.parquet')
        for path in sorted(seedpath.glob('*_predictions.parquet')):
            name=path.name.removesuffix('_predictions.parquet');frame=pl.read_parquet(path)
            if not frame.select('query','user','timestamp','target').equals(baseline.select('query','user','timestamp','target')):raise ValueError('Query registry mismatch')
            recommendations=np.asarray(frame['items'].to_list())
            if (recommendations<1).any() or (recommendations>vocab).any():raise ValueError('Out-of-catalog recommendation')
            if any(len(set(row))!=len(row) for row in recommendations.tolist()):raise ValueError('Duplicate within list')
            for cohort,mask in masks.items():
                part=frame.filter(pl.Series(mask))
                result=dict(seed=seed,solution=name,cohort=cohort,queries=part.height,users=part['user'].n_unique())
                for metric in ['ndcg10','recall10','reward','uncertainty','familiar_fraction','mean_duration_seconds','redundancy']:
                    result[metric]=float(part.group_by('user').agg(pl.col(metric).mean())[metric].mean()) if part.height else None
                rows.append(result)
            flat=recommendations.reshape(-1);exposure=np.bincount(flat,minlength=vocab+1)[1:]
            sorted_exp=np.sort(exposure);gini=float((2*np.arange(1,vocab+1)-vocab-1)@sorted_exp/(vocab*sorted_exp.sum()))
            per_user=frame.sort('timestamp').group_by('user',maintain_order=True).first()['items'].to_list()
            rng=np.random.default_rng(42);jaccard=[]
            for _ in range(min(5000,len(per_user)*10)):
                a,b=rng.choice(len(per_user),2,replace=False);sa,sb=set(per_user[a]),set(per_user[b]);jaccard.append(len(sa&sb)/len(sa|sb))
            head_share=np.array([np.isin(row,list(popular)).mean() for row in recommendations])
            novelty=np.array([(-np.log2(pop[row]/pop.sum())).mean() for row in recommendations])
            auxiliary=frame.with_columns(pl.Series('head_share',head_share),pl.Series('novelty',novelty)).group_by('user').agg(pl.col('head_share').mean(),pl.col('novelty').mean())
            behavior.append(dict(seed=seed,solution=name,distinct_recommended_items=int((exposure>0).sum()),catalog_coverage_fraction=float((exposure>0).mean()),exposure_gini=gini,interuser_jaccard=float(np.mean(jaccard)),top_one_percent_share=float(auxiliary['head_share'].mean()),novelty_bits=float(auxiliary['novelty'].mean())))
            if name.endswith('_nextk'):
                common.append(dict(seed=seed,solution=name.removesuffix('_nextk'),**{metric:float(frame.group_by('user').agg(pl.col(metric).mean())[metric].mean()) for metric in ['ndcg10','reward','recall10']}))
    pl.DataFrame(rows).write_parquet(parent/'cohorts.parquet');pl.DataFrame(behavior).write_parquet(parent/'behavior.parquet');pl.DataFrame(common).write_parquet(parent/'common_decoding.parquet')
    dump(parent/'behavior_manifest.json',dict(script_sha256=digest(__file__),tail_definition='training-count <= item-count 80th percentile; not causal online tail',tail_threshold=tail_threshold,head_definition='top 1% items by full F listen frequency',repeat_definition='within last fixed history window, not lifetime',cohorts={k:int(v.sum()) for k,v in masks.items()},checks='all saved native/common lists: same queries, valid IDs, unique lists'))
    import nbformat as nbf
    report=ROOT/'reports'/f'{parent.name}_comparison.ipynb';notebook=nbf.read(report,as_version=4)
    cells=[nbf.v4.new_markdown_cell('## Behavioral analysis and common decoding\n\nLong tail is defined by initial training frequency. Repeat target means within the available bounded history. Empty cohorts have null metrics. Common Next-K isolates part of the decoding effect.'),nbf.v4.new_code_cell("pl.read_parquet(root/'common_decoding.parquet').group_by('solution').agg(pl.col('ndcg10').mean(),pl.col('reward').mean(),pl.col('recall10').mean()).sort('solution')"),nbf.v4.new_code_cell("pl.read_parquet(root/'behavior.parquet').group_by('solution').agg(pl.col('catalog_coverage_fraction').mean(),pl.col('exposure_gini').mean(),pl.col('interuser_jaccard').mean(),pl.col('top_one_percent_share').mean(),pl.col('novelty_bits').mean()).sort('solution')"),nbf.v4.new_code_cell("pl.read_parquet(root/'cohorts.parquet').filter(pl.col('seed')==42).select('solution','cohort','users','ndcg10','reward')")]
    notebook.cells.extend(cells)
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[str(ROOT/'.venv/Scripts/python.exe'),'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(notebook,timeout=120,km=km,resources={'metadata':{'path':str(ROOT)}}).execute(cleanup_kc=True);nbf.write(notebook,report)
    print('BEHAVIOR ANALYSIS COMPLETE',flush=True)
if __name__=='__main__':main()
