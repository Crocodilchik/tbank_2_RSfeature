"""Preregistered supplementary controls. Run only after the primary matrix completes."""
import sys,copy,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from SaTP.common import config,dump,seed_all,digest,log,set_log_path
from SaTP.run import trial_checkpoint
from SaTP.training import model,load_model,fit_rewards,rl_train,supervised,save,Budget
from SaTP.evaluation import evaluate,summarize
from Data_loader.pipeline import prepare,load
from scripts.frozen_run import frozen_data

class DurationReward:
    def __init__(self,original):
        self.original=original;self.models=original.models;self.duration=original.duration;self.first=original.first;self.offset=0.
    def encode(self,h):return self.original.encode(h)
    def score(self,states,items,timestamps):
        mean,std=self.original.score(states,items,timestamps)
        if items.ndim==1:items=items[:,None]
        seconds=torch.expm1(self.duration[items]*10).clamp(max=600)/600
        available=self.first[items]<timestamps[:,None]
        multiplier=seconds*available
        return (mean+self.original.offset).clamp(0,1)*multiplier,std*multiplier

class PopularityPolicy(nn.Module):
    def __init__(self,counts,device):
        super().__init__();self.catalog_size=len(counts)-2
        self.register_buffer("scores",torch.log1p(torch.as_tensor(counts[1:self.catalog_size+1],device=device,dtype=torch.float32)))
    def forward(self,h):return self.scores[None,:].expand(h.shape[0],-1)

def auxiliary_train(c,base,data,path,device,budget):
    policy=copy.deepcopy(base)
    for p in policy.parameters():p.requires_grad_(True)
    if path.exists():load_model(path,policy);return policy
    head=nn.Linear(c['model']['hidden'],1).to(device)
    optimizer=torch.optim.AdamW(list(policy.parameters())+list(head.parameters()),lr=c['training']['lr'])
    valid=np.flatnonzero(data['target']>0);bs=c['training']['batch_size']
    for epoch in range(c['training']['epochs']):
        policy.train(); np.random.shuffle(valid)
        for start in range(0,len(valid),bs):
            budget.check();ix=valid[start:start+bs]
            hist=data['history'][ix];h=torch.as_tensor(hist,device=device,dtype=torch.long)
            y=torch.as_tensor(data['target'][ix]-1,device=device)
            # Number of distinct historical vocabulary tokens, not length trivially exposed by padding.
            counts=np.array([len(np.unique(row[row>0])) for row in hist])
            aux=torch.as_tensor(np.log1p(counts)/np.log1p(hist.shape[1]),device=device,dtype=torch.float32)
            state=policy.encode(h)
            logits=F.linear(state,policy.item.weight[1:policy.catalog_size+1])
            loss=F.cross_entropy(logits,y)+.1*F.mse_loss(head(state).squeeze(1),aux)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);optimizer.step()
        log('auxiliary_sft',epoch=epoch,loss=loss.item())
    policy.eval();save(path,policy,optimizer,completed=True,auxiliary_head=head.state_dict());return policy

def append_report(parent):
    import nbformat as nbf
    report=ROOT/'reports'/f'{parent.name}_comparison.ipynb';notebook=nbf.read(report,as_version=4)
    notebook.cells.append(nbf.v4.new_markdown_cell('## Supplementary controls\n\nOne training seed (42). All variants evaluated with common Next-K. These controls do not establish robustness across training seeds.'))
    notebook.cells.append(nbf.v4.new_code_cell("pl.read_parquet(root/'ablations_seed42'/'summary.parquet')"))
    notebook.cells.append(nbf.v4.new_code_cell("json.loads((root/'ablations_seed42'/'comparisons.json').read_text())"))
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[str(ROOT/'.venv/Scripts/python.exe'),'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(notebook,timeout=120,km=km,resources={'metadata':{'path':str(ROOT)}}).execute(cleanup_kc=True)
    nbf.write(notebook,report)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    parent=(ROOT/args.run).resolve()
    if not parent.is_relative_to(ROOT):raise ValueError('Outside project')
    status=json.loads((parent/'status.json').read_text())
    if status['state']!='completed':raise ValueError('Primary matrix must complete before supplementary evaluation')
    if args.report_only:
        if json.loads((parent/'ablations_seed42/manifest.json').read_text())['state']!='completed':raise ValueError('Controls incomplete')
        append_report(parent);return
    c,folder=frozen_data(parent);meta=dict(np.load(folder/'metadata.npz'))
    blocks={x:load(folder,x) for x in 'FCRT'};size=len(np.load(folder/'item_ids.npy'))
    device='cuda' if torch.cuda.is_available() else 'cpu';budget=Budget(3)
    out=parent/'ablations_seed42';out.mkdir(exist_ok=True)
    set_log_path(out/'events.jsonl')
    dump(out/'manifest.json',dict(state='running',seed=42,script_sha256=digest(__file__),parent_status=status,scope='preregistered controls, one training seed; exploratory uncertainty'))
    ens=fit_rewards(c,blocks['F'],blocks['C'],parent/'reward_models',size,meta,device,budget)
    base=model(c,size,device);load_model(parent/'seed_42/B0.pt',base);base.eval()
    selected=json.loads((parent/'selection.json').read_text())
    rows=[]
    def record(name,policy,settings):
        started=time.monotonic()
        # Compare training interventions using the same Next-K generation for all.
        frame=evaluate(c,policy,ens,blocks['T'],'A4',device,sequential=True)
        frame.write_parquet(out/f'{name}_predictions.parquet')
        result=dict(name=name,settings=json.dumps(settings),**summarize(frame),evaluation_seconds=time.monotonic()-started)
        rows.append(result);pl.DataFrame(rows).write_parquet(out/'summary.parquet');log('ablation_complete',**result)
    record('B0_nextk',base,{})
    sampled_counts=np.bincount(blocks['F']['target'],minlength=size+2)
    record('popularity_same_anchors',PopularityPolicy(sampled_counts,device),dict(source='same F targets as B0'))
    metadata=json.loads((folder/'manifest.json').read_text());item_ids=np.load(folder/'item_ids.npy');mapping={int(x):i+1 for i,x in enumerate(item_ids)}
    raw_counts=pl.scan_parquet(ROOT/'data/raw/flat/50m/multi_event.parquet').filter(pl.col('uid').is_in(metadata['user_ids'])&(pl.col('timestamp')<metadata['cuts'][0])&(pl.col('event_type').cast(pl.String)=='listen')).group_by('item_id').len().collect()
    full_counts=np.zeros(size+2,np.int64)
    for row in raw_counts.iter_rows(named=True):
        index=mapping.get(int(row['item_id']),0)
        if index:full_counts[index]=row['len']
    record('popularity_full_F',PopularityPolicy(full_counts,device),dict(source='all F listens; supplementary information-volume diagnostic'))
    control=model(c,size,device);load_model(parent/'seed_42/control_sft.pt',control);record('continued_sft',control,{})
    a3=model(c,size,device);load_model(parent/'seed_42/A3.pt',a3);record('A3_nextk',a3,selected['A3']['params'])
    a4=model(c,size,device);load_model(parent/'seed_42/A4.pt',a4);record('A4_H5',a4,selected['A4']['params'])
    del control,a3,a4
    for coefficient in [.2,.5]:
        params=dict(selected['A3']['params'],pessimism=coefficient)
        checkpoint=trial_checkpoint(parent/'seed_42','A3',params)
        if checkpoint.exists():
            policy=model(c,size,device);load_model(checkpoint,policy)
            record(f'A3_pessimism_{coefficient}',policy,params);del policy
    for horizon in [1,3]:
        local=copy.deepcopy(c);local['training']['horizon']=horizon
        params=dict(selected['A4']['params'])
        if horizon==1:params['diversity']=0.
        seed_all(42);started=time.monotonic()
        policy=rl_train(local,base,ens,blocks['R'],out/f'A4_H{horizon}.pt',device,budget,'A4',**params)
        record(f'A4_H{horizon}',policy,dict(horizon=horizon,training_seconds=time.monotonic()-started,**params));del policy
    # Main A4 trial0 is the predeclared eta=0 trial; its other settings are shared.
    zero=model(c,size,device);load_model(trial_checkpoint(parent/'seed_42','A4',dict(selected['A4']['params'],diversity=0.)),zero);record('A4_eta0',zero,dict(selected['A4']['params'],diversity=0.));del zero
    seed_all(42);positive={k:v[np.isfinite(blocks['R']['reward'])&(blocks['R']['reward']>=.8)&blocks['R']['reward_eligible']] for k,v in blocks['R'].items()}
    pos=copy.deepcopy(base)
    for p in pos.parameters():p.requires_grad_(True)
    started=time.monotonic();supervised(c,pos,positive,out/'positive_sft.pt',device,budget)
    record('positive_sft',pos,dict(rows=len(positive['target']),threshold=.8,training_seconds=time.monotonic()-started));del pos
    seed_all(42);started=time.monotonic();aux=auxiliary_train(c,base,blocks['R'],out/'auxiliary_sft.pt',device,budget)
    record('auxiliary_sft',aux,dict(auxiliary='distinct historical item count',weight=.1,training_seconds=time.monotonic()-started));del aux
    seed_all(42);started=time.monotonic();params=selected['A3']['params']
    duration=rl_train(c,base,DurationReward(ens),blocks['R'],out/'duration_weighted.pt',device,budget,'A3',**params)
    record('duration_weighted',duration,dict(reward='ratio*min(duration,600)/600',training_seconds=time.monotonic()-started,**params))
    from Comparison.report import paired
    baseline=pl.read_parquet(out/'B0_nextk_predictions.parquet')
    comparisons={name:paired(baseline,pl.read_parquet(out/f'{name}_predictions.parquet')) for name in [r['name'] for r in rows] if name!='B0_nextk'}
    dump(out/'comparisons.json',comparisons)
    dump(out/'manifest.json',dict(state='completed',seed=42,script_sha256=digest(__file__),scope='preregistered controls, one training seed; do not infer training-seed robustness',parent=parent.name,cells=len(rows)))
    append_report(parent)
    print('ABLATIONS COMPLETE',flush=True)
if __name__=='__main__':main()
