"""Independent real-data prefix audit using Polars filtering instead of searchsorted."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import polars as pl
from SaTP.common import config,dump,digest
from Data_loader.pipeline import prepare,load
c=config(ROOT/'configs/study.yaml');folder=prepare(c)
m=json.loads((folder/'manifest.json').read_text());items=np.load(folder/'item_ids.npy')
mapping={int(x):i+1 for i,x in enumerate(items)}
raw=pl.scan_parquet(ROOT/'data/raw/flat/50m/multi_event.parquet').filter(pl.col('uid').is_in(m['user_ids'])&(pl.col('event_type').cast(pl.String)=='listen')).select('uid','timestamp','item_id').collect()
by_user={int(f['uid'][0]):f.sort(['timestamp','item_id']) for f in raw.partition_by('uid',maintain_order=True)}
rng=np.random.default_rng(42);violations=[];checked={}
for split in 'FCRVT':
    d=load(folder,split);index=rng.choice(len(d['target']),min(100,len(d['target'])),replace=False)
    for i in index:
        u=int(d['user'][i]);t=int(d['timestamp'][i]);n=d['history'].shape[1]
        prior=by_user[u].filter(pl.col('timestamp')<t).tail(n)
        expected=np.zeros(n,np.int32)
        if prior.height:expected[-prior.height:]=[mapping.get(int(x),len(items)+1) for x in prior['item_id'].to_list()]
        if not np.array_equal(expected,d['history'][i]):violations.append(dict(split=split,row=int(i),reason='history mismatch'))
        original=int(d['original_target'][i])
        if mapping.get(original,0)!=d['target'][i]:violations.append(dict(split=split,row=int(i),reason='target encoding mismatch'))
        if prior.height and int(prior['timestamp'].max())>=t:violations.append(dict(split=split,row=int(i),reason='nonpast timestamp'))
    end=m['cuts']['FCRVT'.index(split)] if split!='T' else int(raw['timestamp'].max())+1
    if end is not None and not np.array_equal(d['reward_eligible'],d['reward_available_at']<end):violations.append(dict(split=split,reason='maturation boundary mismatch'))
    if np.any(d['reward_available_at']<d['timestamp']+c['data']['reward_guard_units']):violations.append(dict(split=split,reason='reward delay too short'))
    checked[split]=len(index)
raw_manifest=json.loads((ROOT/'data/raw/manifest.json').read_text())
checksum_ok=digest(ROOT/'data/raw/flat/50m/multi_event.parquet')==raw_manifest['sha256']
dump(ROOT/'reports/data_audit.json',dict(checked=checked,violations=violations,raw_checksum_matches=checksum_ok,method='independent Polars prefix filtering; sampled real query audit, not exhaustive row recomputation'))
print(json.dumps(dict(checked=checked,violations=len(violations),raw_checksum_matches=checksum_ok)),flush=True)
if violations or not checksum_ok:raise SystemExit(1)
