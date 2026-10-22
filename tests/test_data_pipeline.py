import numpy as np
import polars as pl
from Data_loader import pipeline
from SaTP.common import config,ROOT

def test_preprocessing_keeps_tied_events_out_of_history(tmp_path,monkeypatch):
    c=config(ROOT/'configs/smoke.yaml')
    c['data']['user_limit']=10000
    c['data']['max_train_per_user']=1000
    c['data']['max_eval_per_user']=1000
    times=np.repeat(np.arange(1,101),2)
    frame=pl.DataFrame(dict(uid=np.zeros(len(times),dtype=np.int64),item_id=np.tile([1,2],100),timestamp=times,is_organic=np.zeros(len(times),dtype=np.int8),event_type=['listen']*len(times),played_ratio_pct=np.full(len(times),75),track_length_seconds=np.full(len(times),200)))
    raw=tmp_path/'fixture.parquet';frame.write_parquet(raw)
    monkeypatch.setattr(pipeline,'download',lambda c:(raw,{'sha256':'fixture'}))
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    folder=pipeline.prepare(c)
    for name in 'FCRVT':
        data=pipeline.load(folder,name)
        # Each earlier timestamp contains one of each ID: history must have balanced item counts.
        for h in data['history']:
            assert np.sum(h==1)==np.sum(h==2)
    assert pipeline.load(folder,'F')['history'].shape[1]==32
