import json,time
import polars as pl
from SaTP.common import ROOT,dump,log

def main():
    started=time.monotonic(); path=ROOT/'data/raw/flat/50m/multi_event.parquet'
    lf=pl.scan_parquet(path)
    schema={k:str(v) for k,v in lf.collect_schema().items()}
    profile=lf.group_by('event_type').agg(pl.len().alias('rows'),pl.col('timestamp').min().alias('min_t'),pl.col('timestamp').max().alias('max_t')).collect()
    log('raw_schema',schema=schema);log('raw_counts',counts=profile.to_dicts())
    # Public schema must be confirmed before fixing the event predicate.
    mask=((pl.col('uid').cast(pl.UInt64)*2654435761)%10000)<1000
    df=lf.filter(mask & (pl.col('event_type').cast(pl.String)=='listen')).collect()
    if not df.height: raise ValueError('No listens; inspect schema event mapping')
    cut=int(df['timestamp'].quantile(.5,interpolation='lower'))
    train=df.filter(pl.col('timestamp')<cut);future=df.filter((pl.col('timestamp')>=cut)&(pl.col('is_organic')==0))
    catalog=train.group_by('item_id').len().sort(['len','item_id'],descending=[True,False])
    coverage={}
    for k in [2000,20000,50000,100000,catalog.height]:
        items=catalog.head(k)['item_id'];coverage[str(k)]=float(future.select(pl.col('item_id').is_in(items.implode()).mean()).item())
    result=dict(schema=schema,counts=profile.to_dicts(),cohort_users=df['uid'].n_unique(),cohort_listens=df.height,initial_catalog=catalog.height,cut=cut,coverage=coverage,ratio_quantiles={str(q):float(df['played_ratio_pct'].quantile(q)) for q in [0,.1,.5,.9,1]},missing_ratio=int(df['played_ratio_pct'].null_count()),seconds=time.monotonic()-started)
    dump(ROOT/'reports/data_profile.json',result); log('profile_complete',**result)
if __name__=='__main__':main()
