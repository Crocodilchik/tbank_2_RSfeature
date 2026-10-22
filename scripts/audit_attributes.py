"""Descriptive attribute consistency checks; never change fitted features."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import polars as pl
from scripts.frozen_run import frozen_data
from SaTP.common import dump,digest
parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
parent=(ROOT/args.run).resolve();c,folder=frozen_data(parent)
m=json.loads((folder/'manifest.json').read_text())
cohort=pl.scan_parquet(ROOT/'data/raw/flat/50m/multi_event.parquet').filter(pl.col('uid').is_in(m['user_ids'])&(pl.col('event_type').cast(pl.String)=='listen'))
fit=cohort.filter(pl.col('timestamp')<m['cuts'][0])
result={}
for name,lf in [('initial_F',fit),('all_cohort_descriptive_only',cohort)]:
    by_item=lf.group_by('item_id').agg(pl.col('track_length_seconds').n_unique().alias('distinct_lengths'))
    counts=by_item.select(pl.len().alias('items'),(pl.col('distinct_lengths')>1).sum().alias('items_with_conflicting_lengths')).collect().to_dicts()[0]
    stats=lf.select(pl.len().alias('rows'),(pl.col('track_length_seconds')==0).sum().alias('zero_length_rows'),pl.col('track_length_seconds').min().alias('min_length'),pl.col('track_length_seconds').max().alias('max_length'),(pl.col('timestamp')%5!=0).sum().alias('timestamps_not_multiple_of_5')).collect().to_dicts()[0]
    result[name]=dict(**counts,**stats)
result['feature_policy']='Use earliest observed F duration only when its timestamp is strictly before query time. No backfill from later attributes. Full-cohort checks are reporting only.'
result['script_sha256']=digest(__file__)
dump(parent/'attribute_audit.json',result);print(json.dumps(result))
