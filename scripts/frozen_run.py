"""Resolve the immutable data/config snapshot belonging to an executed run."""
import json,hashlib,zipfile
from pathlib import Path
from SaTP.common import ROOT,digest

def frozen_data(parent):
    parent=Path(parent).resolve()
    if not parent.is_relative_to(ROOT):raise ValueError('Run is outside project')
    env=json.loads((parent/'environment.json').read_text());c=env['config']
    source=ROOT/'artifacts/code'/f"{env['code_sha256']}.zip"
    if not source.exists():raise ValueError('Missing archived training source')
    with zipfile.ZipFile(source) as archive:
        pipeline_hash=hashlib.sha256(archive.read('Data_loader/pipeline.py')).hexdigest()[:8]
    folder=ROOT/'data/processed'/(c['config_hash'][:12]+'_'+pipeline_hash)
    if not (folder/'manifest.json').exists():raise ValueError('Frozen processed dataset is unavailable')
    if digest(folder/'manifest.json')!=env['data_manifest_sha256']:raise ValueError('Frozen data manifest changed')
    return c,folder
