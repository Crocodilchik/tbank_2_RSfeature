"""Snapshot final source and analysis helpers without data, environments or weights."""
import hashlib,json,zipfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from SaTP.common import dump,code_fingerprint,digest

files=[]
for folder in ['Data_loader','GPT2_models','Reward_models','SaTP','Comparison','scripts','tests','configs','docs']:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files.extend(p for p in ROOT.iterdir() if p.is_file() and (p.suffix in ['.md','.toml','.ps1'] or p.name in ['requirements.lock.txt','.gitignore']))
files=sorted(set(files));hashes={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files}
signature=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
target=ROOT/'artifacts'/f'delivery_{signature[:12]}.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in hashes:z.write(ROOT/name,name)
with zipfile.ZipFile(target) as z:
    for name,expected in hashes.items():
        if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise ValueError('Archive mismatch')
dump(target.with_suffix('.json'),dict(sha256=digest(target),core_code_sha256=code_fingerprint(),files=hashes,excludes='Dataset, weights, environments, caches, generated reports'))
print(target)
