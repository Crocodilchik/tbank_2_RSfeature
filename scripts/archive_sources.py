from pathlib import Path
import sys,zipfile,json,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from SaTP.common import code_fingerprint,dump
fingerprint=code_fingerprint();target=root/'artifacts/code'/f'{fingerprint}.zip';target.parent.mkdir(parents=True,exist_ok=True)
paths=[]
for directory in ['Data_loader','GPT2_models','Reward_models','SaTP','Comparison','tests','configs','docs','scripts']:
    paths.extend(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
paths.extend(root/p for p in ['README.md','pyproject.toml','requirements.lock.txt','project_env.ps1','run.ps1','install.ps1'])
if not target.exists():
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:archive.write(path,path.relative_to(root))
with zipfile.ZipFile(target) as archive:
    archived_hashes={name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
dump(root/'artifacts/code'/f'{fingerprint}.json',dict(code_sha256=fingerprint,archive=str(target.relative_to(root)),files=archived_hashes))
print('Archived source version',fingerprint,flush=True)
