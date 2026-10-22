import time
from huggingface_hub import HfApi, hf_hub_download
from SaTP.common import ROOT, dump, digest, log

def download(c):
    folder = ROOT/"data/raw"; folder.mkdir(parents=True, exist_ok=True)
    manifest = folder/"manifest.json"
    if manifest.exists():
        import json
        m=json.loads(manifest.read_text()); p=folder/c["data"]["file"]
        if p.exists() and digest(p)==m["sha256"]: return p,m
        raise ValueError("Raw checksum mismatch; refusing silent replacement")
    revision=HfApi().dataset_info(c["data"]["repo"]).sha
    log("download_start", revision=revision)
    path=hf_hub_download(repo_id=c["data"]["repo"], filename=c["data"]["file"], repo_type="dataset", revision=revision, local_dir=folder)
    m=dict(repo=c["data"]["repo"], file=c["data"]["file"], revision=revision, sha256=digest(path), bytes=__import__("os").path.getsize(path))
    dump(manifest,m); log("download_complete", **m)
    return path,m
