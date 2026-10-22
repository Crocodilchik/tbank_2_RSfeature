import hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]

def inside(path):
    p = Path(path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f"Path outside project: {p}")
    return p

def dump(path, obj):
    p = inside(path); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(p)

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""): h.update(b)
    return h.hexdigest()

def config(path):
    c = yaml.safe_load(inside(path).read_text(encoding="utf-8"))
    q=c["data"]["split_quantiles"]
    if len(q)!=4 or not all(0<a<b<1 for a,b in zip(q,q[1:])):raise ValueError("Require four strictly increasing temporal quantiles")
    if not 0<c["data"]["user_limit"]<=10000:raise ValueError("Invalid user cohort threshold")
    if c["data"]["catalog_size"]<10:raise ValueError("Catalog must contain at least ten items")
    if c["model"]["hidden"]%c["model"]["heads"]:raise ValueError("Hidden dimension must be divisible by attention heads")
    if len(set(c["seeds"]))!=len(c["seeds"]) or not c["seeds"]:raise ValueError("Training seeds must be unique and nonempty")
    if not 0<c["training"]["max_hours"]<=c["training"]["hard_max_hours"]<20:raise ValueError("Execution budget must be below twenty hours")
    if c["data"]["reward_guard_units"]<0:raise ValueError("Negative label-maturation delay")
    c["config_hash"] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
    return c

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2)//2)))
    torch.backends.cudnn.benchmark = False

LOG_PATH=None

def set_log_path(path):
    global LOG_PATH
    LOG_PATH=inside(path); LOG_PATH.parent.mkdir(parents=True,exist_ok=True)

def log(event, **kwargs):
    line=json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"),event=event,**kwargs))
    print(line,flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a",encoding="utf-8") as handle:handle.write(line+"\n")


def code_fingerprint():
    h=hashlib.sha256()
    for directory in ["Data_loader","GPT2_models","Reward_models","SaTP","Comparison"]:
        for path in sorted((ROOT/directory).glob("*.py")):
            h.update(str(path.relative_to(ROOT)).encode()); h.update(path.read_bytes())
    return h.hexdigest()
