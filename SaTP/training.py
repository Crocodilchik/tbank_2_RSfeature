import time, copy
import numpy as np
import torch
from torch.nn import functional as F
from GPT2_models.backbone import GPT2Rec
from Reward_models.models import SequenceReward,RewardEnsemble,tabular_features
from SaTP.common import dump,seed_all,log
from Reward_models.reward_functions import pessimistic_reward

class BudgetExceeded(RuntimeError): pass
class Budget:
    def __init__(self,hours): self.start=time.monotonic(); self.limit=hours*3600
    def check(self):
        if time.monotonic()-self.start>self.limit: raise BudgetExceeded("Configured execution budget exhausted")

def model(c,size,device): return GPT2Rec(size,c["data"]["history_length"],**c["model"]).to(device)
def save(path,m,optimizer=None,**extra):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".writing")
    torch.save(dict(model=m.state_dict(),optimizer=optimizer.state_dict() if optimizer else None,numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],**extra),temporary)
    temporary.replace(path)
def load_model(path,m):
    ck=torch.load(path,map_location="cpu",weights_only=False); m.load_state_dict(ck["model"]); return ck

def supervised(c,m,data,path,device,budget):
    if path.exists(): load_model(path,m); return
    opt=torch.optim.AdamW(m.parameters(),lr=c["training"]["lr"])
    ok=np.flatnonzero(data["target"]>0); bs=c["training"]["batch_size"]
    if not len(ok): raise ValueError("No supported training targets")
    started=time.monotonic()
    start_epoch=0
    partial=path.with_suffix(".partial.pt")
    if partial.exists():
        ck=load_model(partial,m); opt.load_state_dict(ck["optimizer"])
        start_epoch=ck["epoch"]+1; ok=ck["sample_order"]
        np.random.set_state(ck["numpy_rng"]); torch.set_rng_state(ck["torch_rng"])
        if device=="cuda" and ck["cuda_rng"]: torch.cuda.set_rng_state_all(ck["cuda_rng"])
    for epoch in range(start_epoch,c["training"]["epochs"]):
        m.train(); losses=[]
        for start in range(0,len(ok),bs):
            budget.check(); ix=ok[start:start+bs]
            h=torch.as_tensor(data["history"][ix],device=device,dtype=torch.long); y=torch.as_tensor(data["target"][ix]-1,device=device)
            opt.zero_grad(set_to_none=True); loss=F.cross_entropy(m(h),y); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); losses.append(loss.item())
        np.random.shuffle(ok); log("sft_epoch",epoch=epoch,loss=float(np.mean(losses)),seconds=time.monotonic()-started)
        if epoch+1<c["training"]["epochs"]: save(path.with_suffix(".partial.pt"),m,opt,epoch=epoch,sample_order=ok)
    m.eval(); save(path,m,completed=True)

def fit_rewards(c,data,calibration,folder,size,meta,device,budget):
    folder.mkdir(parents=True,exist_ok=True); models=[]; bs=c["training"]["batch_size"]
    ok=np.flatnonzero((data["target"]>0)&np.isfinite(data["reward"])&data["reward_eligible"])
    unique=np.unique(data["user"][ok])
    for member in range(c["training"]["rm_members"]):
        ok=np.flatnonzero((data["target"]>0)&np.isfinite(data["reward"])&data["reward_eligible"])
        seed_all(1000+member); m=SequenceReward(size).to(device); path=folder/f"rm_{member}.pt"
        if path.exists(): load_model(path,m)
        else:
            opt=torch.optim.AdamW(m.parameters(),lr=.001)
            sampled=np.random.choice(unique,size=len(unique),replace=True)
            weights={u:int(np.sum(sampled==u)) for u in unique}
            for epoch in range(c["training"]["rm_epochs"]):
                m.train(); losses=[]
                for start in range(0,len(ok),bs):
                    budget.check(); ix=ok[start:start+bs]
                    h=torch.as_tensor(data["history"][ix],device=device,dtype=torch.long); a=torch.as_tensor(data["target"][ix],device=device)
                    d=torch.as_tensor(data["duration"][ix],device=device); y=torch.as_tensor(data["reward"][ix],device=device)
                    w=torch.tensor([weights[u] for u in data["user"][ix]],device=device,dtype=torch.float32)
                    opt.zero_grad(set_to_none=True); loss=((m(h,a,d)-y)**2*w).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); losses.append(loss.item())
                np.random.shuffle(ok); log("rm_epoch",member=member,epoch=epoch,loss=float(np.mean(losses)))
            m.eval(); save(path,m,completed=True)
        models.append(m)
    ensemble=RewardEnsemble(models,meta["duration"],meta["first"],device)
    # Scalar bias calibration on C; C has no policy fitting.
    cal_path=folder/"calibration.json"
    if not cal_path.exists():
        values,targets=reward_predictions(ensemble,calibration,device,bs)
        offset=float(np.mean(targets-values)) if len(targets) else 0.
        dump(cal_path,dict(offset=offset,rows=len(targets),raw_mae=float(np.mean(abs(targets-values))) if len(targets) else None))
    import json
    ensemble.offset=json.loads(cal_path.read_text())["offset"]
    return ensemble

def reward_predictions(ensemble,data,device,bs):
    predictions=[]; targets=[]
    for start in range(0,len(data["target"]),bs):
        ix=np.arange(start,min(start+bs,len(data["target"]))); ix=ix[(data["target"][ix]>0)&np.isfinite(data["reward"][ix])&data["reward_eligible"][ix]]
        if not len(ix): continue
        h=torch.as_tensor(data["history"][ix],device=device,dtype=torch.long); a=torch.as_tensor(data["target"][ix],device=device); t=torch.as_tensor(data["timestamp"][ix],device=device)
        mean,_=ensemble.score(ensemble.encode(h),a,t)
        predictions.extend(mean[:,0].cpu().numpy()); targets.extend(data["reward"][ix])
    return np.asarray(predictions),np.asarray(targets)

def rl_train(c,base,ensemble,data,path,device,budget,solution,beta=None,pessimism=None,diversity=None):
    m=copy.deepcopy(base)
    for parameter in m.parameters(): parameter.requires_grad_(True)
    if path.exists(): load_model(path,m); return m
    m.eval(); base.eval()
    for p in base.parameters(): p.requires_grad_(False); p.grad=None
    opt=torch.optim.AdamW(m.parameters(),lr=c["training"]["rl_lr"])
    beta=c["training"]["beta"] if beta is None else beta
    lam=(c["training"]["pessimism"] if pessimism is None else pessimism) if solution in ["A3","A4"] else 0.
    eta=(c["training"]["diversity"] if diversity is None else diversity) if solution=="A4" else 0.
    horizon=c["training"]["horizon"] if solution=="A4" else 1
    bs=c["training"]["batch_size"]; trace=[]; running=0.
    start_step=0
    partial=path.with_suffix(".partial.pt")
    if partial.exists():
        ck=load_model(partial,m); opt.load_state_dict(ck["optimizer"])
        start_step=ck["step"]+1; running=ck["running"]
        np.random.set_state(ck["numpy_rng"]); torch.set_rng_state(ck["torch_rng"])
        if device=="cuda" and ck["cuda_rng"]: torch.cuda.set_rng_state_all(ck["cuda_rng"])
    # eval mode intentionally disables dropout in the behavior policy likelihood.
    for step in range(start_step,c["training"]["rl_steps"]):
        budget.check(); ix=np.random.randint(0,len(data["target"]),size=bs)
        h=torch.as_tensor(data["history"][ix],device=device,dtype=torch.long); ts=torch.as_tensor(data["timestamp"][ix],device=device)
        states=ensemble.encode(h); context=h; chosen=[]; logps=[]; kls=[]; rewards=[]
        for position in range(horizon):
            logits=m(context)
            with torch.no_grad(): ref=base(context)
            mask=torch.ones_like(logits,dtype=torch.bool)
            if chosen: mask.scatter_(1,torch.stack(chosen,dim=1)-1,False)
            logits=logits.masked_fill(~mask,-1e9); ref=ref.masked_fill(~mask,-1e9)
            lp=logits.log_softmax(-1); rp=ref.log_softmax(-1)
            action=torch.multinomial(lp.exp(),1).squeeze(1)+1
            mean,std=ensemble.score(states,action,ts)
            r=pessimistic_reward(mean[:,0],std[:,0],ensemble.offset,lam)
            if chosen:
                with torch.no_grad():
                    emb=F.normalize(base.item.weight,dim=-1)
                    similarities=(emb[action][:,None,:]*emb[torch.stack(chosen,1)]).sum(-1).clamp(min=0)
                    r=r-eta*similarities.mean(1)
            chosen.append(action); logps.append(lp.gather(1,(action-1)[:,None]).squeeze(1)); kls.append((lp.exp()*(lp-rp)).sum(1)); rewards.append(r)
            context=torch.cat([context,action[:,None]],dim=1)
        r=torch.stack(rewards,1); logp=torch.stack(logps,1); kl=torch.stack(kls,1)
        returns=torch.flip(torch.cumsum(torch.flip(r,[1]),1),[1])/horizon
        baseline=(returns.sum(0,keepdim=True)-returns)/max(1,bs-1) if bs>1 else torch.full_like(returns,running)
        advantage=returns-baseline
        loss=-(advantage.detach()*logp).sum(1).mean()+beta*kl.mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        running=.9*running+.1*returns.mean().item()
        if step%10==0 or step==c["training"]["rl_steps"]-1:
            row=dict(step=step,reward=r.mean().item(),kl=kl.mean().item(),loss=loss.item()); trace.append(row); log("rl",solution=solution,**row)
        if (step+1)%c["training"].get("checkpoint_every",200)==0 and step+1<c["training"]["rl_steps"]: save(path.with_suffix(".partial.pt"),m,opt,step=step,running=running)
    dump(path.with_suffix(".trace.json"),trace); save(path,m,completed=True); return m
