import argparse,time,json
import numpy as np
import torch
from SaTP.common import ROOT,config,dump,seed_all,log
from GPT2_models.backbone import GPT2Rec
from Reward_models.models import SequenceReward,RewardEnsemble
from torch.nn import functional as F

def main():
    seed_all(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cpu': raise RuntimeError('CUDA is required for the planned local GPU study')
    results=[]
    for catalog in [2000,20000,100000]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m=GPT2Rec(catalog,64,hidden=128,layers=2,heads=4,dropout=.1).to(device)
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        h=torch.randint(1,catalog+1,(128,64),device=device); y=h[:,-1]-1
        measurements=[]
        for step in range(12):
            torch.cuda.synchronize(); start=time.perf_counter()
            opt.zero_grad(set_to_none=True); loss=F.cross_entropy(m(h),y); loss.backward(); opt.step()
            torch.cuda.synchronize()
            if step>2: measurements.append(time.perf_counter()-start)
        row=dict(catalog=catalog,seconds_per_batch=float(np.median(measurements)),samples_per_second=128/float(np.median(measurements)),peak_vram_bytes=torch.cuda.max_memory_allocated())
        results.append(row); log('benchmark',**row)
        del opt,m,h,y
    dump(ROOT/'reports/resource_probe.json',dict(device=torch.cuda.get_device_name(),batch_size=128,history=64,results=results))
if __name__=='__main__':main()
