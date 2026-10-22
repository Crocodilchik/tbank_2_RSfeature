import numpy as np
import torch
from torch import nn

class SequenceReward(nn.Module):
    def __init__(self,catalog_size,hidden=48):
        super().__init__(); self.emb=nn.Embedding(catalog_size+2,hidden,padding_idx=0)
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.out=nn.Sequential(nn.Linear(2*hidden+1,64),nn.ReLU(),nn.Linear(64,1))
    def encode(self,h):
        # Compact left-padded history to the right of a valid-prefix tensor for packing.
        valid=h.ne(0); lengths=valid.sum(1); order=torch.argsort((~valid).long(),dim=1,stable=True)
        compact=h.gather(1,order)
        packed=nn.utils.rnn.pack_padded_sequence(self.emb(compact),lengths.clamp(min=1).cpu(),batch_first=True,enforce_sorted=False)
        _,state=self.gru(packed); return state[-1]*lengths.gt(0)[:,None]
    def forward(self,h,a,d):
        return self.predict(self.encode(h),a,d)
    def predict(self,state,a,d):
        return torch.sigmoid(self.out(torch.cat([state,self.emb(a),d[:,None]],dim=-1))).squeeze(-1)

class RewardEnsemble:
    def __init__(self,models,duration,first,device):
        self.models=models; self.device=device
        self.duration=torch.as_tensor(duration,device=device); self.first=torch.as_tensor(first,device=device)
        for m in models:
            m.eval()
            for p in m.parameters(): p.requires_grad_(False); p.grad=None
    @torch.no_grad()
    def encode(self,h): return [m.encode(h) for m in self.models]
    @torch.no_grad()
    def score(self,states,items,timestamps):
        if items.ndim==1: items=items[:,None]
        b,k=items.shape; flat=items.reshape(-1)
        ts=timestamps[:,None].expand(-1,k).reshape(-1)
        dur=self.duration[flat]*self.first[flat].lt(ts)
        values=[]
        for m,state in zip(self.models,states):
            values.append(m.predict(state[:,None,:].expand(-1,k,-1).reshape(b*k,-1),flat,dur).reshape(b,k))
        stack=torch.stack(values)
        return stack.mean(0),stack.std(0,unbiased=False)

def tabular_features(data,duration=None,first=None):
    h=data["history"]; target=data["target"]
    count=(h!=0).sum(1)
    repeat=(h==target[:,None]).sum(1)
    last=(h[:,-1]==target).astype(float)
    unique=np.array([len(np.unique(row[row>0])) for row in h])
    d=data["duration"]
    return np.column_stack([count,repeat,last,unique,d,target]).astype(np.float32)
