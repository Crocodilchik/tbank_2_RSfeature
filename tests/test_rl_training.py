import copy
from pathlib import Path
import numpy as np
import torch
from GPT2_models.backbone import GPT2Rec
from Reward_models.models import SequenceReward,RewardEnsemble
from SaTP.training import rl_train,Budget
from SaTP.common import seed_all

def test_real_rl_function_unfreezes_copied_policy(tmp_path):
    seed_all(7); base=GPT2Rec(12,8,hidden=16,layers=1,heads=2,dropout=0).eval()
    for p in base.parameters(): p.requires_grad_(False)
    data=dict(history=np.array([[0,0,1,2],[0,3,4,5]],np.int32),target=np.array([3,6]),timestamp=np.array([10,20]))
    ens=RewardEnsemble([SequenceReward(12,hidden=8) for _ in range(2)],np.zeros(14,np.float32),np.zeros(14,np.int64),'cpu');ens.offset=0.
    c=dict(training=dict(batch_size=2,rl_lr=.001,beta=.03,pessimism=.2,diversity=.05,horizon=2,rl_steps=2))
    initial={k:v.clone() for k,v in base.state_dict().items()}
    result=rl_train(c,base,ens,data,tmp_path/'model.pt','cpu',Budget(.1),'A3')
    assert any(not torch.equal(initial[k],v) for k,v in result.state_dict().items())
    assert all(torch.equal(initial[k],v) for k,v in base.state_dict().items())
    assert all(p.grad is None for m in ens.models for p in m.parameters())
