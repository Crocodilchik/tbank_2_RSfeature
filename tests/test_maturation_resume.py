import shutil
import numpy as np
import torch
import pytest
from Data_loader.pipeline import reward_available_time
from SaTP.run import trial_checkpoint
from SaTP.training import save,rl_train,Budget
from SaTP.common import seed_all
from GPT2_models.backbone import GPT2Rec
from Reward_models.models import SequenceReward,RewardEnsemble

def test_label_maturation_uses_elapsed_time_and_minimum_guard():
    available=reward_available_time(np.array([10,20,30]),np.array([100.,100.,100.]),np.array([.5,1.5,np.nan]),20)
    assert available[:2].tolist()==[60,170]
    assert available[2]>10**9

def test_hpo_checkpoint_identity_independent_of_trial_order(tmp_path):
    assert trial_checkpoint(tmp_path,'A2',{'beta':.01})!=trial_checkpoint(tmp_path,'A2',{'beta':.03})
    assert trial_checkpoint(tmp_path,'A3',{'beta':.03,'pessimism':.2})==trial_checkpoint(tmp_path,'A3',{'pessimism':.2,'beta':.03})

def test_atomic_checkpoint_keeps_previous_on_failed_write(tmp_path,monkeypatch):
    m=GPT2Rec(8,8,hidden=16,layers=1,heads=2,dropout=0);p=tmp_path/'model.pt';save(p,m)
    original=p.read_bytes()
    def fail(*args,**kwargs):raise RuntimeError('simulated interrupted serialization')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(RuntimeError):save(p,m)
    assert p.read_bytes()==original

def test_partial_rl_resume_restores_rng_and_optimizer(tmp_path):
    seed_all(7);base=GPT2Rec(12,8,hidden=16,layers=1,heads=2,dropout=0).eval()
    data=dict(history=np.array([[0,0,1,2],[0,3,4,5]],np.int32),target=np.array([3,6]),timestamp=np.array([10,20]))
    ens=RewardEnsemble([SequenceReward(12,hidden=8) for _ in range(2)],np.zeros(14,np.float32),np.zeros(14,np.int64),'cpu');ens.offset=0.
    c=dict(training=dict(batch_size=2,rl_lr=.001,beta=.03,pessimism=.2,diversity=.05,horizon=2,rl_steps=4,checkpoint_every=2))
    seed_all(42);expected=rl_train(c,base,ens,data,tmp_path/'original.pt','cpu',Budget(.1),'A3')
    shutil.copyfile(tmp_path/'original.partial.pt',tmp_path/'resumed.partial.pt')
    seed_all(999);resumed=rl_train(c,base,ens,data,tmp_path/'resumed.pt','cpu',Budget(.1),'A3')
    assert all(torch.equal(expected.state_dict()[k],resumed.state_dict()[k]) for k in expected.state_dict())
