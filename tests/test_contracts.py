import numpy as np
import pytest
import torch
from Data_loader.pipeline import past_indices
from GPT2_models.backbone import GPT2Rec
from SaTP.evaluation import list_metrics
from SaTP.common import inside,ROOT
from Comparison.report import holm

def test_tied_events_and_future_mutation():
    ts=np.array([1,2,2,3,4])
    assert past_indices(ts,2,10).tolist()==[0]
    assert past_indices(ts,3,10).tolist()==[0,1,2]
    changed=np.array([1,2,2,30,40])
    assert np.array_equal(past_indices(ts,2,10),past_indices(changed,2,10))

def test_padding_and_batch_independence():
    torch.manual_seed(42); model=GPT2Rec(20,10,hidden=16,layers=2,heads=2,dropout=0).eval()
    short=torch.tensor([[0,0,1,2]])
    long=torch.tensor([[0,0,0,0,1,2]])
    with torch.no_grad():
        assert torch.allclose(model(short),model(long),atol=1e-6)
        out=model(torch.tensor([[0,0,1,2],[3,4,5,6]]))
        assert torch.allclose(out[0],model(short)[0],atol=1e-6)
        assert torch.isfinite(model(torch.zeros(1,4,dtype=torch.long))).all()

def test_metrics_unknown_target_and_rank():
    result=list_metrics(np.array([0,2,3]),np.array([[1,2,3],[2,1,3],[1,2,3]]))
    assert result["ndcg10"].tolist()==[0.,1.,.5]

def test_no_writes_outside_project():
    with pytest.raises(ValueError): inside(ROOT.parent/"forbidden.txt")

def test_holm():
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])

def test_policy_gradient_and_frozen_reference():
    import copy
    torch.manual_seed(1); ref=GPT2Rec(8,8,hidden=16,layers=1,heads=2,dropout=0).eval()
    policy=copy.deepcopy(ref)
    for p in ref.parameters(): p.requires_grad_(False)
    h=torch.tensor([[0,1,2],[0,3,4]])
    logits=policy(h); loss=-logits.log_softmax(-1)[:,0].mean(); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in policy.parameters())
    assert all(p.grad is None for p in ref.parameters())
