import numpy as np
import polars as pl
import torch
from torch.nn import functional as F

def stable_topk(scores,k):
    # Catalog IDs are sorted by raw item-ID; stable sorting resolves equal scores.
    index=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:k]
    return scores.gather(1,index),index

def list_metrics(target,items):
    hit=items==target[:,None]
    positions=np.arange(1,items.shape[1]+1)
    return {"ndcg10":(hit[:,:10]/np.log2(positions[:10]+1)).sum(1),"recall5":hit[:,:5].any(1).astype(float),"recall10":hit[:,:10].any(1).astype(float)}

@torch.no_grad()
def evaluate(c,m,ensemble,data,solution,device,alpha=1.,candidate_count=None,sequential=False):
    m.eval(); bs=c["training"]["batch_size"]; all_items=[]; all_rewards=[]; all_unc=[]; candidate_hits=[]
    for start in range(0,len(data["target"]),bs):
        ix=np.arange(start,min(start+bs,len(data["target"])))
        h=torch.as_tensor(data["history"][ix],device=device,dtype=torch.long); ts=torch.as_tensor(data["timestamp"][ix],device=device)
        states=ensemble.encode(h); logits=m(h); k=min(10,m.catalog_size)
        if sequential:
            chosen=[]; context=h
            for pos in range(k):
                scores=m(context).log_softmax(-1)
                if chosen: scores.scatter_(1,torch.stack(chosen,1)-1,-1e9)
                if solution=="A1":
                    values,indices=stable_topk(scores,min(c["training"]["candidates"],m.catalog_size))
                    mu,_=ensemble.score(states,indices+1,ts)
                    values=values+alpha*(mu+ensemble.offset).clamp(0,1)
                    a=indices.gather(1,values.argmax(1)[:,None]).squeeze(1)+1
                else: a=scores.argmax(1)+1
                chosen.append(a); context=torch.cat([context,a[:,None]],dim=1)
            result=torch.stack(chosen,1)
        elif solution=="A1":
            count=min(candidate_count or c["training"]["candidates"],m.catalog_size)
            scores,items=stable_topk(logits.log_softmax(-1),count); items=items+1
            mu,_=ensemble.score(states,items,ts)
            scores=scores+alpha*(mu+ensemble.offset).clamp(0,1)
            candidates=items.cpu().numpy(); candidate_hits.extend((candidates==data["target"][ix,None]).any(1).tolist())
            result=items.gather(1,stable_topk(scores,k)[1])
        elif solution=="A4" or sequential:
            chosen=[]; context=h
            for pos in range(k):
                scores=m(context)
                if chosen: scores.scatter_(1,torch.stack(chosen,1)-1,-1e9)
                a=scores.argmax(1)+1; chosen.append(a); context=torch.cat([context,a[:,None]],1)
            result=torch.stack(chosen,1)
        else: result=stable_topk(logits,k)[1]+1
        mu,std=ensemble.score(states,result,ts)
        all_items.append(result.cpu().numpy()); all_rewards.extend((mu+ensemble.offset).clamp(0,1).mean(1).cpu().numpy()); all_unc.extend(std.mean(1).cpu().numpy())
    items=np.concatenate(all_items); metrics=list_metrics(data["target"],items)
    rows=dict(query=np.arange(len(items)),user=data["user"],timestamp=data["timestamp"],target=data["target"],original_target=data["original_target"],items=items.tolist(),reward=all_rewards,uncertainty=all_unc,coverage=(data["target"]>0).astype(float))
    rows.update(metrics)
    history=data["history"]
    rows["familiar_fraction"]=np.array([np.isin(items[i],history[i][history[i]>0]).mean() for i in range(len(items))])
    ts=data["timestamp"][:,None]
    duration=ensemble.duration.cpu().numpy()[items]
    known=ensemble.first.cpu().numpy()[items]<ts
    seconds=np.expm1(duration*10)
    rows["known_duration_fraction"]=known.mean(1)
    rows["mean_duration_seconds"]=np.where(known,seconds,0).sum(1)/known.sum(1).clip(min=1)
    emb=F.normalize(ensemble.models[0].emb.weight.detach(),dim=-1)
    redundancy=[]
    for start in range(0,len(items),bs):
        e=emb[torch.as_tensor(items[start:start+bs],device=device)]
        cosine=(e@e.transpose(1,2)).clamp(min=0)
        k=cosine.shape[1]
        redundancy.extend(((cosine.sum((1,2))-k)/max(1,k*(k-1))).cpu().numpy())
    rows["redundancy"]=redundancy
    if candidate_hits: rows["candidate_hit"]=candidate_hits
    frame=pl.DataFrame(rows)
    return frame

def summarize(frame):
    numeric=[x for x in ["reward","uncertainty","ndcg10","recall5","recall10","coverage","candidate_hit","familiar_fraction","mean_duration_seconds","known_duration_fraction","redundancy"] if x in frame.columns]
    result=frame.group_by("user").agg([pl.col(x).mean() for x in numeric]).select([pl.col(x).mean() for x in numeric]).to_dicts()[0]
    supported=frame.filter(pl.col("coverage")>0)
    result["conditional_ndcg10"]=float(supported.group_by("user").agg(pl.col("ndcg10").mean())["ndcg10"].mean()) if supported.height else None
    result.update(queries=frame.height,users=frame["user"].n_unique(),catalog_coverage=len(set(v for row in frame["items"].to_list() for v in row)))
    return result
