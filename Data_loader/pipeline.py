import json,hashlib
import numpy as np
import polars as pl
from SaTP.common import ROOT, dump, log
from Data_loader.download import download

def reward_available_time(times,lengths,ratios,guard):
    elapsed=np.nan_to_num(np.ceil(lengths*np.maximum(ratios,0)),nan=10**12)
    return times+np.maximum(guard,elapsed).astype(np.int64)

def past_indices(times, index, length):
    end=int(np.searchsorted(times, times[index], side="left"))
    return np.arange(max(0,end-length),end)

def prepare(c):
    dest=ROOT/"data/processed"/(c["config_hash"][:12]+"_"+hashlib.sha256(open(__file__,"rb").read()).hexdigest()[:8])
    if (dest/"manifest.json").exists(): return dest
    dest.mkdir(parents=True,exist_ok=True)
    path,raw=download(c)
    lf=pl.scan_parquet(path)
    required={"uid","item_id","timestamp","is_organic","event_type","played_ratio_pct","track_length_seconds"}
    if not required.issubset(lf.collect_schema().names()): raise ValueError("Unexpected schema")
    profile=lf.group_by("event_type").agg(pl.len().alias("rows"),pl.col("timestamp").min().alias("min_t"),pl.col("timestamp").max().alias("max_t")).collect()
    dump(dest/"raw_profile.json", profile.to_dicts())
    # Stable ID-based user cohort, independent of future interaction counts.
    users=lf.select("uid").unique().collect().sort("uid").get_column("uid").to_numpy()
    users=users[((users.astype(np.uint64)*np.uint64(2654435761)) % np.uint64(10000)) < c["data"]["user_limit"]]
    df=lf.filter(pl.col("uid").is_in(users.tolist()) & (pl.col("event_type").cast(pl.String)=="listen")).select(sorted(required)).collect().sort(["uid","timestamp","item_id"])
    if not df.height: raise ValueError("No listens")
    cuts=[int(df.get_column("timestamp").quantile(q,interpolation="lower")) for q in c["data"]["split_quantiles"]]
    if len(set(cuts))!=4: raise ValueError("Collapsed split boundaries")
    train=df.filter(pl.col("timestamp")<cuts[0])
    catalog=train.group_by("item_id").len().sort(["len","item_id"],descending=[True,False]).head(c["data"]["catalog_size"]).sort("item_id")
    ids=catalog.get_column("item_id").to_numpy(); mapping={int(x):i+1 for i,x in enumerate(ids)}
    np.save(dest/"item_ids.npy",ids)
    # Immutable item metadata observed only in F; strict historical RM features use first_seen<T.
    meta=train.filter(pl.col("track_length_seconds").is_not_null()).group_by("item_id").agg(pl.col("timestamp").min().alias("first_seen"),pl.col("track_length_seconds").sort_by("timestamp").first().alias("duration"))
    duration=np.zeros(len(ids)+1,np.float32); first=np.full(len(ids)+1,np.iinfo(np.int64).max,np.int64)
    for row in meta.iter_rows(named=True):
        k=mapping.get(int(row["item_id"]),0)
        if k: duration[k]=np.log1p(row["duration"])/10; first[k]=row["first_seen"]
    np.savez(dest/"metadata.npz",duration=duration,first=first)
    bins={s:[] for s in ["F","C","R","V","T"]}; n=c["data"]["history_length"]
    def choose(index,maximum):
        return index[np.linspace(0,len(index)-1,min(maximum,len(index)),dtype=int)] if len(index) else index
    for user in df.partition_by("uid",maintain_order=True):
        t=user["timestamp"].to_numpy().astype(np.int64)
        original=user["item_id"].to_numpy(); tokens=np.array([mapping.get(int(x),0) for x in original],np.int32)
        ratio=user["played_ratio_pct"].cast(pl.Float32).to_numpy()/100
        actual_length=user["track_length_seconds"].cast(pl.Float64).to_numpy()
        available=reward_available_time(t,actual_length,ratio,c["data"]["reward_guard_units"])
        organic=user["is_organic"].to_numpy(); uid=int(user["uid"][0])
        stages=np.searchsorted(cuts,t,side="right")
        for stage,name in enumerate(bins):
            idx=np.flatnonzero(stages==stage)
            if name!="F": idx=idx[organic[idx]==0]
            cap=c["data"]["max_train_per_user"] if name in ["F","C"] else c["data"]["max_rl_per_user"] if name=="R" else c["data"]["max_eval_per_user"]
            # Select distinct recorded timestamps; no unverified conversion to calendar days.
            if name in ["V","T"] and len(idx):
                _,where=np.unique(t[idx],return_index=True); idx=idx[where]
            for i in choose(idx,cap):
                prior=past_indices(t,i,n)
                history=np.zeros(n,np.int32); v=tokens[prior]
                # Unknown history item has its own token, separate from padding.
                if len(v): history[-len(v):]=np.where(v>0,v,len(ids)+1)
                if len(v)<c["data"]["min_history"]: continue
                dur=duration[tokens[i]] if tokens[i] and first[tokens[i]]<t[i] else 0.
                bins[name].append((history,int(tokens[i]),float(min(ratio[i],1)) if np.isfinite(ratio[i]) else np.nan,uid,int(t[i]),float(dur),int(original[i]),int(available[i])))
        if sum(len(x) for x in bins.values())%20000<300: log("prepare_progress",users=uid,rows=sum(len(x) for x in bins.values()))
    counts={}
    if not len(ids): raise ValueError("Empty initial catalog")
    for name,rows in bins.items():
        if not rows: raise ValueError(f"Empty block {name}")
        h,y,r,u,t,d,original,available=zip(*rows)
        split_end=cuts[list(bins).index(name)] if name!="T" else int(df["timestamp"].max())+1
        mature=np.array(available,np.int64)<split_end
        np.savez_compressed(dest/f"{name}.npz",history=np.stack(h),target=np.array(y,np.int64),reward=np.array(r,np.float32),user=np.array(u,np.int64),timestamp=np.array(t,np.int64),duration=np.array(d,np.float32),original_target=np.array(original,np.int64),reward_available_at=np.array(available,np.int64),reward_eligible=mature)
        counts[name]=dict(rows=len(rows),target_coverage=float(np.mean(np.array(y)>0)),users=len(set(u)),mature_rewards=int(mature.sum()))
    dump(dest/"manifest.json",dict(config=c,raw=raw,cuts=cuts,counts=counts,catalog_size=len(ids),user_ids=users.tolist(),time_unit="source order-preserving bins; units require dataset-version verification",availability_assumption="Historical listen IDs are treated as available after their recorded bin; historical completion feedback is NOT an input. Duration uses first prior observation. Reward labels use a conservative maturation guard of max(configured source-unit delay, estimated played seconds). This is a documented delay assumption, not observed event-end timestamps."))
    log("prepared",directory=str(dest),counts=counts)
    return dest

def load(folder,split):
    with np.load(folder/f"{split}.npz") as f: return {k:f[k] for k in f.files}
