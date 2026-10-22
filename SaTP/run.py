import argparse, copy, json, platform, time, traceback,hashlib,os
from pathlib import Path
import numpy as np
import polars as pl
import torch
from catboost import CatBoostRegressor
from SaTP.common import ROOT,config,dump,seed_all,log,digest,code_fingerprint,set_log_path
from Data_loader.pipeline import prepare,load
from Reward_models.models import tabular_features
from SaTP.training import Budget,BudgetExceeded,model,supervised,fit_rewards,rl_train,reward_predictions
from SaTP.evaluation import evaluate,summarize

def cat_features(data):
    matrix=tabular_features(data).astype(object)
    matrix[:,5]=[str(int(x)) for x in matrix[:,5]]
    return matrix

def diagnostic(c,blocks,folder,budget):
    path=folder/"diagnostic.cbm"
    fit=blocks["F"]; valid=np.isfinite(fit["reward"])&(fit["target"]>0)&fit["reward_eligible"]
    m=CatBoostRegressor(iterations=150,depth=5,learning_rate=.05,loss_function="RMSE",random_seed=987,verbose=False,thread_count=4,allow_writing_files=False)
    if path.exists(): m.load_model(str(path))
    else:
        budget.check(); m.fit(cat_features({k:v[valid] for k,v in fit.items()}),fit["reward"][valid],cat_features=[5]); m.save_model(str(path))
    return m

def predicted_diagnostic(diag,data,frame,meta):
    items=np.asarray(frame["items"].to_list(),dtype=np.int64); b,k=items.shape
    expanded={key:np.repeat(value,k,axis=0) for key,value in data.items()}
    expanded["target"]=items.reshape(-1)
    ts=expanded["timestamp"]; a=expanded["target"]
    expanded["duration"]=meta["duration"][a]*(meta["first"][a]<ts)
    pred=diag.predict(cat_features(expanded)).reshape(b,k).mean(1)
    return frame.with_columns(pl.Series("diagnostic_reward",pred))

def trial_checkpoint(folder,solution,params):
    signature=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:12]
    return folder/f"{solution}_params_{signature}.pt"

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/smoke.yaml"); parser.add_argument("--prepare-only",action="store_true"); parser.add_argument("--seed",type=int); parser.add_argument("--no-tune",action="store_true")
    args=parser.parse_args(); c=config(ROOT/args.config); start=time.monotonic()
    runroot=ROOT/"experiments"/(c["config_hash"][:12]+"_"+code_fingerprint()[:8]); runroot.mkdir(parents=True,exist_ok=True)
    set_log_path(runroot/"events.jsonl")
    budget=Budget(min(c["training"]["max_hours"],c["training"]["hard_max_hours"]))
    status=runroot/"status.json"
    dump(status,dict(state="running",config=c,pid=os.getpid(),start_unix=time.time()))
    try:
        folder=prepare(c)
        if args.prepare_only:
            dump(status,dict(state="prepared",folder=str(folder))); return
        blocks={name:load(folder,name) for name in "FCRVT"}
        meta=dict(np.load(folder/"metadata.npz")); size=len(np.load(folder/"item_ids.npy"))
        device="cuda" if torch.cuda.is_available() else "cpu"
        dump(runroot/"environment.json",dict(python=platform.python_version(),torch=torch.__version__,device=device,gpu=torch.cuda.get_device_name() if device=="cuda" else None,config=c,code_sha256=code_fingerprint(),data_manifest_sha256=digest(folder/"manifest.json")))
        ensemble=fit_rewards(c,blocks["F"],blocks["C"],runroot/"reward_models",size,meta,device,budget)
        diag=diagnostic(c,blocks,runroot/"reward_models",budget)
        rm_quality={}
        for split in ["V"]:
            pred,y=reward_predictions(ensemble,blocks[split],device,c["training"]["batch_size"])
            pred=np.clip(pred+ensemble.offset,0,1)
            good=(blocks[split]["target"]>0)&np.isfinite(blocks[split]["reward"])&blocks[split]["reward_eligible"]
            dp=diag.predict(cat_features({key:value[good] for key,value in blocks[split].items()}))
            constant=float(np.nanmean(blocks["F"]["reward"][blocks["F"]["reward_eligible"]]))
            rm_quality[split]=dict(constant_mae=float(np.mean(abs(y-constant))),rows=len(y),mae=float(np.mean(abs(pred-y))),rmse=float(np.sqrt(np.mean((pred-y)**2))),diagnostic_mae=float(np.mean(abs(dp-blocks[split]["reward"][good]))))
        # Test RM metrics are written for reporting only and never used in policy selection.
        dump(runroot/"reward_quality.json",rm_quality)
        selection_path=runroot/"selection.json"; selections=json.loads(selection_path.read_text()) if selection_path.exists() else {}
        seeds=[args.seed] if args.seed is not None else c["seeds"]
        if seeds[0]!=42 and not selections: raise ValueError("Select hyperparameters on seed42 before other seeds")
        for seed in seeds:
            seed_all(seed); out=runroot/f"seed_{seed}"; out.mkdir(exist_ok=True)
            base=model(c,size,device); supervised(c,base,blocks["F"],out/"B0.pt",device,budget)
            base.eval()
            models={"B0":base}
            vbase=summarize(evaluate(c,base,ensemble,blocks["V"],"B0",device))
            threshold=vbase["ndcg10"]*(1-c["training"]["relevance_tolerance"])
            for solution in ["A1","A2","A3","A4"]:
                if solution in selections: options=[selections[solution]["params"]]
                elif solution=="A1": options=[dict(alpha=x) for x in [0.,.5,2.,8.]]
                elif solution=="A2": options=[dict(beta=x) for x in ([c["training"]["beta"]] if args.no_tune else [.01,.03,.1])]
                elif solution=="A3": options=[dict(pessimism=x,beta=selections.get("A2",{}).get("params",{}).get("beta",c["training"]["beta"])) for x in ([c["training"]["pessimism"]] if args.no_tune else [0.,.2,.5])]
                else: options=[dict(pessimism=selections.get("A3",{}).get("params",{}).get("pessimism",c["training"]["pessimism"]),beta=selections.get("A3",{}).get("params",{}).get("beta",c["training"]["beta"]),diversity=x) for x in ([c["training"]["diversity"]] if args.no_tune else [0.,.05])]
                candidates=[]
                for trial,params in enumerate(options):
                    budget.check(); seed_all(seed)
                    policy=base if solution=="A1" else rl_train(c,base,ensemble,blocks["R"],trial_checkpoint(out,solution,params),device,budget,solution,**params)
                    vf=evaluate(c,policy,ensemble,blocks["V"],solution,device,alpha=params.get("alpha",1.))
                    metrics=summarize(vf); candidates.append((policy,params,metrics,trial)); log("validation",solution=solution,seed=seed,params=params,metrics=metrics)
                feasible=[x for x in candidates if x[2]["ndcg10"]>=threshold]
                selected=max(feasible,key=lambda x:x[2]["reward"]) if feasible else max(candidates,key=lambda x:x[2]["ndcg10"])
                policy,params,metrics,trial=selected
                models[solution]=policy
                if solution not in selections:
                    selections[solution]=dict(params=params,validation=metrics,meets_relevance_constraint=bool(feasible),selected_on_seed=seed,selected_checkpoint=trial_checkpoint(out,solution,params).name if solution!="A1" else "B0.pt")
                    dump(selection_path,selections)
                if solution!="A1":
                    from SaTP.training import save
                    save(out/f"{solution}.pt",policy,completed=True,params=params)
            if "T" not in rm_quality:
                pred,y=reward_predictions(ensemble,blocks["T"],device,c["training"]["batch_size"])
                pred=np.clip(pred+ensemble.offset,0,1)
                constant=float(np.nanmean(blocks["F"]["reward"][blocks["F"]["reward_eligible"]]))
                rm_quality["T"]=dict(constant_mae=float(np.mean(abs(y-constant))),rows=len(y),mae=float(np.mean(abs(pred-y))),rmse=float(np.sqrt(np.mean((pred-y)**2))))
                dump(runroot/"reward_quality.json",rm_quality)
            # Control uses precisely the R block, with the same initialization.
            continued=copy.deepcopy(base)
            for p in continued.parameters(): p.requires_grad_(True)
            supervised(c,continued,blocks["R"],out/"control_sft.pt",device,budget)
            models["control_sft"]=continued
            summaries={}
            # Evaluation is performed only after model/parameter selection is complete.
            for solution,policy in models.items():
                budget.check()
                params=selections.get(solution,{}).get("params",{})
                eval_started=time.monotonic()
                frame=evaluate(c,policy,ensemble,blocks["T"],solution,device,alpha=params.get("alpha",1.))
                frame=predicted_diagnostic(diag,blocks["T"],frame,meta)
                frame.write_parquet(out/f"{solution}_predictions.parquet")
                summary=summarize(frame); summary["diagnostic_reward"]=float(frame.group_by("user").agg(pl.col("diagnostic_reward").mean())["diagnostic_reward"].mean())
                summary["evaluation_seconds"]=time.monotonic()-eval_started
                summaries[solution]=summary
                common=evaluate(c,policy,ensemble,blocks["T"],solution,device,alpha=params.get("alpha",1.),sequential=True)
                common.write_parquet(out/f"{solution}_nextk_predictions.parquet")
                dump(out/f"{solution}_nextk_metrics.json",summarize(common))
                log("test_complete",solution=solution,seed=seed,metrics=summary)
            dump(out/"metrics.json",summaries)
        dump(status,dict(state="completed",seconds=time.monotonic()-start,config_hash=c["config_hash"],seeds=seeds,scope="technical smoke" if "smoke" in c["protocol"] else "resource-bounded research cohort",peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else None))
        from Comparison.report import build
        build(runroot)
    except Exception as exc:
        dump(status,dict(state="budget_exceeded" if isinstance(exc,BudgetExceeded) else "failed",error=str(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start)); raise

if __name__=="__main__": main()
