import argparse,json
from pathlib import Path
import numpy as np
import polars as pl
import nbformat as nbf
from SaTP.common import ROOT,dump

def paired(a,b,column="ndcg10",replicates=2000):
    aa=a.group_by("user").agg(pl.col(column).mean()).rename({column:"a"})
    bb=b.group_by("user").agg(pl.col(column).mean()).rename({column:"b"})
    joined=aa.join(bb,on="user",how="inner").sort("user")
    if joined.height!=aa.height or joined.height!=bb.height: raise ValueError("Different user populations")
    delta=(joined["b"]-joined["a"]).to_numpy(); rng=np.random.default_rng(42)
    boots=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(replicates)])
    # Paired sign-flip test under exchangeability of the model labels within users.
    perm=np.array([(delta*rng.choice([-1,1],len(delta))).mean() for _ in range(replicates)])
    return dict(delta=float(delta.mean()),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),p=float((1+np.sum(abs(perm)>=abs(delta.mean())))/(replicates+1)),users=len(delta))

def noninferiority(a,b,margin=.02,replicates=2000):
    aa=a.group_by("user").agg(pl.col("ndcg10").mean()).rename({"ndcg10":"a"})
    bb=b.group_by("user").agg(pl.col("ndcg10").mean()).rename({"ndcg10":"b"})
    joined=aa.join(bb,on="user",how="inner").sort("user")
    contrast=(joined["b"]-(1-margin)*joined["a"]).to_numpy()
    rng=np.random.default_rng(42)
    boot=np.array([rng.choice(contrast,len(contrast),replace=True).mean() for _ in range(replicates)])
    return dict(relative_margin=margin,contrast=float(contrast.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)))

def holm(values):
    order=np.argsort(values); adjusted=np.ones(len(values)); previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,(len(values)-rank)*values[i])); adjusted[i]=previous
    return adjusted.tolist()

def build(root):
    root=Path(root).resolve()
    if not root.is_relative_to(ROOT): raise ValueError("Outside project")
    rows=[]; frames={}
    for folder in sorted(root.glob("seed_*")):
        if not (folder/"metrics.json").exists(): continue
        metrics=json.loads((folder/"metrics.json").read_text())
        for name,values in metrics.items():
            rows.append(dict(seed=int(folder.name.split("_")[-1]),solution=name,**values))
            frame=pl.read_parquet(folder/f"{name}_predictions.parquet")
            frames.setdefault(name,[]).append(frame)
    if not rows: raise ValueError("No completed cells")
    table=pl.DataFrame(rows); table.write_parquet(root/"summary.parquet")
    # Never silently compare different seed sets or query registries.
    base_seeds={r["seed"] for r in rows if r["solution"]=="B0"}
    for name in frames:
        if {r["seed"] for r in rows if r["solution"]==name}!=base_seeds: raise ValueError("Different seed sets")
        for frame,base in zip(frames[name],frames["B0"]):
            if not frame.select("query","user","timestamp","target").equals(base.select("query","user","timestamp","target")): raise ValueError("Different evaluation queries")
    averages={name:pl.concat(parts).group_by("query","user","timestamp","target").agg(pl.col("ndcg10").mean(),pl.col("reward").mean()) for name,parts in frames.items()}
    comparisons={}
    for name in ["A1","A2","A3","A4"]:
        comparisons[name]=paired(averages["B0"],averages[name]) if name in averages else dict(p=1.,state="missing")
    for name in comparisons:
        if name in averages:
            comparisons[name]["reward_comparison"]=paired(averages["B0"],averages[name],column="reward")
            baseline=float(averages["B0"].group_by("user").agg(pl.col("ndcg10").mean())["ndcg10"].mean())
            comparisons[name]["relative_delta"]=comparisons[name]["delta"]/baseline if baseline>0 else None
            comparisons[name]["noninferiority_margin"]=-.02*baseline
            comparisons[name]["noninferiority"]=noninferiority(averages["B0"],averages[name])
            comparisons[name]["proxy_gain_with_noninferiority_ci"]=comparisons[name]["noninferiority"]["ci_low"]>=0 and comparisons[name]["reward_comparison"]["ci_low"]>0
    adjusted=holm([comparisons[n]["p"] for n in comparisons])
    for name,p in zip(comparisons,adjusted): comparisons[name]["holm_p"]=p
    dump(root/"comparisons.json",comparisons)
    status=json.loads((root/"status.json").read_text()); config=json.loads((root/"environment.json").read_text())["config"]
    notebook=nbf.v4.new_notebook()
    notebook.metadata["kernelspec"]={"display_name":"Project Python","language":"python","name":"python3"}
    notebook.cells=[nbf.v4.new_markdown_cell("# Yambda: GPT-2 and reward optimization\n\n"+"Technical smoke results are not evidence of scientific superiority.\n\n"+"This report uses observed-listen logs, a restricted fixed catalog, and model-predicted reward. It does not estimate causal retention effects."),nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport polars as pl\nimport matplotlib.pyplot as plt\nroot=Path("+repr(str(root))+ ")\nsummary=pl.read_parquet(root/'summary.parquet')\nsummary"),nbf.v4.new_code_cell("summary.group_by('solution').agg(pl.col('ndcg10').mean(),pl.col('ndcg10').std().alias('seed_std'),pl.col('reward').mean(),pl.col('diagnostic_reward').mean(),pl.col('coverage').mean()).sort('solution')"),nbf.v4.new_code_cell("json.loads((root/'reward_quality.json').read_text())"),nbf.v4.new_code_cell("json.loads((root/'comparisons.json').read_text())"),nbf.v4.new_code_cell("means=summary.group_by('solution').agg(pl.col('ndcg10').mean(),pl.col('reward').mean())\nfig,ax=plt.subplots(figsize=(7,4))\nfor row in means.iter_rows(named=True):\n    ax.scatter(row['ndcg10'],row['reward'])\n    ax.annotate(row['solution'],(row['ndcg10'],row['reward']))\nax.set(xlabel='NDCG@10',ylabel='Predicted reward',title='Relevance and proxy reward')\nfig.tight_layout()\nplt.show()"),nbf.v4.new_markdown_cell("## Scope and limitations\n\nCatalog coverage is included in unconditional metrics. Unknown targets receive zero hits. Previously listened tracks remain eligible. Same-bin events do not enter each other's histories. Reward model errors are measured only on observed actions. Ensemble disagreement is not a calibrated confidence interval. The training cohort and compute budget are explicitly restricted.\n\nPending extensions: full cross-decoding evaluation, richer behavioral slices, all preregistered ablations, strict completion-availability evidence, and final resource study."),nbf.v4.new_code_cell("json.loads((root/'status.json').read_text())")]
    target=ROOT/"reports"/(root.name+"_comparison.ipynb"); nbf.write(notebook,target)
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    manager=KernelManager(kernel_name="python3")
    manager.kernel_spec.argv=[str(ROOT/".venv/Scripts/python.exe"),"-m","ipykernel_launcher","-f","{connection_file}"]
    NotebookClient(notebook,timeout=120,km=manager,resources={"metadata":{"path":str(ROOT)}} ).execute(cleanup_kc=True)
    nbf.write(notebook,target)
    print("REPORT",target,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("run"); build(parser.parse_args().run)
