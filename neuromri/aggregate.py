"""Tidy result summaries and paired group bootstrap conditional on one fold/seed.

Never bootstrap stacked repetitions as if repeated proxy groups were new patients.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .statistics import endpoints
from .common import provenance


def paired_bootstrap(left,right,draws=2000,seed=6101,threshold_left=.5,threshold_right=.5):
    for p in [left,right]:
        if not p.group_id.is_unique:
            raise ValueError("Select ONE task/fold/seed/test-cluster/fit before bootstrap; duplicate groups found")
    if set(left.group_id)!=set(right.group_id):
        raise ValueError("Paired contrast requires the exact same final test groups")
    a=left.set_index("group_id").sort_index()
    b=right.set_index("group_id").sort_index()
    np.testing.assert_array_equal(a.y,b.y)
    y=a.y.to_numpy()
    pa,pb=a.score.to_numpy(),b.score.to_numpy()
    keys=["auroc","balanced_accuracy","sensitivity","specificity","brier"]
    def calculate(idx):
        # Compute essential endpoints directly: avoid refitting calibration regressions
        # in thousands of bootstrap draws.
        from sklearn.metrics import roc_auc_score
        yy,aa,bb=y[idx],pa[idx],pb[idx]
        def est(p,t):
            pred=p>=t
            sn=float(pred[yy==1].mean())
            sp=float((~pred[yy==0]).mean())
            return np.array([roc_auc_score(yy,p),(sn+sp)/2,sn,sp,np.mean((p-yy)**2)])
        return est(bb,threshold_right)-est(aa,threshold_left)
    rng=np.random.default_rng(seed)
    pos,neg=np.flatnonzero(y==1),np.flatnonzero(y==0)
    if min(len(pos),len(neg))==0:
        raise ValueError("Both labels required")
    values=np.stack([calculate(np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]) for _ in range(draws)])
    lo,hi=np.quantile(values,[.025,.975],axis=0)
    delta=calculate(np.arange(len(y)))
    return pd.DataFrame(dict(metric=keys,right_minus_left=delta,ci_lo=lo,ci_hi=hi,
        groups=len(y),positive_groups=len(pos),negative_groups=len(neg),bootstrap_draws=draws))


def read_prediction(path,query):
    p=pd.read_csv(path)
    if query:
        p=p.query(query).copy()
    # P1 CSV identity lives in its provenance; P2 CSV contains explicit identity.
    prov=Path(path).parent/"provenance.json"
    c=json.loads(prov.read_text())["config"] if prov.exists() else {}
    identity={}
    for key in ["task","fold","seed","test_cluster"]:
        if key in p:
            if p[key].nunique()!=1:
                raise ValueError(f"Choose one {key} in --query: {path}")
            identity[key]=p[key].iloc[0]
        elif key in c:
            identity[key]=c[key]
    return p,identity


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--metrics",nargs="*",help="Metrics CSV paths; shell glob expansion is supported on Linux")
    p.add_argument("--left")
    p.add_argument("--right")
    p.add_argument("--left-query")
    p.add_argument("--right-query")
    p.add_argument("--left-threshold",type=float,default=.5)
    p.add_argument("--right-threshold",type=float,default=.5)
    p.add_argument("--draws",type=int,default=2000)
    p.add_argument("--seed",type=int,default=6101)
    p.add_argument("--output",default="results/aggregate")
    p.add_argument("--scope",choices=["vega","smoke"],default="vega")
    p.add_argument("--source-integrity-gate", help="Record local CPU reanalysis of a fully gated Vega export")
    a=p.parse_args()
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    inputs=list(a.metrics or [])+[s for s in [a.left,a.right] if s]
    if a.source_integrity_gate:
        if a.scope != 'vega':
            p.error('--source-integrity-gate requires Vega source scope')
        from tools.p1_postprocessing import provenance as local_provenance
        local_provenance(vars(a),out,inputs,a.source_integrity_gate)
    else:
        provenance(vars(a),out,inputs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if a.metrics:
        f=pd.concat([pd.read_csv(path).assign(input_file=str(path)) for path in a.metrics],ignore_index=True)
        f.to_csv(out/"all_metrics.csv",index=False)
        if "status" in f:
            f.groupby(["strategy","k","status"],dropna=False).size().rename("evaluations").to_csv(out/"availability.csv")
        cols=[k for k in ["encoder","task","fold","seed","train_cluster","test_cluster","alpha","policy","strategy","k"] if k in f]
        nums=[k for k in ["auroc","balanced_accuracy","sensitivity","specificity","brier"] if k in f]
        summary=f.groupby(cols,dropna=False)[nums].mean().reset_index()
        summary.to_csv(out/"summary.csv",index=False)
        if "strategy" in summary:
            # These means describe calibration-draw variation; they are not independent N.
            v=summary.groupby(["strategy","k"],dropna=False).specificity.mean().unstack(0)
            ax=v.plot(marker="o",figsize=(9,4))
            ax.axhline(.9,color="black",ls="--",lw=1)
            ax.set(xlabel="Target negative groups",ylabel="Mean group specificity",title="Calibration-draw means; no pooled patient inference")
            ax.figure.tight_layout()
            ax.figure.savefig(out/"specificity.png",dpi=180)
            plt.close(ax.figure)
    if a.left and a.right:
        left,li=read_prediction(a.left,a.left_query)
        right,ri=read_prediction(a.right,a.right_query)
        if li!=ri:
            raise ValueError(f"Pair one task/fold/seed/test population: {li} versus {ri}")
        table=paired_bootstrap(left,right,a.draws,a.seed,a.left_threshold,a.right_threshold)
        table.to_csv(out/"paired_difference.csv",index=False)
        table["identity"]=str(li)
        fig,ax=plt.subplots(figsize=(7,3))
        x=table.right_minus_left.to_numpy()
        ax.errorbar(x,np.arange(len(table)),xerr=np.maximum(0,np.stack([x-table.ci_lo,table.ci_hi-x])),fmt="o")
        ax.set_yticks(np.arange(len(table)),table.metric)
        ax.axvline(0,color="black",ls="--",lw=1)
        ax.set_xlabel("Right − left (95% paired proxy-group bootstrap CI)")
        fig.tight_layout()
        fig.savefig(out/"paired_difference.png",dpi=180)
        plt.close(fig)
        (out/"analysis.json").write_text(json.dumps(dict(left=a.left,right=a.right,left_query=a.left_query,
            right_query=a.right_query,left_threshold=a.left_threshold,right_threshold=a.right_threshold,
            seed=a.seed,draws=a.draws,identity=li,interpretation="Conditional on this fit/fold/seed; stratified proxy-group resampling; not patient inference"),indent=2,default=str))
        print(table.to_string(index=False))
    if not a.metrics and not (a.left and a.right):
        p.error("Provide --metrics or both --left/--right")


if __name__=="__main__":
    main()
