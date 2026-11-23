"""Frozen group-disjoint P2 probes and threshold transport, CPU only.

Parallel unit = encoder/task/fold/seed/train-cluster; never refit per target draw.
"""
import json
import os
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from .common import config_cli, digest, provenance, canonical_hash, config_hash
from .data import task_roles, groups_for, group_scores, sample_groups
from .statistics import endpoints, decision_endpoints, quantile_threshold, ba_threshold, temperature


def fit_unit(c, name, task, fold, seed, source):
    threadpool_limits(c.get("threads_per_fit",1))
    d = task_roles(c,task,fold)
    cache = Path(c["embeddings"]) / name
    ids = np.load(cache / "image_ids.npy")
    x = np.load(cache / "embeddings.npy", mmap_mode="r")
    position = pd.Series(np.arange(len(ids)),index=ids)
    if not d.index.isin(position.index).all():
        raise ValueError("Embedding IDs do not cover manifest/subset")
    tr = groups_for(d,"train",source)
    # Resample equal negative/positive group budgets, keeping positive set paired across sources.
    rng = np.random.default_rng(seed)
    negs = [groups_for(d,"train",i).query("y == 0").group_id.nunique() for i in range(3)]
    n = min(negs)
    tr = pd.concat([tr[tr.y == 1],sample_groups(tr[tr.y == 0],n,rng)]).sort_index()
    val, cal = groups_for(d,"source_val",source), groups_for(d,"source_cal",source)
    if min(len(tr),len(val),len(cal)) == 0 or min(tr.y.nunique(),val.y.nunique(),cal.y.nunique()) < 2:
        raise ValueError(f"Insufficient development data: {name}/{task}/{fold}/{source}")
    def raw(v):
        return normalize(np.asarray(x[position.loc[v.index].to_numpy()]),norm="l2")
    scaler = StandardScaler().fit(raw(tr))
    xt, xv = scaler.transform(raw(tr)), scaler.transform(raw(val))
    sizes = tr.groupby("group_id").size()
    ng = tr.drop_duplicates("group_id").y.value_counts()
    w = np.array([len(tr)/(2*ng.loc[y]*sizes.loc[g]) for y,g in zip(tr.y,tr.group_id)])
    candidates = []
    for C in c.get("C_grid",[.01,.1,1]):
        model = LogisticRegression(C=C,max_iter=c.get("max_iter",1000),random_state=seed,solver="lbfgs")
        model.fit(xt,tr.y,sample_weight=w)
        v = group_scores(val,model.predict_proba(xv)[:,1])
        candidates.append((log_loss(v.y,v.score,labels=[0,1]),C,model))
    _, best_C, model = min(candidates,key=lambda v:(v[0],v[1]))
    def score(v):
        return group_scores(v,model.predict_proba(scaler.transform(raw(v)))[:,1])
    sc = score(cal)
    ba = ba_threshold(sc.y,sc.score)
    source_q = quantile_threshold(sc.loc[sc.y==0,"score"])
    temp = temperature(sc.y,sc.score)
    records, prediction_parts, image_parts = [], [], []
    identity = dict(encoder=name,task=task,fold=fold,seed=seed,train_cluster=source,C=best_C,
        unsupported_source=task in ["PD","Stroke"],scope=c.get("scope","smoke"))
    for target in c.get("test_clusters",[0,1,2]):
        test_images = groups_for(d,"test",target)
        image_prob = model.predict_proba(scaler.transform(raw(test_images)))[:,1]
        test = group_scores(test_images,image_prob)
        pool = score(d[(d.role=="target_cal") & (d.y==0) & (d.normal_cluster==target)])
        assert not set(pool.group_id) & set(test.group_id)
        pred = test.assign(**identity,test_cluster=target)
        prediction_parts.append(pred)
        ip=test_images[["image_id","group_id","y","normal_cluster"]].assign(score=image_prob,**identity,test_cluster=target)
        image_parts.append(ip)
        # Ranking/calibration endpoints are unchanged by threshold draws.
        base_metrics = endpoints(test.y,test.score)
        tp = expit(logit(np.clip(test.score,1e-7,1-1e-7))/temp)
        temp_metrics = endpoints(test.y,tp)
        image_metrics = endpoints(test_images.y,image_prob)
        itp = expit(logit(np.clip(image_prob,1e-7,1-1e-7))/temp)
        temp_image_metrics = endpoints(test_images.y,itp)
        def record(strategy,threshold,p=None,k=0,draw=-1,status="ok",calibration_groups=()):
            p = test.score.to_numpy() if p is None else p
            row = dict(**identity,test_cluster=target,strategy=strategy,k=k,draw=draw,
                threshold=threshold,status=status,temperature=temp if strategy=="source_temperature" else 1.,
                calibration_groups=";".join(map(str,calibration_groups)),
                n_target_pool=len(pool),config_hash=config_hash(c))
            if status == "ok":
                row.update(temp_metrics if strategy=="source_temperature" else base_metrics)
                row.update(decision_endpoints(test.y,p,threshold))
                image_base = temp_image_metrics if strategy=="source_temperature" else image_metrics
                image_values = dict(image_base)
                image_values.update(decision_endpoints(test_images.y,itp if strategy=="source_temperature" else image_prob,threshold))
                row.update({("n_images" if k=="n_groups" else "image_"+k):v for k,v in image_values.items()})
            records.append(row)
        record("fixed_0.5",.5)
        record("source_ba",ba)
        record("source_specificity90",source_q)
        record("source_temperature",.5,tp)
        for k in c.get("budgets",[0,16,32,64]):
            for draw in range(c.get("draws",20)):
                if k == 0:
                    for method in ["target_quantile","target_tolerance"]:
                        record(method,source_q,k=k,draw=draw)  # Prespecified no-target fallback.
                    continue
                if k > len(pool):
                    for method in ["target_quantile","target_tolerance"]:
                        record(method,np.nan,k=k,draw=draw,status="insufficient_target_groups")
                    continue
                # Same sampled target groups across source models and encoders.
                rng = np.random.default_rng(np.random.SeedSequence([seed,fold,target,k,draw]))
                selected = pool.iloc[rng.choice(len(pool),k,replace=False)]
                for method,tol in [("target_quantile",False),("target_tolerance",True)]:
                    threshold = quantile_threshold(selected.score,tolerance=tol)
                    record(method,threshold,k=k,draw=draw,status="ok" if np.isfinite(threshold) else "bound_unavailable",
                        calibration_groups=selected.group_id)
    return records, pd.concat(prediction_parts,ignore_index=True), pd.concat(image_parts,ignore_index=True)


def run(c):
    out = Path(c.get("output","results/p2"))
    if (out / "metrics.csv").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {out}")
    artifacts = [Path(c.get("manifests","manifests"))/n for n in ["images.csv","roles.csv"]]
    for name in c["encoders"]:
        cache = Path(c["embeddings"])/name
        meta = json.loads((cache/"cache.json").read_text())
        assert digest(cache/"embeddings.npy") == meta["sha256"]
        assert digest(cache/"image_ids.npy") == meta["ids_sha256"]
        assert meta["identity"]["manifest"] == digest(artifacts[0])
        if c.get("scope") == "vega" and meta["scope"] != "vega":
            raise RuntimeError("Reported P2 must use Vega-generated embeddings")
        if c.get("scope") == "vega" and meta["identity"]["container_sha256"] != os.environ.get("NEUROMRI_CONTAINER_SHA256"):
            raise RuntimeError("P2 and frozen extraction must use the same recorded SIF")
        artifacts.append(cache/"cache.json")
    provenance(c,out,artifacts)
    units = list(product(c["encoders"],c["tasks"],c["folds"],c["seeds"],c.get("train_clusters",[0,1,2])))
    results = Parallel(n_jobs=c.get("jobs",1),verbose=5)(delayed(fit_unit)(c,*unit) for unit in units)
    rows = [r for records,_,_ in results for r in records]
    metrics=pd.DataFrame(rows)
    keys=["encoder","task","fold","seed","train_cluster","strategy","k","draw"]
    grouped=metrics.groupby(keys,dropna=False).specificity
    metrics["worst_negative_specificity"]=grouped.transform("min").where(grouped.transform("count")==len(c.get("test_clusters",[0,1,2])))
    metrics.to_csv(out/"metrics.csv",index=False)
    pd.concat([p for _,p,_ in results],ignore_index=True).to_csv(out/"predictions.csv",index=False)
    pd.concat([p for _,_,p in results],ignore_index=True).to_csv(out/"image_predictions.csv",index=False)
    print(f"{len(units)} score fits; {len(rows)} evaluations -> {out}")


if __name__ == "__main__":
    # Call through the package module so joblib/loky workers unpickle fit functions by
    # module reference; pickling them from __main__ breaks on scipy ufuncs (e.g. expit).
    from neuromri.p2_threshold_transport import run as _run
    _run(config_cli("configs/p2.yaml"))
