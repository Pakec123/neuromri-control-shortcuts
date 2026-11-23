"""All-layer bf16 training, group-disjoint inner selection and negative-only repairs."""
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .common import config_cli, seed_all, provenance, digest
from .data import MATCH, task_roles, support_arm, sample_groups, loader, group_scores
from .models import encoder
from .statistics import endpoints


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -g


class Classifier(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dim,2)
        self.adversary = nn.Linear(dim,3)

    def forward(self,x):
        f = self.backbone(x)
        return self.head(f), f


def sampling_weights(d, policy):
    size = d.groupby("group_id").size()
    counts = d.drop_duplicates("group_id").y.value_counts()
    if policy == "equal_negative_cluster":
        neg_counts = d[d.y==0].drop_duplicates("group_id").normal_cluster.value_counts()
        return [1/(2*counts.loc[1]*size.loc[g]) if y else 1/(2*len(neg_counts)*neg_counts.loc[cl]*size.loc[g])
            for y,g,cl in zip(d.y,d.group_id,d.normal_cluster)]
    return [1/(2*counts.loc[y]*size.loc[g]) for y,g in zip(d.y,d.group_id)]


def evaluate(model, data, c, mean, std, device):
    model.eval()
    scores, ys = [], []
    loss_sum, count = 0., 0
    with torch.inference_mode():
        for x,y,_,_ in loader(data,c,mean,std):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda" and c.get("amp",True)):
                logits,_ = model(x.to(device))
            logits = logits.float()
            loss_sum += F.cross_entropy(logits,y.to(device),reduction="sum").item()
            count += len(y)
            scores.extend(logits.softmax(1)[:,1].cpu().numpy())
            ys.extend(y.numpy())
    return np.asarray(scores), loss_sum/count


def train(c):
    seed_all(c["seed"],c.get("threads",4))
    out = Path(c["output"])
    if (out/"metrics.csv").exists():
        raise FileExistsError(f"Refusing to overwrite completed fit: {out}")
    d = task_roles(c,c["task"],c["fold"],p1=True)
    policy = c["policy"]
    if policy not in ["erm","pooled_erm","equal_negative_cluster","worst_negative_group","negative_adversarial"]:
        raise ValueError(policy)
    pooled = policy != "erm"
    tr = support_arm(d,c["task"],c["alpha"],c["seed"],pooled=pooled)
    if pooled:
        # Keep negative GROUP counts equal to the core support arms; preserve pooled prevalence.
        reference = support_arm(d,c["task"],0.,c["seed"])
        n = reference[reference.y==0].group_id.nunique()
        tr = pd.concat([tr[tr.y==1],sample_groups(tr[tr.y==0],n,np.random.default_rng(c["seed"]))]).sort_index()
    val = support_arm(d,c["task"],c["alpha"],c["seed"],role="source_val",pooled=pooled)
    test = d[d.role=="test"].copy()
    for a,b in [(tr,val),(tr,test),(val,test)]:
        assert not set(a.group_id)&set(b.group_id)
    if pooled and tr[tr.y==0].normal_cluster.nunique() < 3:
        raise ValueError("Repair block needs all three negative strata; increase subset")
    provenance(c,out,[Path(c.get("manifests","manifests"))/n for n in ["images.csv","roles.csv"]]+["weights/manifest.yaml"])
    pd.concat([tr.assign(partition="train"),val.assign(partition="inner_val"),test.assign(partition="test")])[
        ["image_id","group_id","partition","y","normal_cluster"]].to_csv(out/"selection.csv",index=False)
    backbone,dim,mean,std = encoder(c["encoder"],c["weights_dir"])
    device = torch.device(c.get("device","cuda"))
    model = Classifier(backbone,dim).to(device)
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(model.parameters(),lr=c["lr"],weight_decay=c["weight_decay"])
    train_loader = loader(tr,c,mean,std,train=True,weights=sampling_weights(tr,policy))
    accum = max(1,math.ceil(c.get("effective_batch",64)/c["batch_size"]))
    total_steps = math.ceil(len(train_loader)/accum)*c["epochs"]
    warmup = math.ceil(len(train_loader)/accum)*c.get("warmup_epochs",3)
    def multiplier(step):
        if warmup and step < warmup:
            return (step+1)/warmup
        t = (step-warmup)/max(1,total_steps-warmup)
        return .5*(1+math.cos(math.pi*min(max(t,0),1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,multiplier)
    best, best_state, best_epoch, stale = float("inf"), None, -1, 0
    history = []
    for epoch in range(c["epochs"]):
        model.train()
        if device.type=="cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        losses=[]
        for step,(x,y,_,cl) in enumerate(train_loader):
            x,y,cl = x.to(device),y.to(device),cl.to(device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda" and c.get("amp",True)):
                logits,features = model(x)
                loss_per = F.cross_entropy(logits.float(),y,reduction="none")
                if policy=="worst_negative_group" and (y==0).any() and (y==1).any():
                    negative_losses = [loss_per[(y==0)&(cl==i)].mean() for i in range(3) if ((y==0)&(cl==i)).any()]
                    loss = .5*loss_per[y==1].mean()+.5*torch.stack(negative_losses).max()
                else:
                    loss = loss_per.mean()
                if policy=="negative_adversarial" and (y==0).any():
                    source_logits = model.adversary(Reverse.apply(features[y==0]))
                    loss = loss+c.get("repair_weight",.1)*F.cross_entropy(source_logits.float(),cl[y==0])
            # Normalize final partial accumulation window too.
            window_size = min(accum,len(train_loader)-(step//accum)*accum)
            (loss/window_size).backward()
            if (step+1)%accum==0 or step+1==len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        if device.type=="cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        _, val_loss = evaluate(model,val,c,mean,std,device)
        row = dict(epoch=epoch+1,train_loss=float(np.mean(losses)),inner_val_loss=val_loss,
            train_seconds=seconds,images=len(tr),batch_size=c["batch_size"],
            effective_batch=c["batch_size"]*accum,
            peak_memory_gb=torch.cuda.max_memory_allocated()/1e9 if device.type=="cuda" else 0.)
        history.append(row)
        pd.DataFrame(history).to_csv(out/"history.csv",index=False)
        print(row,flush=True)
        if val_loss < best:
            best,best_epoch,stale = val_loss,epoch+1,0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if c.get("early_stopping",True) and stale >= c.get("patience",5):
            break
    model.load_state_dict(best_state)
    # Weights are hashed in memory; writing them is opt-in (1,350 checkpoints would need
    # ~230 GB against a 100 GiB Vega home quota). Predictions/metrics are the study outputs.
    import hashlib, io
    buffer = io.BytesIO()
    torch.save(dict(state_dict=best_state,best_epoch=best_epoch,config=c),buffer)
    checkpoint_sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()
    if c.get("save_checkpoint", False):
        (out/"best.pt").write_bytes(buffer.getvalue())
    del buffer
    p,_ = evaluate(model,test,c,mean,std,device)
    predictions = test[["image_id","group_id","y","normal_cluster"]].copy()
    predictions["score"] = p
    predictions.to_csv(out/"image_predictions.csv",index=False)
    g = group_scores(test,p)
    g.to_csv(out/"predictions.csv",index=False)
    metrics=[]
    identity = {k:c[k] for k in ["encoder","task","alpha","fold","seed","policy"]}
    ng=tr[tr.y==0].drop_duplicates("group_id")
    identity.update(realized_alpha=float(ng.normal_cluster.eq(MATCH[c["task"]]).mean()),
        train_positive_groups=tr[tr.y==1].group_id.nunique(),train_negative_groups=len(ng))
    for cl in range(3):
        v = g[(g.y==1)|(g.normal_cluster==cl)]
        metrics.append(dict(**identity,test_cluster=cl,best_epoch=best_epoch,**endpoints(v.y,v.score)))
    worst = min(r["specificity"] for r in metrics)
    for r in metrics:
        r["worst_negative_specificity"] = worst
    # Same held-out positives; each negative cluster receives equal weight.
    pooled = dict(**identity,test_cluster="standardized_pool",best_epoch=best_epoch,
        worst_negative_specificity=worst,n_groups=len(g))
    for key in ["auroc","balanced_accuracy","sensitivity","specificity","brier"]:
        pooled[key]=float(np.mean([r[key] for r in metrics]))
    metrics.append(pooled)
    pd.DataFrame(metrics).to_csv(out/"metrics.csv",index=False)
    (out/"checkpoint.sha256").write_text(checkpoint_sha256+"\n")
    return pd.DataFrame(metrics)


if __name__ == "__main__":
    train(config_cli("configs/p1.yaml"))
