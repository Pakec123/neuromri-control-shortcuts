"""P1 run list (core + repair block) and an optional GPU-hour budget from a timing table.

Designs (fixed, pre-registered):
  main (default, PI decision 19 Nov 2026, Vega budget ~550-650 GPU-h in total)
    core   6 backbones x 3 tasks x alpha {0, 0.5, 1} x 5 outer folds x 3 seeds   = 810 fits
    repair 3 backbones x 3 tasks x 4 policies    x 5 outer folds x 3 seeds        = 540 fits
  lean (report section 6b, kept for reference)
    core   4 backbones x 3 tasks x alpha {0, 1}   x 3 folds x 2 seeds             = 144 fits
    repair 2 backbones x 3 tasks x 4 policies     x 3 folds x 2 seeds             = 144 fits
"""
import argparse
from itertools import product
from pathlib import Path
import pandas as pd

TASKS = ["AD", "Tumour", "MS"]
POLICIES = ["pooled_erm", "equal_negative_cluster", "worst_negative_group", "negative_adversarial"]
DESIGNS = {
    "main": dict(core_models=["resnet50", "convnext_tiny", "efficientnet_v2_s", "vit_b_16", "swin_t", "dinov2_b"],
                 alphas=[0, 0.5, 1], folds=[0, 1, 2, 3, 4], seeds=[5101, 5102, 5103],
                 repair_models=["resnet50", "convnext_tiny", "vit_b_16"], expect=(810, 540)),
    "lean": dict(core_models=["resnet50", "convnext_tiny", "vit_b_16", "dinov2_b"],
                 alphas=[0, 1], folds=[0, 1, 2], seeds=[5101, 5102],
                 repair_models=["resnet50", "vit_b_16"], expect=(144, 144)),
}


def make_runs(design="main", repair=True):
    d = DESIGNS[design]
    rows = [dict(block="core", encoder=m, task=t, alpha=a, fold=f, seed=s, policy="erm")
            for m, t, a, f, s in product(d["core_models"], TASKS, d["alphas"], d["folds"], d["seeds"])]
    assert len(rows) == d["expect"][0]
    if repair:
        # Repair support is a common pooled negative-group sample (alpha=-1); realized matched fraction is logged.
        rep = [dict(block="repair", encoder=m, task=t, alpha=-1, fold=f, seed=s, policy=p)
               for m, t, f, s, p in product(d["repair_models"], TASKS, d["folds"], d["seeds"], POLICIES)]
        assert len(rep) == d["expect"][1]
        rows += rep
    for i, r in enumerate(rows):
        r.update(run_id=f"{i:04d}_{r['block']}_{r['encoder']}_{r['task']}_a{r['alpha']:g}_f{r['fold']}_s{r['seed']}_{r['policy']}",
                 batch_size=32, epochs=30)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", choices=sorted(DESIGNS), default="main")
    p.add_argument("--out", default="configs/p1_runs.csv")
    p.add_argument("--core-only", action="store_true")
    p.add_argument("--timing", help="tools/benchmark.py timing.csv (columns encoder, train_seconds, images)")
    p.add_argument("--a100-speedup", type=float, default=1.,
                   help="A100 throughput relative to the GPU that produced --timing")
    p.add_argument("--train-images", type=int, default=8000)
    p.add_argument("--reserve", type=float, default=1.3)
    a = p.parse_args()
    runs = make_runs(a.design, not a.core_only)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(a.out, index=False, lineterminator="\n")
    print(runs.groupby(["block", "encoder"]).size().to_string())
    if a.timing:
        t = pd.read_csv(a.timing).drop_duplicates("encoder", keep="last").set_index("encoder")
        missing = sorted(set(runs.encoder) - set(t.index))
        if missing:
            raise SystemExit(f"timing table lacks: {missing}")
        rows = []
        for (block, name), v in runs.groupby(["block", "encoder"]):
            sec = t.loc[name, "train_seconds"] * a.train_images / t.loc[name, "images"] / a.a100_speedup
            raw = len(v) * 30 * sec / 3600
            rows.append(dict(block=block, encoder=name, runs=len(v), a100_epoch_estimate_s=round(sec, 2),
                             gpu_hours_30epochs=round(raw, 2), gpu_hours_with_reserve=round(raw * a.reserve, 2),
                             a100_speedup_assumption=a.a100_speedup, train_images_assumption=a.train_images))
        b = pd.DataFrame(rows)
        b.to_csv(Path(a.out).with_suffix(".budget.csv"), index=False, lineterminator="\n")
        print(b.to_string(index=False))
        print(f"TOTAL fits={b.runs.sum()}  GPU-h(30 ep)={b.gpu_hours_30epochs.sum():.1f}  "
              f"with reserve x{a.reserve}={b.gpu_hours_with_reserve.sum():.1f}  "
              f"(ideal 4-GPU node-h = /4)")


if __name__ == "__main__":
    main()
