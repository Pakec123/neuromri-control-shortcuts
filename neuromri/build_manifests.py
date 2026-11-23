"""Freeze existing audit evidence; never recluster or modify the image dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .common import digest

TASKS = ["AD", "Tumour", "MS", "PD", "Stroke"]
MATCH = {"AD": 1, "Tumour": 2, "MS": 0}


def freeze(research, out):
    out.mkdir(parents=True, exist_ok=True)
    aud = research / "codex_audit/tables"
    r2 = research / "rounds/R2/tables"
    paths = [aud / n for n in ["image_inventory.csv", "pseudo_groups.csv", "source_clusters.csv", "publisher_manifest_join.csv"]]
    inv, gr, cl, pub = [pd.read_csv(p) for p in paths]
    for frame in [gr, cl, pub]:
        assert np.array_equal(inv["index"], frame["index"])
    d = inv[["index", "relative_path", "class_name", "file_md5"]].copy()
    d.columns = ["image_id", "file", "class_name", "md5"]
    d["file"] = d.file.str.replace("\\", "/", regex=False)
    d["source"] = pub.source_dataset
    d["normal_cluster"] = cl.normal_cluster_selected.astype(int)
    d["group_primary"] = gr.group_primary
    d["group_strict"] = gr.group_strict
    d["family"] = np.select([d.class_name.str.startswith("AD "),
        d.class_name.str.contains("glioma|meningioma|pituitary", case=False),
        d.class_name.str.contains("sclerosis", case=False),
        d.class_name.eq("Parkinson Disease"), d.class_name.eq("Brain Stroke"),
        d.class_name.eq("Normal")], ["AD", "Tumour", "MS", "PD", "Stroke", "Normal"], default="UNKNOWN")
    assert not d.family.eq("UNKNOWN").any()
    # Similarity attribution confidence = cohort nearest-family fraction, NOT ancestry probability.
    mapping = pd.read_csv(r2 / "control_mapping.csv")
    confidence = {c: float(mapping[(mapping.model == "DINOv2-S14") & (mapping.cluster == c)
        & (mapping.family == fam)].nearest_fraction.iloc[0]) for fam, c in MATCH.items()}
    d["mapping_confidence"] = d.normal_cluster.map(confidence)
    d["mapping_family"] = d.normal_cluster.map({v: k for k, v in MATCH.items()})
    bad = d.groupby("group_primary").class_name.nunique()
    d["excluded_cross_label"] = d.group_primary.isin(bad[bad > 1].index)
    # Ten disjoint group strata: two per outer fold, one P2 target pool and one final test.
    # P1 uses both halves as its held-out fold and explicitly reuses P2 outcomes.
    assignments = []
    for mode in ["primary", "strict"]:
        key = "group_" + mode
        valid = d[~d.excluded_cross_label].copy()
        strata = valid.family + ":" + valid.normal_cluster.astype(str)
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=4101)
        fold_ids = np.full(len(valid), -1, int)
        for f, (_, te) in enumerate(outer.split(valid, strata, valid[key])):
            fold_ids[te] = f
        valid["fold"] = fold_ids
        # Within each fold, split calibration and test using a second group-stratified split.
        valid["half"] = -1
        for f in range(5):
            v = valid[valid.fold == f]
            ss = v.family + ":" + v.normal_cluster.astype(str)
            for h, (_, te) in enumerate(StratifiedGroupKFold(2, shuffle=True, random_state=4201+f).split(v, ss, v[key])):
                valid.loc[v.index[te], "half"] = h
        for task in TASKS:
            v = valid[valid.family.isin([task, "Normal"])]
            for row in v.itertuples():
                assignments.append(dict(grouping=mode, task=task, image_id=row.image_id,
                    group_id=getattr(row, key), fold=row.fold, half=row.half))
    folds = pd.DataFrame(assignments)
    d.to_csv(out / "images.csv", index=False,lineterminator="\n")
    folds.to_csv(out / "folds.csv", index=False,lineterminator="\n")
    # Freeze all role assignments, including group-disjoint source development/calibration.
    roles = []
    for (mode, task), v in folds.groupby(["grouping", "task"]):
        for outer_fold in range(5):
            tr = v[v.fold != outer_fold].copy()
            labels = d.set_index("image_id").loc[tr.image_id]
            strata = labels.family.to_numpy() + ":" + labels.normal_cluster.astype(str).to_numpy()
            inner = StratifiedGroupKFold(5, shuffle=True, random_state=4301+outer_fold)
            sub = np.full(len(tr), -1, int)
            for f, (_, ids) in enumerate(inner.split(tr, strata, tr.group_id)):
                sub[ids] = f
            for (_, row), s in zip(tr.iterrows(), sub):
                role = "source_val" if s == 0 else "source_cal" if s == 1 else "train"
                roles.append((mode, task, outer_fold, int(row.image_id), role))
            for row in v[v.fold == outer_fold].itertuples():
                roles.append((mode, task, outer_fold, row.image_id, "target_cal" if row.half == 0 else "test"))
    pd.DataFrame(roles, columns=["grouping", "task", "outer_fold", "image_id", "role"]).to_csv(out / "roles.csv", index=False,lineterminator="\n")
    pd.DataFrame([dict(task=t, train_cluster=i, test_cluster=j, matched_cluster=MATCH.get(t, -1),
        unsupported_source=t in ["PD", "Stroke"]) for t in TASKS for i in range(3) for j in range(3)]).to_csv(out / "crossed_design.csv", index=False,lineterminator="\n")
    paths += [r2 / "control_mapping.csv"]
    evidence = dict(seed_outer=4101, seeds_sampling=[5101,5102,5103], normal_mapping=MATCH,
        builder_sklearn=__import__("sklearn").__version__,
        builder_sha256=digest(__file__),
        mapping_confidence_meaning="DINOv2 nearest disease-family share; not per-image ancestry probability",
        group_meaning="audit near-duplicate proxy components, not patients",
        p1_p2_test_reuse="P1 follow-up reuses P2 test outcomes; never call it independent confirmation",
        p1_outer_folds=5, p1_lean_outer_folds=[0,1,2], p2_pool="heldout half=0; final test half=1",
        sources={str(p.relative_to(research)).replace("\\", "/"): digest(p) for p in paths},
        n_images=len(d), excluded=int(d.excluded_cross_label.sum()))
    (out / "provenance.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8",newline="\n")
    freeze_checksums(out)
    print(json.dumps(evidence, indent=2))


def freeze_checksums(out):
    names=["images.csv","folds.csv","roles.csv","crossed_design.csv","provenance.json"]
    (out/"checksums.json").write_text(json.dumps({n:digest(out/n) for n in names},indent=2),newline="\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("manifests"))
    a = p.parse_args()
    freeze(a.research, a.out)
