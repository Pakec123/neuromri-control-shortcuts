"""Fail on absent/changed images and verify frozen group-role isolation."""
import argparse
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from .common import digest


def verify_splits(manifests):
    check_frozen(manifests)
    d = pd.read_csv(Path(manifests) / "images.csv").set_index("image_id")
    folds = pd.read_csv(Path(manifests)/"folds.csv")
    assert set(folds.fold)==set(range(5))
    assert folds.groupby(["grouping","task","group_id"]).fold.nunique().max()==1
    assert folds.groupby(["grouping","task","group_id"]).half.nunique().max()==1
    roles = pd.read_csv(Path(manifests) / "roles.csv")
    for (grouping, task, fold), r in roles.groupby(["grouping", "task", "outer_fold"]):
        assert r.image_id.is_unique
        r = r.join(d[["group_"+grouping, "excluded_cross_label"]], on="image_id")
        assert not r.excluded_cross_label.any()
        assert d.loc[r.image_id].family.isin([task,"Normal"]).all()
        assert r.groupby("group_"+grouping).role.nunique().max() == 1, (grouping, task, fold)
        assert set(r.role) == {"train", "source_val", "source_cal", "target_cal", "test"}
    return len(d), len(roles)


def check_frozen(manifests):
    for file,sha in json.loads((Path(manifests)/"checksums.json").read_text()).items():
        if digest(Path(manifests)/file)!=sha:
            raise ValueError(f"Frozen manifest changed: {file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("NEUROMRI_DATA"))
    p.add_argument("--manifests", default="manifests")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--splits-only", action="store_true")
    a = p.parse_args()
    print("Split verification:", verify_splits(a.manifests))
    if a.splits_only:
        return
    if not a.data_root:
        p.error("Set NEUROMRI_DATA to Resized_224")
    d = pd.read_csv(Path(a.manifests) / "images.csv")
    def check(r):
        f = Path(a.data_root) / r.file
        return r.file if not f.is_file() or digest(f, "md5") != r.md5 else None
    with ThreadPoolExecutor(a.workers) as ex:
        errors = [e for e in ex.map(check, d.itertuples()) if e]
    if errors:
        raise RuntimeError(f"{len(errors)} missing/changed files: {errors[:10]}")
    print(f"MD5 verified {len(d)} read-only images")


if __name__ == "__main__":
    main()
