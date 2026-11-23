"""Manifest-based data and paired training arms; no dataset mutation."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from .common import seed_worker

MATCH = {"AD": 1, "Tumour": 2, "MS": 0}


@lru_cache(maxsize=8)
def _images(manifests, grouping, subset):
    from .verify_data import check_frozen
    check_frozen(manifests)
    d = pd.read_csv(Path(manifests) / "images.csv")
    d = d[~d.excluded_cross_label].copy()
    key = "group_"+grouping
    d["normal_cluster_image"] = d.normal_cluster
    # Fix membership before subset selection, keeping each proxy group intact.
    modes = d.groupby(key).normal_cluster.value_counts().rename("n").reset_index().sort_values(
        [key,"n","normal_cluster"],ascending=[True,False,True]).drop_duplicates(key).set_index(key).normal_cluster
    d["normal_cluster"] = d[key].map(modes)
    if subset:
        ids = pd.read_csv(subset).image_id
        # Reject partial proxy groups instead of quietly truncating their scores.
        selected = d[key].isin(d.loc[d.image_id.isin(ids),key])
        if set(d.loc[selected,"image_id"]) != set(ids):
            raise ValueError("Subset must contain complete proxy groups")
        d = d[d.image_id.isin(ids)]
    return d.set_index("image_id", drop=False)


def images(c):
    return _images(c.get("manifests","manifests"),c.get("grouping","primary"),c.get("subset")).copy()


@lru_cache(maxsize=2)
def _roles(manifests):
    return pd.read_csv(Path(manifests) / "roles.csv")


def task_roles(c, task, fold, p1=False):
    d = images(c)
    roles = _roles(c.get("manifests", "manifests"))
    roles = roles[(roles.grouping == c.get("grouping", "primary")) & (roles.task == task) & (roles.outer_fold == fold)]
    d = d.join(roles.set_index("image_id")[["role"]], how="inner")
    d["group_id"] = d["group_"+c.get("grouping", "primary")]
    d["y"] = (d.family == task).astype(int)
    if p1:
        # P1 uses full held-out fold; reuse is disclosed in manifest provenance.
        d.loc[d.role == "target_cal", "role"] = "test"
        d.loc[d.role == "source_cal", "role"] = "train"
    return d


def groups_for(d, role, cluster=None):
    m = d.role.eq(role)
    if cluster is not None:
        m &= d.y.eq(1) | d.normal_cluster.eq(cluster)
    return d[m].copy()


def sample_groups(d, n, rng):
    gs = np.sort(d.group_id.unique())
    if n > len(gs):
        raise ValueError(f"Requested {n} groups, have {len(gs)}")
    return d[d.group_id.isin(rng.choice(gs, n, replace=False))]


def support_arm(d, task, alpha, seed, role="train", pooled=False):
    """Equal negative GROUP counts at alpha=0/1; positive set identical across arms."""
    rng = np.random.default_rng(seed)
    v = d[d.role == role]
    pos, neg = v[v.y == 1], v[v.y == 0]
    if pooled:
        return v.copy()
    matched = MATCH[task]
    a, b = neg[neg.normal_cluster == matched], neg[neg.normal_cluster != matched]
    n = min(a.group_id.nunique(), b.group_id.nunique(), pos.group_id.nunique())
    if n < 1:
        raise ValueError("Empty support arm")
    na = int(round(n * alpha))
    return pd.concat([pos, sample_groups(a, na, rng), sample_groups(b, n-na, rng)]).sort_index()


class ImageDataset(Dataset):
    def __init__(self, d, root, mean=(.485,.456,.406), std=(.229,.224,.225), augment=False):
        self.d = d
        self.root = Path(root)
        if not root or not self.root.is_dir():
            raise ValueError("Set NEUROMRI_DATA/data_root to the read-only Resized_224 directory")
        ops = [T.RandomRotation(5, fill=0), T.ColorJitter(brightness=.05, contrast=.05)] if augment else []
        self.transform = T.Compose(ops + [T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.d)

    def __getitem__(self, k):
        row = self.d.iloc[k]
        with Image.open(self.root / row.file) as im:
            im = ImageOps.pad(im.convert("RGB"), (224,224), method=Image.Resampling.BILINEAR, color=0)
            x = self.transform(im)
        return x, int(row.get("y", 0)), int(row.image_id), int(row.normal_cluster)


def loader(d, c, mean, std, train=False, weights=None):
    gen = torch.Generator().manual_seed(int(c.get("seed",5101)))
    sampler = None
    if weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
            len(d), replacement=True, generator=gen)
    return DataLoader(ImageDataset(d, c["data_root"], mean, std, augment=train and c.get("augment", True)),
        batch_size=int(c.get("batch_size",64)), shuffle=train and sampler is None,
        sampler=sampler, num_workers=int(c.get("workers",0)), pin_memory=True,
        worker_init_fn=seed_worker, generator=gen, drop_last=False)


def group_scores(d, scores):
    f = d[["image_id", "group_id", "y", "normal_cluster"]].copy()
    f["score"] = np.asarray(scores)
    assert f.groupby("group_id").y.nunique().max() == 1
    # Membership was fixed to the full-group mode in images(), before slicing.
    assert f.groupby("group_id").normal_cluster.nunique().max() == 1
    return f.groupby("group_id", sort=True).agg(y=("y","first"), score=("score","mean"),
        normal_cluster=("normal_cluster","first"), n_images=("image_id","size")).reset_index()
