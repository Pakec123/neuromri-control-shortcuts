"""External manifest adapter using the P2 encoder, loader and cache convention."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import canonical_hash, config_cli, digest, provenance, seed_all


def read_cache(folder, manifest, scope="smoke", encoder_name=None):
    folder = Path(folder)
    meta = json.loads((folder/"cache.json").read_text())
    if meta["key"] != canonical_hash(meta["identity"]):
        raise ValueError("Embedding identity key mismatch")
    for file, key in [("embeddings.npy", "sha256"), ("image_ids.npy", "ids_sha256")]:
        if digest(folder/file) != meta[key]:
            raise ValueError(f"Corrupt embedding cache: {folder/file}")
    if meta["identity"]["manifest"] != digest(manifest):
        raise ValueError("Embedding manifest changed")
    identity = meta["identity"]
    if encoder_name is not None and identity.get("encoder") != encoder_name:
        raise ValueError("Embedding encoder identity mismatch")
    for key, path in [("weights_manifest", Path("weights/manifest.yaml")),
                      ("encoder_code", Path(__file__).with_name("models.py")),
                      ("data_code", Path(__file__).with_name("data.py"))]:
        if identity.get(key) != digest(path):
            raise ValueError(f"Embedding {key} changed")
    if scope == "vega":
        if meta["scope"] != "vega" or meta["identity"]["container_sha256"] != os.environ.get("NEUROMRI_CONTAINER_SHA256"):
            raise ValueError("Vega probes require embeddings from the same Vega SIF")
    ids = np.load(folder/"image_ids.npy", allow_pickle=False)
    x = np.load(folder/"embeddings.npy", mmap_mode="r", allow_pickle=False)
    if (x.ndim != 2 or x.dtype != np.float32 or len(ids) != len(x) or
            len(set(ids)) != len(ids) or list(x.shape) != meta["shape"] or len(ids) != meta["n"] or
            ids.tolist() != identity["image_ids"]):
        raise ValueError("Invalid embedding rows or duplicate IDs")
    return x, pd.Series(np.arange(len(ids)), index=ids), meta


def extract(c):
    import torch
    from .data import loader
    from .models import encoder

    seed_all(c.get("seed", 5101), c.get("threads", 4))
    manifest = Path(c["external_manifest"])
    d = pd.read_csv(manifest)
    if not d.image_id.is_unique or not d.groupby("subject_id").diagnosis.nunique().eq(1).all():
        raise ValueError("Invalid external slice identity")
    root = Path(c["external_root"])
    for row in d.itertuples():
        path = root/row.file
        path.resolve().relative_to(root.resolve())
        if digest(path) != row.sha256:
            raise ValueError(f"External PNG changed: {path}")
    out = Path(c.get("external_embeddings", "cache/external"))/c["encoder"]
    identity = dict(encoder=c["encoder"], scope=c.get("scope", "smoke"),
        manifest=digest(manifest), weights_manifest=digest("weights/manifest.yaml"),
        image_ids=d.image_id.tolist(), precision="float32", seed=c.get("seed", 5101),
        encoder_code=digest(Path(__file__).with_name("models.py")),
        data_code=digest(Path(__file__).with_name("data.py")), extraction_code=digest(__file__),
        pipeline_hashes=sorted(d.pipeline_hash.unique()),
        container_sha256=os.environ.get("NEUROMRI_CONTAINER_SHA256", "LOCAL_SMOKE_NO_CONTAINER"),
        preprocessing="manifest PNG RGB224; P2 aspect padding and encoder normalization")
    key = canonical_hash(identity)
    if (out/"cache.json").exists():
        _, _, meta = read_cache(out, manifest, c.get("scope", "smoke"), c["encoder"])
        if meta["key"] != key:
            raise ValueError("Incompatible external cache: use a new output directory")
        print("Verified cached embeddings:", out)
        return out
    provenance(c, out, [manifest, "weights/manifest.yaml"])
    model, _, mean, std = encoder(c["encoder"], c["weights_dir"])
    model.to(c.get("device", "cuda")).eval()
    d = d.assign(y=d.diagnosis.eq("PD").astype(int), normal_cluster=-1)
    loader_config = dict(c, data_root=str(root))
    start = time.perf_counter()
    features = []
    with torch.inference_mode():
        for x, _, _, _ in loader(d, loader_config, mean, std):
            features.append(model(x.to(c.get("device", "cuda"))).float().cpu().numpy())
    x = np.concatenate(features)
    np.save(out/"embeddings.npy", x)
    np.save(out/"image_ids.npy", d.image_id.to_numpy())
    meta = dict(key=key, identity=identity, sha256=digest(out/"embeddings.npy"),
        ids_sha256=digest(out/"image_ids.npy"), n=len(d), shape=list(x.shape),
        seconds=time.perf_counter()-start, scope=c.get("scope", "smoke"))
    (out/"cache.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(d)} external slices; {meta['seconds']:.2f}s -> {out}")
    return out


if __name__ == "__main__":
    extract(config_cli("configs/p2_external.yaml"))
