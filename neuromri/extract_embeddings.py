"""Extract once per encoder/GPU; cache with row IDs, input and array hashes."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from .common import config_cli, seed_all, digest, provenance, canonical_hash
from .data import images, loader
from .models import encoder


def extract(c):
    seed_all(c.get("seed",5101), c.get("threads",4))
    out = Path(c.get("output", "cache/embeddings")) / c["encoder"]
    d = images(c)
    identity = dict(encoder=c["encoder"], scope=c.get("scope","smoke"), manifest=digest(Path(c.get("manifests","manifests"))/"images.csv"),
        weights_manifest=digest("weights/manifest.yaml"), image_ids=d.image_id.tolist(), precision="float32",
        encoder_code=digest(Path(__file__).with_name("models.py")),
        data_code=digest(Path(__file__).with_name("data.py")),
        extraction_code=digest(__file__),
        container_sha256=os.environ.get("NEUROMRI_CONTAINER_SHA256","LOCAL_SMOKE_NO_CONTAINER"),
        preprocessing="RGB224 aspect-preserving black padding; encoder normalization", seed=c.get("seed",5101))
    key = canonical_hash(identity)
    if (out / "cache.json").exists():
        old = json.loads((out / "cache.json").read_text())
        if old["key"] == key and old["sha256"] == digest(out / "embeddings.npy") and old["ids_sha256"] == digest(out/"image_ids.npy"):
            print("Verified cached embeddings:", out)
            return out
        raise RuntimeError("Incompatible/corrupt cache; use a new output directory")
    provenance(c, out, [Path(c.get("manifests","manifests"))/"images.csv", "weights/manifest.yaml"])
    model, _, mean, std = encoder(c["encoder"], c["weights_dir"])
    device = c.get("device","cuda")
    model.to(device).eval()
    start = time.perf_counter()
    features = []
    with torch.inference_mode():
        for x, _, _, _ in loader(d, c, mean, std):
            features.append(model(x.to(device)).float().cpu().numpy())
    x = np.concatenate(features)
    np.save(out / "embeddings.npy", x)
    np.save(out / "image_ids.npy", d.image_id.to_numpy())
    record = dict(key=key, identity=identity, sha256=digest(out / "embeddings.npy"),
        ids_sha256=digest(out/"image_ids.npy"), n=len(d), shape=list(x.shape),
        seconds=time.perf_counter()-start, scope=c.get("scope","smoke"))
    (out / "cache.json").write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k != "identity"},indent=2))
    return out


if __name__ == "__main__":
    extract(config_cli("configs/extract.yaml"))
