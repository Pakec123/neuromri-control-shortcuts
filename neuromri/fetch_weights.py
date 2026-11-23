"""Login-node download and mandatory SHA256 verification, no runtime downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import hashlib
import shutil
import tempfile
import zipfile
import requests
import yaml
from .common import digest


def prepare_dino(root, revision, verify_only=False):
    """Prepare pinned code on the login node, before read-only compute mounts."""
    code = root / ("dinov2-"+revision)
    with zipfile.ZipFile(root/"dinov2-code.zip") as z:
        if not code.exists():
            if verify_only:
                raise FileNotFoundError(f"{code}; run fetch_weights on the login node")
            tmp = Path(tempfile.mkdtemp(prefix="dino-extract-",dir=root))
            try:
                for member in z.namelist():
                    if not (tmp/member).resolve().is_relative_to(tmp.resolve()):
                        raise ValueError("Unsafe source archive")
                z.extractall(tmp)
                (tmp/code.name).rename(code)
            finally:
                if not tmp.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Temporary source directory escaped weight root")
                shutil.rmtree(tmp)
        # Authenticate extracted code as well as the archive; pycache is ignored.
        for info in z.infolist():
            if info.is_dir():
                continue
            target = root/info.filename
            if not target.resolve().is_relative_to(code.resolve()):
                raise ValueError("Source archive root differs from pinned revision")
            if not target.is_file() or digest(target) != hashlib.sha256(z.read(info)).hexdigest():
                raise ValueError(f"Modified pinned DINO code: {target}")
    return code


def fetch(item, root, verify_only=False):
    target = root / item["file"]
    expected = item["sha256"]
    if not expected or len(expected) != 64:
        raise ValueError(f"Unpinned artifact: {item['file']}")
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f"HASH MISMATCH {target}; remove corrupted file explicitly")
        return str(target)
    if verify_only:
        raise FileNotFoundError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    headers = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") and "huggingface.co" in item["url"] else {}
    with requests.get(item["url"], stream=True, timeout=(30, 120), headers=headers) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for block in r.iter_content(4 * 1024 * 1024):
                f.write(block)
    if digest(tmp) != expected:
        raise ValueError(f"Downloaded hash mismatch: {tmp}")
    tmp.replace(target)
    return str(target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="weights/manifest.yaml")
    p.add_argument("--dest", type=Path, default=Path(os.environ.get("NEUROMRI_WEIGHTS", "weights/files")))
    p.add_argument("--encoders", nargs="*")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    m = yaml.safe_load(Path(a.manifest).read_text())
    selected = a.encoders or [k for k,v in m["encoders"].items() if v["status"] == "available"]
    items = {}
    for k in selected:
        e = m["encoders"][k]
        if e["status"] != "available":
            raise ValueError(f"{k}: {e['status']}: {e.get('note')}")
        for artifact in e["artifacts"]:
            items[artifact["file"]] = artifact
    with ThreadPoolExecutor(a.workers) as ex:
        for path in ex.map(lambda i: fetch(i, a.dest, a.verify_only), items.values()):
            print("Verified", path, flush=True)
    revisions = {m["encoders"][k]["revision"] for k in selected if m["encoders"][k]["backend"]=="dinov2"}
    for revision in revisions:
        print("Verified source",prepare_dino(a.dest,revision,a.verify_only))


if __name__ == "__main__":
    main()
