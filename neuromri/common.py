"""Configuration, deterministic execution and per-run provenance."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("XFORMERS_DISABLED", "1")
import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


PATH_KEYS = {"data_root", "weights_dir", "output", "embeddings"}


def config_hash(c):
    """Hash of the experiment definition only: machine/run-specific paths are excluded,
    so a row keeps its identity across nodes (node-local data staging) and output dirs.
    The full config, paths included, is still stored in provenance.json."""
    return canonical_hash({k: v for k, v in c.items()
                           if k not in PATH_KEYS and not k.endswith(("_root", "_dir"))})


def config_cli(default):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=default)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = p.parse_args()
    c = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    for s in a.set:
        k, v = s.split("=", 1)
        c[k] = yaml.safe_load(v)
    c["data_root"] = os.environ.get("NEUROMRI_DATA", c.get("data_root", ""))
    c["weights_dir"] = os.environ.get("NEUROMRI_WEIGHTS", c.get("weights_dir", "weights/files"))
    return c


def seed_all(seed, threads=4):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Fused attention can select nondeterministic backward kernels on CUDA.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def seed_worker(worker_id):
    import torch
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return str(e)


REPO_ROOT = Path(__file__).resolve().parents[1]


def git_commit_from_files(root=REPO_ROOT):
    """Resolve HEAD from .git files; compute nodes may have no git binary."""
    try:
        git = root / ".git"
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
        return f"unresolved ref {ref}"
    except OSError as e:
        return str(e)


def source_fingerprint(root=REPO_ROOT):
    """SHA-256 over executed source/config files, independent of git availability."""
    h = hashlib.sha256()
    files = sorted(p for pat in ("neuromri/**/*.py", "configs/**/*", "slurm/**/*", "tools/**/*.py")
                   for p in root.glob(pat) if p.is_file() and "__pycache__" not in p.parts
                   and not p.name.endswith(".local.env"))
    for p in files:
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return dict(sha256=h.hexdigest(), files=len(files))


def provenance(c, out, artifacts=()):
    import torch
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    record = dict(config=c, config_sha256=config_hash(c),
                  git_commit=command(["git", "rev-parse", "HEAD"]),
                  git_dirty=command(["git", "status", "--porcelain"]),
                  git_commit_files=git_commit_from_files(),
                  source_fingerprint=source_fingerprint(),
                  container_sha256=os.environ.get("NEUROMRI_CONTAINER_SHA256", "LOCAL_SMOKE_NO_CONTAINER"),
                  slurm_job=os.environ.get("SLURM_JOB_ID"),
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                  nvidia_smi=command(["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"]),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  artifacts={str(p): digest(p) for p in artifacts},
                  result_scope=c.get("scope", "smoke"))
    if c.get("scope") == "vega":
        if not os.environ.get("SLURM_JOB_ID") or record["container_sha256"] == "LOCAL_SMOKE_NO_CONTAINER":
            raise RuntimeError("Vega results require SLURM and the recorded SIF hash")
        # CPU postprocessing is permitted, GPU work must be on A100.
        if torch.cuda.is_available() and "A100" not in record["gpu"]:
            raise RuntimeError("Reported GPU results require Vega A100")
        if torch.cuda.is_available() and torch.cuda.device_count()!=1:
            raise RuntimeError("Each GPU experiment must see exactly one assigned A100")
    (out / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out / "pip-freeze.txt").write_text(command([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")
    return record
