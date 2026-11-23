"""Audit a read-only P1 export before aggregation or inference.

This command never computes endpoints, changes provenance, or selects duplicate
fits. A nonzero exit status blocks the downstream analysis. Full file hashes and
per-fit findings are written outside the input tree.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuromri.result_validation import identity, validate_provenance
from tools.p1_attestation import verify_attestation, accepts_missing_git

PRODUCTION_COMMIT = "469ee8f1aa6f5bbacaf6dc3d078b9920a01b5941"
PATH_KEYS = {"data_root", "weights_dir", "output"}
REQUIRED = {"metrics.csv", "provenance.json", "predictions.csv", "selection.csv",
            "history.csv", "checkpoint.sha256", "pip-freeze.txt"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def definition(config):
    return {k: v for k, v in config.items()
            if k not in PATH_KEYS and not k.endswith(("_root", "_dir"))}


def config_sha(config):
    # Exact serialization used by neuromri.common.config_hash; no torch import.
    return hashlib.sha256(json.dumps(definition(config), sort_keys=True,
                                     default=str).encode()).hexdigest()


def check_record(record, run, base, commit, attestation=None):
    failures = []
    attested = accepts_missing_git(record, attestation)
    try:
        validate_provenance(record, fit=True, attested=attested)
    except ValueError as exc:
        failures.append("existing_provenance_gate: " + str(exc))
    if record.get("git_commit") != commit and not attested:
        failures.append("production_commit_unverified")
    if record.get("git_dirty") != "" and not attested:
        failures.append("clean_checkout_unverified")
    config = record.get("config", {})
    if record.get("config_sha256") != config_sha(config):
        failures.append("config_hash_mismatch")
    expected = dict(base)
    for key in ["encoder", "task", "policy"]:
        expected[key] = run[key]
    for key in ["fold", "seed", "batch_size", "epochs"]:
        expected[key] = int(run[key])
    expected["alpha"] = float(run["alpha"])
    if definition(config) != definition(expected):
        failures.append("protocol_config_mismatch")
    if str(config.get("output", "")).replace("\\", "/").rstrip("/").split("/")[-1] != run["run_id"]:
        failures.append("config_output_run_id_mismatch")
    for field in ["slurm_job", "gpu", "nvidia_smi", "torch", "cuda"]:
        if not record.get(field):
            failures.append("missing_provenance_field: " + field)
    return failures


def audit(run_root, output, commit=PRODUCTION_COMMIT, reported_count=1400, attestation_dir=None):
    run_root, output = run_root.resolve(), output.resolve()
    if output.is_relative_to(run_root):
        raise ValueError("Audit output must be outside the read-only input tree")
    runlist = rows(ROOT / "configs/p1_runs.csv")
    planned = {r["run_id"] for r in runlist}
    failures = []
    if len(planned) != 1350 or len(runlist) != 1350 or len({identity(r) for r in runlist}) != 1350:
        failures.append("invalid_planned_grid")
    metrics = sorted(run_root.rglob("metrics.csv"))
    present = {p.parent.relative_to(run_root).as_posix() for p in metrics}
    missing, extra = sorted(planned - present), sorted(present - planned)
    if missing or extra:
        failures.append("missing_or_extra_metric_paths")
    base = yaml.safe_load((ROOT / "configs/p1.yaml").read_text(encoding="utf-8"))
    attestation = None
    if attestation_dir:
        try:
            attestation = verify_attestation(attestation_dir, runlist, [json.loads(
                (run_root / r["run_id"] / "provenance.json").read_text(encoding="utf-8")) for r in runlist], commit)
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
            failures.append("attestation_failed: " + str(exc))
    frozen = {str(p): sha(ROOT / p) for p in
              [Path("manifests/images.csv"), Path("manifests/roles.csv"), Path("weights/manifest.yaml")]}
    frozen = {k.replace("\\", "/"): v for k, v in frozen.items()}
    fits, records, metric_rows = [], [], 0
    for run in runlist:
        folder = run_root / run["run_id"]
        errors = []
        absent = sorted(name for name in REQUIRED if not (folder / name).is_file())
        if absent:
            errors.append("missing_files: " + ", ".join(absent))
        record = {}
        try:
            record = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
            records.append(record)
            errors.extend(check_record(record, run, base, commit, attestation))
            if record.get("artifacts") != frozen:
                errors.append("frozen_input_hash_mismatch")
            data = rows(folder / "metrics.csv")
            metric_rows += len(data)
            if len(data) != 4 or Counter(r["test_cluster"] for r in data) != Counter(
                    ["0", "1", "2", "standardized_pool"]):
                errors.append("metric_strata_mismatch")
            if any(identity(r) != identity(run) for r in data):
                errors.append("metric_identity_mismatch")
            if not re.fullmatch("[0-9a-f]{64}", (folder / "checkpoint.sha256").read_text().strip()):
                errors.append("invalid_checkpoint_hash")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"unreadable_fit: {type(exc).__name__}: {exc}")
        fits.append(dict(run_id=run["run_id"], eligible=not errors, failures=errors,
                         config_sha256=record.get("config_sha256")))
    if any(r["failures"] for r in fits):
        failures.append("fit_integrity_failures")
    # Hash every exported byte, including any unexpected/nested files.
    inventory = [dict(path=p.relative_to(run_root).as_posix(), bytes=p.stat().st_size,
                      sha256=sha(p)) for p in sorted(run_root.rglob("*")) if p.is_file()]
    output.mkdir(parents=True, exist_ok=True)
    inventory_path = output / "input_inventory.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(inventory)
    versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "torch", "statsmodels", "PyYAML"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    result = dict(
        status="FAIL" if failures else "PASS", failures=failures,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), input_root=str(run_root),
        expected_production_commit=commit, reported_metric_file_count=reported_count,
        commit_identity_attestation=attestation,
        observed_metric_file_count=len(metrics), reported_count_agrees=len(metrics) == reported_count,
        planned_fits=len(runlist), missing_run_ids=missing, extra_metric_paths=extra,
        metric_rows=metric_rows, eligible_fits=sum(r["eligible"] for r in fits),
        failure_counts=dict(Counter(e for r in fits for e in r["failures"])),
        provenance_values={k: dict(Counter(str(r.get(k)) for r in records)) for k in
                           ["git_commit", "git_dirty", "container_sha256", "slurm_job", "gpu", "torch", "cuda", "result_scope"]},
        determinism=dict(explicit_runtime_flags_persisted=False, repeat_evidence_present=False,
                         note="No per-fit runtime determinism flags or GPU repeat evidence in this export; source intent alone cannot verify execution without code identity."),
        input_inventory_sha256=sha(inventory_path), input_files=len(inventory),
        source_hashes={str(p): sha(ROOT / p) for p in [Path("configs/p1_runs.csv"), Path("configs/p1.yaml"),
                       Path("docs/P1_ANALYSIS_PLAN.md"), Path("env/requirements.lock"),
                       Path("neuromri/common.py"), Path("neuromri/p1_finetune.py"),
                       Path("tools/p1_integrity_gate.py"), Path("tools/p1_attestation.py"),
                       Path("docs/P1_PROVENANCE_ATTESTATION.md")]},
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                         packages=versions, git_commit=subprocess.check_output(
                             ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()),
        downstream_analysis_run=False, fits=fits)
    (output / "integrity_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-commit", default=PRODUCTION_COMMIT)
    parser.add_argument("--reported-metrics-count", type=int, default=1400)
    parser.add_argument("--attestation", type=Path, help="Original P1 scheduler/reflog attestation directory")
    args = parser.parse_args()
    result = audit(args.run_root, args.output, args.production_commit, args.reported_metrics_count, args.attestation)
    print(json.dumps({k: result[k] for k in ["status", "planned_fits", "observed_metric_file_count",
                     "metric_rows", "eligible_fits", "failure_counts", "input_files", "input_inventory_sha256"]}, indent=2))
    sys.exit(1 if result["status"] == "FAIL" else 0)
