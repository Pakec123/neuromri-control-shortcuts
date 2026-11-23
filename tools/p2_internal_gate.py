"""Integrity gate for the read-only P2 internal Vega export (SLURM job 50579635).

Generalises the P1 gate pattern (tools/p1_integrity_gate.py, tools/p1_attestation.py)
and the dated amendment docs/P2_INTERNAL_PROVENANCE_ATTESTATION.md. It reads only
provenance, scheduler/reflog text, git objects, file bytes and identity columns;
no endpoint column (AUROC, sensitivity, specificity, ...) is read. A nonzero exit
status blocks every downstream P2 internal analysis.
"""
import argparse
import os
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib.metadata
from itertools import product
import json
from pathlib import Path
import platform
import re
import sys
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.attestation_common import (MISSING_GIT, apply_exact, changed_paths, git, parse_reflog,
                                      scheduler_time, sha, show)

EXECUTED = "ada9d384411f23bddac563532763c8f31a868413"    # production clone HEAD during the job
PRODUCTION = "469ee8f1aa6f5bbacaf6dc3d078b9920a01b5941"  # previous HEAD (P1 production commit)
PLAN_COMMIT = "fc079823f7c8761cfb9b98ad46adedfaa4e4143a"  # last change of docs/P2_RESULTS_PLAN.md
TAR_SHA = "4051b422b8f6ad9779fac775f4df130517438c49784a8ead7e07cf8f3d4402b1"
JOB = "50579635"
WORKDIR = os.environ.get("NEUROMRI_ORIGINAL_INTERNAL_WORKDIR", "<SET_ORIGINAL_WORKDIR>")
CONTAINER = "291ced61377a23dd27bc1d0b060ae1e882eb2c1db1303cd2ae295a8afdccd944"
FILES = ["metrics.csv", "predictions.csv", "image_predictions.csv", "provenance.json", "pip-freeze.txt"]
# Everything the internal P2 job imports, configures, mounts or analyses.
RELEVANT = ["neuromri/__init__.py", "neuromri/common.py", "neuromri/data.py", "neuromri/statistics.py",
            "neuromri/p2_threshold_transport.py", "neuromri/verify_data.py", "neuromri/aggregate.py",
            "neuromri/extract_embeddings.py", "neuromri/models.py", "neuromri/fetch_weights.py",
            "configs/p2.yaml", "manifests/images.csv", "manifests/roles.csv", "manifests/folds.csv",
            "manifests/checksums.json", "manifests/crossed_design.csv", "manifests/provenance.json",
            "weights/manifest.yaml", "slurm/p2_cpu.sbatch", "slurm/extract.sbatch", "slurm/runtime.sh",
            "slurm/stage_data.sh", "slurm/submit.sh", "slurm/hpc.env", "env/neuromri.def",
            "env/requirements.lock", "env/build_sif.sh", "docs/P2_RESULTS_PLAN.md", "pyproject.toml"]
JOBLIB_OLD = '    run(config_cli("configs/p2.yaml"))\n'
JOBLIB_NEW = ('    # Call through the package module so joblib/loky workers unpickle fit functions by\n'
              '    # module reference; pickling them from __main__ breaks on scipy ufuncs (e.g. expit).\n'
              '    from neuromri.p2_threshold_transport import run as _run\n'
              '    _run(config_cli("configs/p2.yaml"))\n')
CONFIG_HASH_FN = ('PATH_KEYS = {"data_root", "weights_dir", "output", "embeddings"}\n\n\n'
                  'def config_hash(c):\n'
                  '    """Hash of the experiment definition only: machine/run-specific paths are excluded,\n'
                  '    so a row keeps its identity across nodes (node-local data staging) and output dirs.\n'
                  '    The full config, paths included, is still stored in provenance.json."""\n'
                  '    return canonical_hash({k: v for k, v in c.items()\n'
                  '                           if k not in PATH_KEYS and not k.endswith(("_root", "_dir"))})\n\n\n')
TRAP = "trap 'rm -rf \"/tmp/${USER}/nmri-${SLURM_JOB_ID}\"' EXIT\n"
STAGE = ('# Node-local copy of the dataset (see stage_data.sh); falls back to DATA_ROOT on shared storage.\n'
         'if [[ -n "${DATA_TAR:-}" && -n "${SLURM_JOB_ID:-}" ]]; then source slurm/stage_data.sh; fi\n')
# Plan commit -> production commit: non-analytic infrastructure hunks, each removed exactly once.
INFRASTRUCTURE = {
    "neuromri/common.py": [(CONFIG_HASH_FN, ""), ("config_sha256=config_hash(c)", "config_sha256=canonical_hash(c)")],
    "neuromri/p2_threshold_transport.py": [
        ("canonical_hash, config_hash\n", "canonical_hash\n"),
        ("n_target_pool=len(pool),config_hash=config_hash(c))", "n_target_pool=len(pool),config_hash=canonical_hash(c))")],
    "slurm/p2_cpu.sbatch": [(TRAP, "")],
    "slurm/extract.sbatch": [(TRAP, "")],
    "slurm/runtime.sh": [(STAGE, "")],
}
PATH_KEYS = {"data_root", "weights_dir", "output", "embeddings"}
ENCODERS = ["resnet18", "resnet50", "dinov2_s", "biomedclip"]
TASKS = ["AD", "Tumour", "MS", "PD", "Stroke"]
STRATEGIES = ["fixed_0.5", "source_ba", "source_specificity90", "source_temperature"]
IDENTITY = ["encoder", "task", "fold", "seed", "train_cluster", "test_cluster", "strategy", "k", "draw"]


def definition(config):
    return {k: v for k, v in config.items() if k not in PATH_KEYS and not k.endswith(("_root", "_dir"))}


def config_sha(config):
    import hashlib
    return hashlib.sha256(json.dumps(definition(config), sort_keys=True, default=str).encode()).hexdigest()


def check(results, name, ok, detail=""):
    results[name] = dict(status="PASS" if ok else "FAIL", detail=detail)
    return ok


def verify_source_identity(results):
    """Executed commit = production + joblib fix; plan commit -> production = infrastructure only."""
    lines = git("show", "-s", "--format=%H %P", EXECUTED).split()
    check(results, "executed_commit_parentage", lines[0] == EXECUTED and len(lines) == 2,
          "ada9d38 is a single-parent commit present in local git")
    ancestors = git("rev-list", f"{PRODUCTION}..{EXECUTED}").split()
    check(results, "production_is_ancestor", EXECUTED in ancestors and
          git("merge-base", PRODUCTION, EXECUTED).strip() == PRODUCTION,
          f"{len(ancestors)} commits between production and executed HEAD")
    changed = changed_paths(PRODUCTION, EXECUTED, RELEVANT)
    ok = changed == ["neuromri/p2_threshold_transport.py"]
    detail = "changed: " + ", ".join(changed)
    if ok:
        old = show(PRODUCTION, "neuromri/p2_threshold_transport.py")
        new = show(EXECUTED, "neuromri/p2_threshold_transport.py")
        ok = apply_exact(new, [(JOBLIB_NEW, JOBLIB_OLD)], "executed p2_threshold_transport") == old
        detail += "; the only hunk is the __main__ joblib entry-point fix"
    check(results, "production_to_executed_only_joblib_fix", ok, detail)
    changed = changed_paths(PLAN_COMMIT, PRODUCTION, RELEVANT)
    ok = set(changed) == set(INFRASTRUCTURE) | {"slurm/stage_data.sh"}
    for path, hunks in INFRASTRUCTURE.items():
        if not ok:
            break
        ok = apply_exact(show(PRODUCTION, path), hunks, path) == show(PLAN_COMMIT, path)
    ok = ok and show(PLAN_COMMIT, "slurm/stage_data.sh") is None
    check(results, "plan_to_production_infrastructure_only", ok,
          "changed: " + ", ".join(changed) + "; path-free identity hash, node-local SHA-verified dataset "
          "staging (sourced only when DATA_TAR is set; the internal job reads embeddings, not images) and "
          "temporary-directory cleanup. No analysis line changed.")
    plan = [show(c, "docs/P2_RESULTS_PLAN.md") for c in [PLAN_COMMIT, PRODUCTION, EXECUTED]]
    local = (ROOT / "docs/P2_RESULTS_PLAN.md").read_text(encoding="utf-8")
    check(results, "results_plan_unchanged", plan[0] == plan[1] == plan[2] == local.replace("\r\n", "\n"),
          "docs/P2_RESULTS_PLAN.md identical at plan, production, executed and local commits")
    # Local CPU aggregation must use the executed analysis bytes.
    local_changed = changed_paths(EXECUTED, "HEAD", RELEVANT)
    ok = set(local_changed) <= {"neuromri/aggregate.py", "neuromri/p2_threshold_transport.py"}
    if ok and "neuromri/aggregate.py" in local_changed:
        hunk_old = '    provenance(vars(a),out,inputs)\n'
        hunk_new = ("    if a.source_integrity_gate:\n        if a.scope != 'vega':\n"
                    "            p.error('--source-integrity-gate requires Vega source scope')\n"
                    "        from tools.p1_postprocessing import provenance as local_provenance\n"
                    "        local_provenance(vars(a),out,inputs,a.source_integrity_gate)\n"
                    "    else:\n        provenance(vars(a),out,inputs)\n")
        arg = '    p.add_argument("--source-integrity-gate", help="Record local CPU reanalysis of a fully gated Vega export")\n'
        ok = apply_exact(show("HEAD", "neuromri/aggregate.py"), [(hunk_new, hunk_old), (arg, "")],
                         "local aggregate") == show(EXECUTED, "neuromri/aggregate.py")
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", *RELEVANT).strip()
    check(results, "local_analysis_bytes", ok and not dirty,
          "local differs from executed only by the gated local-provenance option of aggregate.py "
          "and the pre-fix __main__ entry point (not executed locally); tracked relevant files clean")


def verify_attestation(path, record, results):
    """Machine check of the plain-text attestation collected on Vega after the job."""
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    head = re.fullmatch(r"HEAD ([0-9a-f]{40}) tracked_dirty_lines=(\d+)", lines[0].strip())
    check(results, "attested_head", bool(head) and head[1] == EXECUTED, lines[0].strip())
    check(results, "attested_tracked_clean", bool(head) and head[2] == "0",
          "tracked-file status at collection time (after job end); not recorded at run time")
    reflog = parse_reflog([l for l in lines if "HEAD@{" in l])
    adjacent = len(reflog) >= 2 and reflog[0][0] == EXECUTED and reflog[1][0] == PRODUCTION
    check(results, "reflog_adjacent_heads", adjacent and bool(head) and head[1] == reflog[0][0],
          "current HEAD is the newest reflog entry; previous HEAD is the P1 production commit")
    tasks = [l.split("|") for l in lines if "|" in l]
    ok = len(tasks) == 1 and len(tasks[0]) == 8
    if ok:
        job, state, exitcode, submit, start, end, node, workdir = tasks[0]
        lo = reflog[0][1]
        ok_state = job == JOB and state == "COMPLETED" and exitcode == "0:0" and bool(node) and workdir == WORKDIR
        check(results, "scheduler_state", ok_state, f"{job} {state} {exitcode} {node} {workdir}")
        inside = lo < scheduler_time(submit) <= scheduler_time(start) <= scheduler_time(end)
        # The upper bound is the collection itself: sacct already reports End, and HEAD was
        # still the newest reflog entry when the attestation was collected.
        check(results, "scheduler_window_inside_reflog", inside,
              f"HEAD {EXECUTED[:7]} since {lo.isoformat()}; submit {submit}, start {start}, end {end}; "
              "unchanged at collection after End")
        check(results, "provenance_job_matches_scheduler", str(record.get("slurm_job")) == job)
        results["_scheduler"] = dict(job=job, submit=submit, start=start, end=end, node=node, workdir=workdir,
                                     reflog_window_start=lo.isoformat())
    else:
        check(results, "scheduler_state", False, "expected exactly one sacct row")


def verify_export(root, record, results, attested):
    config = record.get("config", {})
    check(results, "scope_vega", record.get("result_scope") == "vega" and config.get("scope") == "vega")
    check(results, "container_identity", record.get("container_sha256") == CONTAINER, CONTAINER)
    check(results, "missing_git_only_with_attestation",
          record.get("git_commit") == record.get("git_dirty") == MISSING_GIT and attested,
          "both original git fields hold the exact missing-git error; accepted only after attestation")
    check(results, "config_hash", record.get("config_sha256") == config_sha(config),
          "recomputed with the executed path-free identity rule")
    base = yaml.safe_load((ROOT / "configs/p2.yaml").read_text(encoding="utf-8"))
    expected = dict(base, jobs=16, threads_per_fit=1, output="results/p2")
    check(results, "protocol_config", definition(config) == definition(expected),
          "configs/p2.yaml with the sbatch overrides jobs=16 (16 CPUs), threads_per_fit=1, output=results/p2")
    frozen = {f"manifests/{n}": sha(ROOT / "manifests" / n) for n in ["images.csv", "roles.csv"]}
    artifacts = record.get("artifacts", {})
    check(results, "frozen_manifests", all(artifacts.get(k) == v for k, v in frozen.items()))
    caches = {k: v for k, v in artifacts.items() if k.startswith("cache/embeddings/")}
    check(results, "embedding_cache_identities",
          sorted(caches) == sorted(f"cache/embeddings/{e}/cache.json" for e in ENCODERS)
          and all(re.fullmatch("[0-9a-f]{64}", v) for v in caches.values()),
          "four recorded cache.json identities; cache bytes are not part of the export")
    # Identity columns only: grid completeness, scope and config identity.
    columns = IDENTITY + ["status", "scope", "config_hash"]
    counts, keys, statuses, scopes, hashes = Counter(), set(), Counter(), Counter(), Counter()
    with (root / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            key = tuple(row[c] for c in IDENTITY)
            counts["rows"] += 1
            keys.add(key)
            statuses[(row["strategy"], row["k"], row["status"])] += 1
            scopes[row["scope"]] += 1
            hashes[row["config_hash"]] += 1
    expected_keys = set()
    for e, t, f, s, tr, te in product(ENCODERS, TASKS, range(5), [5101, 5102, 5103], range(3), range(3)):
        for strategy in STRATEGIES:
            expected_keys.add((e, t, str(f), str(s), str(tr), str(te), strategy, "0", "-1"))
        for method, k, d in product(["target_quantile", "target_tolerance"], [0, 16, 32, 64], range(20)):
            expected_keys.add((e, t, str(f), str(s), str(tr), str(te), method, str(k), str(d)))
    check(results, "metrics_grid", counts["rows"] == 442800 and keys == expected_keys,
          f"{counts['rows']} rows, {len(keys)} unique identities, expected 442800 (900 score fits)")
    check(results, "metrics_scope_and_hash", dict(scopes) == {"vega": 442800} and
          dict(hashes) == {record.get("config_sha256"): 442800})
    allowed = {"ok", "insufficient_target_groups", "bound_unavailable"}
    check(results, "status_vocabulary", {s for _, _, s in statuses} <= allowed,
          json.dumps(sorted({s for _, _, s in statuses})))
    for name in ["predictions.csv", "image_predictions.csv"]:
        units, scope, groups, duplicates = set(), Counter(), Counter(), 0
        group_key = "group_id" if name == "predictions.csv" else "image_id"
        with (root / name).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                unit = tuple(row[c] for c in ["encoder", "task", "fold", "seed", "train_cluster", "test_cluster"])
                units.add(unit)
                scope[row["scope"]] += 1
                groups[unit + (row[group_key],)] += 1
        duplicates = sum(v > 1 for v in groups.values())
        expected_units = {(e, t, str(f), str(s), str(tr), str(te)) for e, t, f, s, tr, te in
                          product(ENCODERS, TASKS, range(5), [5101, 5102, 5103], range(3), range(3))}
        check(results, name.replace(".csv", "_grid"), units == expected_units and set(scope) == {"vega"}
              and duplicates == 0, f"{len(units)} fit/test-cluster units, {sum(scope.values())} rows, "
              f"{duplicates} duplicated unit-level keys")
    return statuses


def audit(run_root, attestation, output, p1_root=None):
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    if output.is_relative_to(run_root):
        raise ValueError("Audit output must be outside the read-only input tree")
    results = {}
    archive = run_root.parents[1] / "p2_results.tar"
    ok = archive.is_file() and sha(archive) == TAR_SHA
    if ok:
        with tarfile.open(archive) as tar:
            members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
            ok = set(members) == set(FILES)
            for name in FILES if ok else []:
                import hashlib
                h = hashlib.sha256()
                stream = tar.extractfile(members[name])
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(block)
                ok = ok and h.hexdigest() == sha(run_root / name)
    check(results, "transfer_archive", ok, f"{archive.name} sha256 {TAR_SHA}; five members equal extracted bytes")
    record = json.loads((run_root / "provenance.json").read_text(encoding="utf-8"))
    try:
        verify_source_identity(results)
    except Exception as exc:  # fail closed, keep the reason
        check(results, "source_identity_error", False, f"{type(exc).__name__}: {exc}")
    try:
        verify_attestation(attestation, record, results)
    except Exception as exc:
        check(results, "attestation_error", False, f"{type(exc).__name__}: {exc}")
    attested = all(results[k]["status"] == "PASS" for k in results if not k.startswith("_"))
    statuses = verify_export(run_root, record, results, attested)
    if p1_root:
        p1 = sorted(Path(p1_root).glob("*/pip-freeze.txt"))
        check(results, "environment_matches_p1_fits", bool(p1) and sha(p1[0]) == sha(run_root / "pip-freeze.txt"),
              "pip-freeze identical to the P1 Vega fits run in the same SIF")
    failures = [k for k, v in results.items() if not k.startswith("_") and v["status"] != "PASS"]
    output.mkdir(parents=True, exist_ok=True)
    inventory = output / "input_inventory.csv"
    with inventory.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for name in FILES:
            writer.writerow(dict(path=name, bytes=(run_root / name).stat().st_size, sha256=sha(run_root / name)))
    versions = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "PyYAML"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    result = dict(gate="p2_internal", status="FAIL" if failures else "PASS", failures=failures,
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), input_root=str(run_root),
                  attestation=str(Path(attestation).resolve()), attestation_sha256=sha(attestation),
                  executed_commit=EXECUTED, production_commit=PRODUCTION, plan_commit=PLAN_COMMIT,
                  transfer_tar_sha256=TAR_SHA, planned_units=900, eligible_units=0 if failures else 900,
                  checks=results, availability={"|".join(k): v for k, v in sorted(statuses.items())},
                  tracked_checkout="attested (tracked files) at collection after the job, not recorded at run time",
                  provenance_values=dict(container_sha256={record.get("container_sha256"): 1},
                                         git_commit={record.get("git_commit"): 1}, slurm_job={record.get("slurm_job"): 1}),
                  input_inventory_sha256=sha(inventory), input_files=len(FILES),
                  source_hashes={p: sha(ROOT / p) for p in ["tools/p2_internal_gate.py", "tools/attestation_common.py",
                                 "docs/P2_INTERNAL_PROVENANCE_ATTESTATION.md", "docs/P2_RESULTS_PLAN.md",
                                 "configs/p2.yaml", "neuromri/aggregate.py"] if (ROOT / p).is_file()},
                  environment=dict(python=sys.version, platform=platform.platform(), packages=versions,
                                   git_commit=git("rev-parse", "HEAD").strip()),
                  endpoints_read=False, downstream_analysis_run=False)
    (output / "integrity_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p1-run-root", type=Path, help="P1 Vega export for the same-SIF environment comparison")
    args = parser.parse_args()
    result = audit(args.run_root, args.attestation, args.output, args.p1_run_root)
    print(json.dumps({k: result[k] for k in ["status", "failures", "eligible_units", "input_inventory_sha256"]}, indent=2))
    for name, value in result["checks"].items():
        if not name.startswith("_"):
            print(f"{value['status']:4}  {name}: {value['detail']}")
    sys.exit(1 if result["status"] == "FAIL" else 0)
