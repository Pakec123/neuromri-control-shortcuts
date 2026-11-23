"""Read-only scheduler/reflog attestation for the original P1 export only."""
import csv
import os
from datetime import datetime
import hashlib
from pathlib import Path
import re
import subprocess
import tarfile
from zoneinfo import ZoneInfo

from tools.attestation_common import parse_reflog

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "469ee8f1aa6f5bbacaf6dc3d078b9920a01b5941"
NEXT = "ada9d384411f23bddac563532763c8f31a868413"
TAR_SHA = "aa1d1ec1bff803852a3cffa1c3dfc6766fac608df75614171042b3e4faf4c345"
WORKDIR = os.environ.get("NEUROMRI_ORIGINAL_INTERNAL_WORKDIR", "<SET_ORIGINAL_WORKDIR>")
MISSING_GIT = "[Errno 2] No such file or directory: 'git'"
RELEVANT = ["neuromri/__init__.py", "neuromri/common.py", "neuromri/data.py",
            "neuromri/models.py", "neuromri/statistics.py", "neuromri/p1_finetune.py",
            "neuromri/run_row.py", "neuromri/make_runlist.py", "configs/p1.yaml",
            "configs/p1_runs.csv", "slurm", "env", "manifests", "weights/manifest.yaml"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_attestation(directory, runlist, records, commit=COMMIT):
    """Fail closed; records are in committed runlist order. No endpoints are read."""
    directory = Path(directory).resolve()
    archive = directory.parent / "p1_attestation.tar"
    if commit != COMMIT or sha(archive) != TAR_SHA:
        raise ValueError("Attestation production commit or archive identity differs")
    names = ["reflog.txt", "sacct_p1.tsv", "p1_relevant_changes.txt",
             "diff_469ee8f_ada9d38.txt", "commit_469ee8f.txt", "last_end.txt"]
    with tarfile.open(archive) as tar:
        for name in names:
            members = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == name]
            if len(members) != 1 or tar.extractfile(members[0]).read() != (directory / name).read_bytes():
                raise ValueError("Attestation file differs from archived bytes: " + name)
    # Shared parser (tools/attestation_common.py): same pattern and strict ordering.
    entries = parse_reflog((directory / "reflog.txt").read_text().splitlines())
    indices = [i for i, (head, _) in enumerate(entries) if head == COMMIT]
    if len(indices) != 1 or indices[0] == 0 or entries[indices[0] - 1][0] != NEXT:
        raise ValueError("Expected adjacent production/next HEAD reflog entries")
    lo, hi = entries[indices[0]][1], entries[indices[0] - 1][1]
    local_commit = subprocess.check_output(
        ["git", "show", "-s", "--format=%H %T %cI", COMMIT], cwd=Path(os.environ.get("NEUROMRI_ATTESTATION_REPO", ROOT)), text=True).strip()
    if (directory / "commit_469ee8f.txt").read_text().strip() != local_commit:
        raise ValueError("Production commit/tree identity differs")
    changes = subprocess.check_output(
        ["git", "diff", "--name-only", COMMIT, NEXT, "--", *RELEVANT], cwd=Path(os.environ.get("NEUROMRI_ATTESTATION_REPO", ROOT)), text=True)
    if changes.strip() or (directory / "p1_relevant_changes.txt").read_text().strip():
        raise ValueError("Nonempty P1-relevant diff to next HEAD")
    # Also bind the locally analysed P1 training/config/input bytes to production.
    if subprocess.check_output(["git", "diff", COMMIT, "--", *RELEVANT], cwd=Path(os.environ.get("NEUROMRI_ATTESTATION_REPO", ROOT))):
        raise ValueError("Local P1 training/config/input bytes differ from production")
    tasks = list(csv.reader((directory / "sacct_p1.tsv").read_text().splitlines(), delimiter="|"))
    expected = {"50548178_0"} | {f"50548180_{i}" for i in range(1, 338)}
    if len(tasks) != 338 or any(len(t) != 8 for t in tasks) or {t[0] for t in tasks} != expected:
        raise ValueError("Scheduler task grid differs")
    zone = ZoneInfo("Europe/Ljubljana")
    def dt(value):
        return datetime.fromisoformat(value).replace(tzinfo=zone)
    for task, state, exitcode, submit, start, end, node, workdir in tasks:
        if state != "COMPLETED" or exitcode != "0:0" or not node or workdir != WORKDIR:
            raise ValueError("Scheduler state/exit/node/unique WorkDir failure: " + task)
        if not lo < dt(submit) <= dt(start) <= dt(end) < hi:
            raise ValueError("Scheduler window is outside production reflog window: " + task)
    last = max(dt(t[5]) for t in tasks)
    if dt((directory / "last_end.txt").read_text().strip()) != last:
        raise ValueError("last_end differs from scheduler accounting")
    if len(runlist) != 1350 or len(records) != 1350:
        raise ValueError("Attestation requires all 1350 fits")
    # Four adjacent immutable runlist rows per array task, two in the final task.
    # sacct JobID is array notation; provenance stores numeric SLURM_JOB_ID.
    jobs = []
    for i in range(0, 1350, 4):
        block = records[i:i + 4]
        ids = {r.get("slurm_job") for r in block}
        if len(ids) != 1 or not re.fullmatch(r"[0-9]+", str(next(iter(ids)))):
            raise ValueError("Fit SLURM identity differs within runlist task block")
        jobs.append(str(next(iter(ids))))
    if len(set(jobs)) != 338 or jobs[0] != "50548178" or jobs[-1] != "50548180":
        raise ValueError("Export task identities/count differ")
    return dict(status="PASS", method="scheduler_reflog", production_commit=COMMIT,
                next_head=NEXT, reflog_window=[lo.isoformat(), hi.isoformat()],
                first_submit=min(t[3] for t in tasks), last_end=last.isoformat(),
                tasks=len(tasks), workdir=WORKDIR, p1_relevant_paths=RELEVANT,
                p1_relevant_diff=[], tracked_checkout="attested (tracked files), not recorded per fit",
                operational_attestation_source="User-supplied Claude operational record, 2026-11-20; see docs/P1_PROVENANCE_ATTESTATION.md",
                numeric_job_array_mapping="Inferred from immutable four-row runlist dispatch; raw numeric JobID mapping was not exported in sacct",
                archive_sha256=sha(archive), directory=str(directory),
                files={name: sha(directory / name) for name in names})


def accepts_missing_git(record, attestation):
    return bool(attestation and attestation.get("status") == "PASS"
                and record.get("git_commit") == record.get("git_dirty") == MISSING_GIT)
