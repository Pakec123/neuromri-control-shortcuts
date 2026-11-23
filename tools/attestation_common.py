"""Shared fail-closed helpers for scheduler/reflog commit attestation.

Generalised from the P1 gate (tools/p1_attestation.py). Nothing here reads an
experimental endpoint: only provenance text, git objects and file bytes.
"""
from datetime import datetime
import hashlib
from pathlib import Path
import re
import os
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
MISSING_GIT = "[Errno 2] No such file or directory: 'git'"
ZONE = ZoneInfo("Europe/Ljubljana")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_reflog(lines):
    """Return [(commit, aware datetime)] and require strict reverse chronology."""
    entries = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{40}) HEAD@\{([^}]+)\} .+", line.strip())
        if not match:
            raise ValueError("Malformed reflog line")
        entries.append((match[1], datetime.fromisoformat(match[2])))
    if not entries:
        raise ValueError("Empty reflog")
    if any(a[1] <= b[1] for a, b in zip(entries, entries[1:])):
        raise ValueError("Reflog is not strictly reverse chronological")
    return entries


def scheduler_time(value):
    """sacct prints local wall time without an offset; Vega runs in Europe/Ljubljana."""
    return datetime.fromisoformat(value).replace(tzinfo=ZONE)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=Path(os.environ.get("NEUROMRI_ATTESTATION_REPO", ROOT)), text=True, encoding="utf-8")


def show(commit, path):
    """File text at a commit, or None when absent."""
    try:
        return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=Path(os.environ.get("NEUROMRI_ATTESTATION_REPO", ROOT)),
                                       stderr=subprocess.DEVNULL).decode("utf-8")
    except subprocess.CalledProcessError:
        return None


def changed_paths(base, head, paths):
    return sorted(p for p in git("diff", "--name-only", base, head, "--", *paths).splitlines() if p)


def apply_exact(text, replacements, label):
    """Apply each (old, new) exactly once; any other text difference stays visible."""
    if text is None:
        raise ValueError("Missing source text: " + label)
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f"Expected one occurrence of an allowlisted hunk in {label}")
        text = text.replace(old, new)
    return text
