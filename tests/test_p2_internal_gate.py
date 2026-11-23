"""Fail-closed P2 internal attestation checks; synthetic evidence, no endpoints."""
import pytest

from tools import p2_internal_gate as gate
from tools.attestation_common import apply_exact, parse_reflog

GOOD = (f"HEAD {gate.EXECUTED} tracked_dirty_lines=0\n"
        f"{gate.EXECUTED} HEAD@{{2026-11-20T08:26:14+01:00}} merge {gate.EXECUTED}: Fast-forward\n"
        f"{gate.PRODUCTION} HEAD@{{2026-11-19T19:10:08+01:00}} pull -q bundle main: Fast-forward\n"
        f"{gate.JOB}|COMPLETED|0:0|2026-11-20T08:26:27|2026-11-20T08:26:27|2026-11-20T08:29:49|node01|{gate.WORKDIR}\n")


def run(tmp_path, text, job=gate.JOB):
    path = tmp_path / "attestation.txt"
    path.write_text(text, encoding="utf-8")
    results = {}
    gate.verify_attestation(path, {"slurm_job": job}, results)
    return {k: v["status"] for k, v in results.items() if not k.startswith("_")}


def test_valid_attestation_passes(tmp_path):
    assert set(run(tmp_path, GOOD).values()) == {"PASS"}


@pytest.mark.parametrize("old,new,failed", [
    ("2026-11-20T08:26:27|2026-11-20T08:26:27", "2026-11-20T08:26:10|2026-11-20T08:26:27", "scheduler_window_inside_reflog"),
    (gate.WORKDIR + "\n", "/elsewhere\n", "scheduler_state"),
    ("|COMPLETED|", "|FAILED|", "scheduler_state"),
    ("|0:0|", "|1:0|", "scheduler_state"),
    ("tracked_dirty_lines=0", "tracked_dirty_lines=2", "attested_tracked_clean"),
    (f"HEAD {gate.EXECUTED} tracked", f"HEAD {'b' * 40} tracked", "attested_head"),
    (f"{gate.PRODUCTION} HEAD@", f"{'c' * 40} HEAD@", "reflog_adjacent_heads"),
])
def test_broken_evidence_fails(tmp_path, old, new, failed):
    assert run(tmp_path, GOOD.replace(old, new, 1))[failed] == "FAIL"


def test_job_identity_must_match_provenance(tmp_path):
    assert run(tmp_path, GOOD, job="123")["provenance_job_matches_scheduler"] == "FAIL"


def test_reflog_order_and_hunks_fail_closed():
    with pytest.raises(ValueError):
        parse_reflog([f"{gate.PRODUCTION} HEAD@{{2026-11-19T19:10:08+01:00}} x",
                      f"{gate.EXECUTED} HEAD@{{2026-11-20T08:26:14+01:00}} y"])
    with pytest.raises(ValueError):
        apply_exact("a\nb\n", [("missing", "")], "synthetic")


def test_identity_hash_ignores_paths_but_not_protocol():
    config = dict(scope="vega", draws=20, budgets=[0, 16, 32, 64], data_root="/a", embeddings="x", output="o")
    moved = dict(config, data_root="/b", embeddings="y", output="p")
    assert gate.config_sha(config) == gate.config_sha(moved)
    assert gate.config_sha(config) != gate.config_sha(dict(config, draws=19))


