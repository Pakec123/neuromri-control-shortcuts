"""Fail-closed provenance checks without reading experimental endpoints."""
import copy
import hashlib
import subprocess
import tarfile

import pytest
import yaml

from tools.p1_integrity_gate import ROOT, PRODUCTION_COMMIT, check_record, config_sha, rows
from tools import p1_attestation as attest


@pytest.fixture
def valid_record():
    base = yaml.safe_load((ROOT / "configs/p1.yaml").read_text())
    run = rows(ROOT / "configs/p1_runs.csv")[0]
    config = dict(base, alpha=0.0, output="results/p1/" + run["run_id"])
    record = dict(config=config, config_sha256=config_sha(config), result_scope="vega",
                  git_commit=PRODUCTION_COMMIT, git_dirty="", slurm_job="50548178",
                  container_sha256="a" * 64, gpu="NVIDIA A100-SXM4-40GB",
                  nvidia_smi="NVIDIA A100-SXM4-40GB, driver, UUID", torch="2.5.1+cu124", cuda="12.4")
    return record, run, base


def test_missing_git_is_not_accepted_as_a_clean_checkout(valid_record):
    record, run, base = valid_record
    assert check_record(record, run, base, PRODUCTION_COMMIT) == []
    record["git_commit"] = record["git_dirty"] = "[Errno 2] No such file or directory: 'git'"
    errors = check_record(record, run, base, PRODUCTION_COMMIT)
    assert "production_commit_unverified" in errors
    assert "clean_checkout_unverified" in errors
    assert any(e.startswith("existing_provenance_gate:") for e in errors)


def test_rehashed_protocol_tampering_and_stale_hash_both_fail(valid_record):
    record, run, base = valid_record
    record["config"]["epochs"] = 29
    assert "config_hash_mismatch" in check_record(record, run, base, PRODUCTION_COMMIT)
    record["config_sha256"] = config_sha(record["config"])
    errors = check_record(record, run, base, PRODUCTION_COMMIT)
    assert "config_hash_mismatch" not in errors
    assert "protocol_config_mismatch" in errors


def test_relocation_does_not_change_definition_but_wrong_run_id_fails(valid_record):
    record, run, base = valid_record
    relocated = copy.deepcopy(record)
    relocated["config"].update(data_root="C:/local/data", weights_dir="C:/local/weights",
                               output="C:/local/p1/" + run["run_id"])
    assert check_record(relocated, run, base, PRODUCTION_COMMIT) == []
    relocated["config"]["output"] = "C:/local/p1/another_run"
    assert "config_output_run_id_mismatch" in check_record(relocated, run, base, PRODUCTION_COMMIT)


@pytest.mark.parametrize("field,value", [("result_scope", "smoke"), ("gpu", "RTX 4090"),
                                         ("container_sha256", "LOCAL_SMOKE_NO_CONTAINER")])
def test_development_and_unidentified_inputs_fail(valid_record, field, value):
    record, run, base = valid_record
    record[field] = value
    assert any(e.startswith("existing_provenance_gate:")
               for e in check_record(record, run, base, PRODUCTION_COMMIT))


@pytest.fixture
def attestation_files(tmp_path, monkeypatch):
    folder = tmp_path / 'p1_attestation'
    folder.mkdir()
    text = {
        'reflog.txt': f'{attest.NEXT} HEAD@{{2026-11-20T08:26:14+01:00}} merge: Fast-forward\n'
                      f'{attest.COMMIT} HEAD@{{2026-11-19T19:10:08+01:00}} pull: Fast-forward\n',
        'commit_469ee8f.txt': attest.COMMIT + ' ' + 'b'*40 + ' 2026-11-19T19:10:08+01:00\n',
        'p1_relevant_changes.txt': '', 'diff_469ee8f_ada9d38.txt': 'non-P1 changes\n',
        'last_end.txt': '2026-11-20T08:20:06\n',
        'sacct_p1.tsv': '\n'.join(
            f'{"50548178_0" if i == 0 else "50548180_" + str(i)}|COMPLETED|0:0|'
            '2026-11-19T19:10:09|2026-11-20T00:06:02|2026-11-20T08:20:06|node01|'
            + attest.WORKDIR for i in range(338)) + '\n'}
    def write():
        for name, value in text.items():
            (folder / name).write_text(value, encoding='utf-8')
        archive = tmp_path / 'p1_attestation.tar'
        with tarfile.open(archive, 'w') as tar:
            for name in text:
                tar.add(folder / name, arcname='p1_attestation/' + name)
        monkeypatch.setattr(attest, 'TAR_SHA', hashlib.sha256(archive.read_bytes()).hexdigest())
    write()
    commit_text = text['commit_469ee8f.txt'].strip()
    def fake_git(args, **kwargs):
        return commit_text if args[1] == 'show' else ('' if kwargs.get('text') else b'')
    monkeypatch.setattr(attest.subprocess, 'check_output', fake_git)
    records = [dict(slurm_job='50548178' if i < 4 else '50548180' if i >= 1348
                    else str(60000000 + i // 4)) for i in range(1350)]
    return folder, text, write, records


def test_attestation_accepts_only_missing_git_and_preserves_other_checks(attestation_files, valid_record):
    folder, _, _, records = attestation_files
    evidence = attest.verify_attestation(folder, rows(ROOT / 'configs/p1_runs.csv'), records)
    record, run, base = valid_record
    record['git_commit'] = record['git_dirty'] = attest.MISSING_GIT
    original = copy.deepcopy(record)
    assert check_record(record, run, base, PRODUCTION_COMMIT, evidence) == []
    assert record == original
    record['gpu'] = 'RTX 4090'
    assert check_record(record, run, base, PRODUCTION_COMMIT, evidence)
    record['gpu'] = original['gpu']
    record['git_dirty'] = ' M neuromri/p1_finetune.py'
    assert 'clean_checkout_unverified' in check_record(record, run, base, PRODUCTION_COMMIT, evidence)


@pytest.mark.parametrize('failure', ['window', 'workdir', 'state', 'exit', 'missing_task',
                                    'diff', 'last_end', 'reflog', 'job_block', 'archive'])
def test_attestation_rejects_broken_evidence(attestation_files, failure):
    folder, text, write, records = attestation_files
    if failure == 'window': text['sacct_p1.tsv'] = text['sacct_p1.tsv'].replace('2026-11-19T19:10:09', '2026-11-19T19:10:07', 1)
    if failure == 'workdir': text['sacct_p1.tsv'] = text['sacct_p1.tsv'].replace(attest.WORKDIR, '/other', 1)
    if failure == 'state': text['sacct_p1.tsv'] = text['sacct_p1.tsv'].replace('COMPLETED', 'FAILED', 1)
    if failure == 'exit': text['sacct_p1.tsv'] = text['sacct_p1.tsv'].replace('0:0', '1:0', 1)
    if failure == 'missing_task': text['sacct_p1.tsv'] = '\n'.join(text['sacct_p1.tsv'].splitlines()[1:])
    if failure == 'diff': text['p1_relevant_changes.txt'] = 'neuromri/p1_finetune.py\n'
    if failure == 'last_end': text['last_end.txt'] = '2026-11-20T08:20:07\n'
    if failure == 'reflog': text['reflog.txt'] = text['reflog.txt'].replace(attest.NEXT, 'a' * 40)
    if failure == 'job_block': records[1]['slurm_job'] = '999'
    write()
    if failure == 'archive': (folder / 'last_end.txt').write_text('changed')
    with pytest.raises(ValueError):
        attest.verify_attestation(folder, rows(ROOT / 'configs/p1_runs.csv'), records)
