"""Synthetic fail-closed checks for external identity evidence, no endpoints."""
import pytest

from tools.attestation_common import sha
from tools.p2_external_gate import verify_outputs, scheduler_identity


def test_volume_exception_is_manifest_bound_and_fail_closed(tmp_path):
    from tools.p2_external_gate import verified_volumes
    check = tmp_path/'check.txt'
    digest = 'a'*64
    outputs = {'volumes/cohort/a.nii': 'b'*64, 'processed/a.png': 'c'*64}
    text = ('OK  volumes/cohort/a.nii\n'
            '# 2026-11-21T21:11:50.995806 volumes listed=1 mismatched=0 '
            f'(checked on Vega login node by Claude; RESULTS_MANIFEST sha256={digest})\n')
    check.write_text(text)
    remote = verified_volumes(check, outputs, digest)
    assert verify_outputs(tmp_path, {'volumes/cohort/a.nii': 'b'*64}, remote)[0]['status'] == 'REMOTE_VOLUME_OK'
    # Exception cannot cover missing PNGs, traversal, or corrupt present volumes.
    for missing in ['processed/a.png', 'volumes/../a.nii']:
        with pytest.raises(ValueError):
            verify_outputs(tmp_path, {missing: 'b'*64}, {missing})
    path = tmp_path/'volumes/cohort/a.nii'
    path.parent.mkdir(parents=True)
    path.write_text('corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_outputs(tmp_path, {'volumes/cohort/a.nii': 'b'*64}, remote)
    for bad in [text.replace(digest, 'd'*64), text.replace('mismatched=0', 'mismatched=1'),
                text.replace('OK  ', 'FAIL  '), text.replace('listed=1', 'listed=2'),
                text.replace('OK  volumes/cohort/a.nii\n', ''),
                'OK  volumes/cohort/a.nii\n'+text,
                text.replace('volumes/cohort/a.nii', 'processed/a.png'),
                text.replace('volumes/cohort/a.nii', 'volumes/../a.nii')]:
        check.write_text(bad)
        with pytest.raises(ValueError):
            verified_volumes(check, outputs, digest)


def test_every_manifest_output_is_required_and_hashed(tmp_path):
    file = tmp_path/'result.csv'
    file.write_text('identity\nexample\n')
    expected = {'result.csv': sha(file)}
    assert len(verify_outputs(tmp_path, expected)) == 1
    file.write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_outputs(tmp_path, expected)
    file.unlink()
    with pytest.raises(ValueError, match='Missing or unsafe'):
        verify_outputs(tmp_path, expected)
    with pytest.raises(ValueError, match='Missing or unsafe'):
        verify_outputs(tmp_path, {'../escape': '0'*64})


def test_scheduler_rejects_dirty_or_wrong_head():
    from tools.p2_external_gate import COMMIT
    template = ('# collected 2026-11-21T21:04:06+01:00 by auditor\n'
                '## clone HEAD\n{head}\n## tracked-dirty count\n{dirty}\n')
    with pytest.raises(ValueError, match='HEAD mismatch'):
        scheduler_identity(template.format(head='0'*40, dirty=0))
    with pytest.raises(ValueError, match='dirty'):
        scheduler_identity(template.format(head=COMMIT, dirty=1))


def test_scheduler_chain_must_fit_head_interval_and_be_sequential(monkeypatch):
    import tools.p2_external_gate as gate
    monkeypatch.setattr(gate, 'git', lambda *args: gate.COMMIT if args[-1] == '808ca5e' else 'a'*40)
    text = ('# collected 2026-11-21T21:04:06+01:00 by auditor\n'
            f'## clone HEAD\n{gate.COMMIT}\n## tracked-dirty count\n0\n## reflog\n'
            '808ca5e HEAD@{2026-11-20T08:30:08+01:00}: merge: Fast-forward\n'
            'aaaaaaa HEAD@{2026-11-20T08:26:15+01:00}: merge: Fast-forward\n## sacct\n')
    for job, start, end in [('50584117', '10:32:26', '17:18:48'),
                             ('50584118', '18:01:23', '19:10:44'),
                             ('50584119', '20:10:53', '21:02:31')]:
        text += (f'{job}|stage|COMPLETED|0:0|2026-11-20T10:30:59|2026-11-20T{start}|'
                 f'2026-11-20T{end}|00:00:01|node01|cpu=4|{gate.WORKDIR}\n')
    text += '## provenance\n'
    assert set(scheduler_identity(text)['jobs']) == set(gate.JOBS)
    for bad in [text.replace('2026-11-20T10:30:59', '2026-11-20T08:20:00', 1),
                text.replace('2026-11-20T18:01:23', '2026-11-20T16:01:23'),
                text.replace('|COMPLETED|0:0|', '|FAILED|1:0|', 1),
                text.replace(gate.WORKDIR, '/other/clone', 1)]:
        with pytest.raises(ValueError, match='Scheduler/HEAD window mismatch'):
            scheduler_identity(bad)
