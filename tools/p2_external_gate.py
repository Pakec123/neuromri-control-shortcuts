"""Fail-closed identity gate for the local final external Vega export.

Reads file bytes, provenance and attestation only; no scientific endpoints.
The source fingerprint function is taken verbatim from a clean 808ca5e checkout.
"""
import argparse
import os
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.attestation_common import MISSING_GIT, git, scheduler_time, sha

COMMIT = '808ca5e4b88ff1eee6699baad655a954d112987f'
JOBS = ['50584117', '50584118', '50584119']
WORKDIR = os.environ.get("NEUROMRI_ORIGINAL_EXTERNAL_WORKDIR", "<SET_ORIGINAL_WORKDIR>")
TAR_SHA = '711a59f358650e693af8f2edf162802a99f17afd7c971908b59efc198f978730'
MANIFEST_SHA = 'f02dd6d21f45a2de7498519a7dd8aff69d5cb23c917c0b6c4bec987ab13a1dcd'
AMENDMENT = ROOT/'docs/P2_EXTERNAL_PROVENANCE_ATTESTATION.md'


def verified_volumes(check, outputs, manifest_sha):
    """Validate remote hash evidence under the PI's 21 Nov pre-endpoint rule."""
    lines = check.read_text(encoding='utf-8').splitlines()
    if not lines:
        raise ValueError('Empty volume check')
    footer = re.fullmatch(r'# (\S+) volumes listed=(\d+) mismatched=0 '
        r'\(checked on Vega login node by Claude; RESULTS_MANIFEST sha256=([0-9a-f]{64})\)', lines[-1])
    if not footer or footer[3] != manifest_sha:
        raise ValueError('Volume check footer/manifest mismatch')
    datetime.fromisoformat(footer[1])
    names = []
    for line in lines[:-1]:
        match = re.fullmatch(r'OK  (volumes/[^\\]+)', line)
        if not match or any(x in ('', '.', '..') for x in match[1].split('/')):
            raise ValueError('Invalid or non-OK volume check entry')
        names.append(match[1])
    expected = {name for name in outputs if name.startswith('volumes/')}
    if len(names) != len(set(names)) or len(names) != int(footer[2]) or set(names) != expected:
        raise ValueError('Volume check inventory mismatch')
    return set(names)


class OutputVerificationError(ValueError):
    def __init__(self, inventory, failures):
        super().__init__(f'{len(failures)} manifest outputs failed verification; first: {failures[0]}')
        self.inventory, self.failures = inventory, failures


def verify_outputs(root, outputs, remote_volumes=frozenset()):
    """Require local hash equality, or validated remote evidence for absent volumes."""
    inventory, failures = [], []
    if not outputs:
        raise ValueError('Empty output inventory')
    for name, expected in outputs.items():
        path = (root/name).resolve()
        safe = path.is_relative_to(root.resolve()) and not any(x in ('', '.', '..') for x in name.split('/')) and '\\' not in name
        if safe and not path.exists() and name.startswith('volumes/') and name in remote_volumes:
            inventory.append(dict(path=name, status='REMOTE_VOLUME_OK', expected_sha256=expected))
            continue
        if not safe or not path.is_file():
            failures.append('Missing or unsafe manifest output: '+name)
            inventory.append(dict(path=name, status='MISSING_OR_UNSAFE', expected_sha256=expected))
            continue
        actual = sha(path)
        if actual != expected:
            failures.append('Output hash mismatch: '+name)
        inventory.append(dict(path=name, bytes=path.stat().st_size, sha256=actual,
                              expected_sha256=expected, status='PASS' if actual == expected else 'HASH_MISMATCH'))
    if failures:
        raise OutputVerificationError(inventory, failures)
    return inventory


def source_identity(checkout):
    if checkout.exists():
        raise ValueError('Clean checkout destination must not exist')
    subprocess.run(['git', 'clone', '--shared', '--no-checkout', os.environ.get('NEUROMRI_ATTESTATION_REPO', str(ROOT)), str(checkout)], check=True,
                   capture_output=True)
    subprocess.run(['git', '-c', 'core.autocrlf=false', 'checkout', '--detach', COMMIT], cwd=checkout,
                   check=True, capture_output=True)
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=checkout)
    if dirty:
        raise ValueError('Fingerprint checkout is dirty')
    source = (checkout/'neuromri/common.py').read_text(encoding='utf-8')
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                    and n.name == 'source_fingerprint')
    namespace = dict(hashlib=hashlib, REPO_ROOT=checkout)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(checkout/'neuromri/common.py'), 'exec'), namespace)
    return namespace['source_fingerprint'](checkout)


def scheduler_identity(text):
    collected = datetime.fromisoformat(re.search(r'collected (\S+) by', text)[1])
    if re.search(r'## clone HEAD\s+(\w+)', text)[1] != COMMIT:
        raise ValueError('Attested HEAD mismatch')
    if re.search(r'## tracked-dirty count\s+(\d+)', text)[1] != '0':
        raise ValueError('Attested tracked checkout is dirty')
    reflog = text.split('## reflog\n')[1].split('## sacct')[0].strip().splitlines()
    entries = []
    for line in reflog:
        match = re.fullmatch(r'([0-9a-f]{7,40}) HEAD@\{([^}]+)\}: .+', line)
        if not match:
            raise ValueError('Malformed external reflog')
        entries.append((git('rev-parse', match[1]).strip(), datetime.fromisoformat(match[2])))
    if entries[0][0] != COMMIT or any(a[1] <= b[1] for a, b in zip(entries, entries[1:])):
        raise ValueError('Invalid HEAD interval')
    rows = {}
    for line in text.split('## sacct\n')[1].split('## provenance')[0].splitlines():
        fields = line.split('|')
        if fields[0] in JOBS:
            if fields[0] in rows:
                raise ValueError('Duplicate scheduler job')
            rows[fields[0]] = fields
    if set(rows) != set(JOBS):
        raise ValueError('Incomplete scheduler chain')
    previous_end = None
    for job in JOBS:
        fields = rows[job]
        submit, start, end = map(scheduler_time, fields[4:7])
        if not (fields[2:4] == ['COMPLETED', '0:0'] and fields[-1] == WORKDIR
                and fields[8] not in ('', 'None assigned')
                and entries[0][1] <= submit <= start <= end <= collected
                and (previous_end is None or previous_end <= start)):
            raise ValueError('Scheduler/HEAD window mismatch: '+job)
        previous_end = end
    return dict(head_start=entries[0][1].isoformat(), collected=collected.isoformat(), jobs=rows)


def audit(root, output):
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Gate output must be outside read-only inputs')
    output.mkdir(parents=True, exist_ok=True)
    result = dict(gate='p2_external', status='FAIL', failures=[], input_root=str(root),
                  executed_commit=COMMIT, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  endpoints_read=False)
    try:
        manifest = root/'RESULTS_MANIFEST.json'
        manifest_sha = sha(manifest)
        if manifest_sha != MANIFEST_SHA:
            raise ValueError('Manifest differs from PI-approved volume-check identity')
        record = json.loads(manifest.read_text(encoding='utf-8'))
        if (record.get('state') != 'complete' or record.get('git_commit') != COMMIT
                or set(record.get('encoders', [])) != {'resnet18', 'resnet50', 'dinov2_s', 'biomedclip'}):
            raise ValueError('Manifest state/commit/encoder identity mismatch')
        check = root/'attestation/volumes_hash_check.txt'
        remote_volumes = verified_volumes(check, record['outputs'], manifest_sha)
        result['acceptance_amendment'] = dict(date='2026-11-21', before_external_endpoints=True,
            document=str(AMENDMENT), sha256=sha(AMENDMENT), volume_check=str(check),
            volume_check_sha256=sha(check), manifest_sha256=manifest_sha, listed=len(remote_volumes))
        try:
            inventory = verify_outputs(root, record['outputs'], remote_volumes)
        except OutputVerificationError as exc:
            inventory = exc.inventory
            result['failures'].append(str(exc))
        result.update(manifest_sha256=sha(manifest), output_files=len(inventory))
        (output/'input_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
        result['input_inventory_sha256'] = sha(output/'input_inventory.json')
        result['output_status_counts'] = {status: sum(r['status'] == status for r in inventory)
                                          for status in ['PASS', 'REMOTE_VOLUME_OK', 'MISSING_OR_UNSAFE', 'HASH_MISMATCH']}
        archive = root.parent/'p2_external_808ca5e.tar'
        transfer = root.parent/'TRANSFER_SHA256.txt'
        if TAR_SHA not in transfer.read_text() or sha(archive) != TAR_SHA:
            raise ValueError('Transfer archive hash mismatch')
        # Verify manifest and every manifested member against the actual transfer.
        with tarfile.open(archive) as tar:
            members = {m.name.removeprefix('./'): m for m in tar.getmembers() if m.isfile()}
            archive_missing, archive_mismatches, archive_remote = [], [], []
            for name, expected in {**record['outputs'], 'RESULTS_MANIFEST.json': sha(manifest)}.items():
                matches = [m for key, m in members.items() if key == name or key.endswith('/'+name)]
                if len(matches) != 1:
                    if not matches and name in remote_volumes:
                        archive_remote.append(name)
                        continue
                    archive_missing.append(name)
                    continue
                h = hashlib.sha256()
                with tar.extractfile(matches[0]) as stream:
                    for block in iter(lambda: stream.read(8*1024*1024), b''):
                        h.update(block)
                if h.hexdigest() != expected:
                    archive_mismatches.append(name)
            result['transfer_members'] = dict(missing=archive_missing, mismatches=archive_mismatches,
                remote_volume_ok=archive_remote,
                verified=len(record['outputs'])+1-len(archive_missing)-len(archive_mismatches)-len(archive_remote))
            if archive_missing or archive_mismatches:
                result['failures'].append(f'Transfer members: {len(archive_missing)} missing, '
                                          f'{len(archive_mismatches)} hash mismatches')
        result['transfer_tar_sha256'] = TAR_SHA
        fingerprint = source_identity(output/'clean-808ca5e')
        result['source_fingerprint'] = fingerprint
        attestation = root/'attestation/p2_external_attestation.txt'
        window = scheduler_identity(attestation.read_text(encoding='utf-8'))
        result.update(attestation=str(attestation), attestation_sha256=sha(attestation), scheduler=window)
        provenances = [p for p in record['outputs'] if p.endswith('/provenance.json')]
        if len(provenances) != 11:
            raise ValueError('Expected 11 stage/encoder provenance records')
        for name in provenances:
            p = json.loads((root/name).read_text(encoding='utf-8'))
            job = '50584117' if name.startswith('processed/') else ('50584118' if name.startswith('cache/') else '50584119')
            container = record['preprocessing_sif_sha256'] if job == '50584117' else record['main_sif_sha256']
            if not (p.get('git_commit_files') == COMMIT and p.get('source_fingerprint') == fingerprint
                    and p.get('git_commit') == MISSING_GIT and str(p.get('slurm_job')) == job
                    and p.get('result_scope') == 'vega' and p.get('container_sha256') == container
                    and (job != '50584118' or 'A100' in p.get('gpu', ''))):
                raise ValueError('Provenance identity mismatch: '+name)
        result.update(status='FAIL' if result['failures'] else 'PASS', provenance_files=len(provenances),
                      tracked_checkout='attested after completion; fingerprint recorded at each stage',
                      source_hashes={p: sha(ROOT/p) for p in ['tools/p2_external_gate.py', 'tools/attestation_common.py']})
    except Exception as exc:
        result['failures'].append(f'{type(exc).__name__}: {exc}')
    (output/'integrity_gate.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_root, args.output)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['status'] == 'PASS' else 1)
