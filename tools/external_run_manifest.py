"""Hash-locked Vega input staging and final results manifest (standard library only)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f'Input changed since login staging: {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['stage', 'verify', 'submitted', 'complete'])
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--raw-root', type=Path)
    p.add_argument('--sif', type=Path)
    p.add_argument('--external-sif', type=Path)
    p.add_argument('--weights', type=Path)
    p.add_argument('--tools', type=Path)
    p.add_argument('--jobs', nargs='*')
    p.add_argument('--git-commit', help='Host git revision; avoids requiring git inside the SIF')
    p.add_argument('--tracked-list', type=Path, help='Host git ls-files -z output after a clean-check')
    a = p.parse_args()
    target = a.run/'RESULTS_MANIFEST.json'
    if a.action == 'stage':
        if target.exists():
            raise FileExistsError(f'Choose a fresh run directory: {target}')
        if not a.tracked_list and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
            raise ValueError('Commit code/manifests before Vega staging')
        files = {}
        for row in csv.DictReader(Path('manifests/external/downloads.csv').open()):
            path = a.raw_root/row['file']
            if path.stat().st_size != int(row['bytes']) or sha(path) != row['sha256']:
                raise ValueError(f'Raw archive byte count/SHA256 mismatch: {path}')
            files[str(path.resolve())] = row['sha256']
        tracked = (a.tracked_list.read_bytes() if a.tracked_list else
                   subprocess.check_output(['git', 'ls-files', '-z'])).decode().split('\0')
        paths = [Path(path) for path in tracked if path]
        paths += [a.sif, a.external_sif]
        if (a.run/'run.env').exists():
            paths.append(a.run/'run.env')
        if a.tracked_list:
            paths.append(a.tracked_list)
        if Path('slurm/hpc.local.env').exists():
            paths.append(Path('slurm/hpc.local.env'))
        paths += [p for root in [a.weights, a.tools] for p in root.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
        files.update({str(path.resolve()): sha(path) for path in paths})
        record = dict(state='staged', git_commit=a.git_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            main_sif_sha256=sha(a.sif), preprocessing_sif_sha256=sha(a.external_sif),
            encoders=['resnet18', 'resnet50', 'dinov2_s', 'biomedclip'], inputs=files,
            claim='Vega results valid only after state=complete; source caches recomputed for this SIF')
    else:
        record = json.loads(target.read_text())
        check(record['inputs'])
        if a.action == 'verify':
            print('Staged inputs unchanged')
            return
        if a.action == 'submitted':
            record.update(state='submitted', slurm_jobs=a.jobs)
        else:
            for required in ['probes/E1_metrics.csv', 'probes/E2_metrics.csv', 'probes/E3_metrics.csv',
                             'probes/E4_metrics.csv', 'probes/E4_tests.csv', 'probes/ad_roles.csv',
                             'probes/diagnostics/domain_metrics.csv', 'probes/diagnostics/intensity_distances.csv',
                             'probes/ixi_age_matched_roles.csv', 'probes/cohort_demographics.csv']:
                if not (a.run/required).is_file():
                    raise FileNotFoundError(a.run/required)
            record['outputs'] = {str(path.relative_to(a.run)): sha(path) for path in sorted(a.run.rglob('*'))
                                 if path.is_file() and path != target}
            record['state'] = 'complete'
    a.run.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(target)
    print(target, record['state'])


if __name__ == '__main__':
    main()
