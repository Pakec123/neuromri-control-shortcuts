"""Run the paper's numerical postprocessing on fresh, completed HPC outputs.

Run inside a SLURM CPU allocation after sourcing slurm/runtime.sh. Historical
exports instead require the documented original attestation gates.
"""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.p1_analysis_commands import commands
from tools.p2_internal_fill import contrasts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--p1-root', type=Path, required=True)
    p.add_argument('--p2-root', type=Path, required=True)
    p.add_argument('--external-root', type=Path, required=True, help='Completed run containing RESULTS_MANIFEST.json and probes/')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    # Verify complete external input/output hashes before consuming tables.
    subprocess.run([sys.executable, 'tools/external_run_manifest.py', 'verify', '--run', str(a.external_root)], check=True)
    import json
    record = json.loads((a.external_root/'RESULTS_MANIFEST.json').read_text())
    if record.get('state') != 'complete':
        raise ValueError('External run must be complete')
    from tools.p2_external_gate import verify_outputs
    verify_outputs(a.external_root, record['outputs'])
    for name in ['partA', 'partB_internal', 'partB_external']:
        (a.output/name).mkdir(parents=True, exist_ok=True)
    import shlex
    for line in commands(a.p1_root, a.output/'partA').splitlines():
        if line.startswith('python -m '):
            argv = shlex.split(line)
            subprocess.run([sys.executable, *argv[1:]], check=True)
    subprocess.run([sys.executable, '-m', 'neuromri.aggregate', '--metrics', str(a.p2_root/'metrics.csv'),
                    '--output', str(a.output/'partB_internal')], check=True)
    contrasts(a.p2_root, a.output/'partB_internal')
    # These external CSVs already contain the prespecified bootstrap/test families.
    import shutil
    for source in (a.external_root/'probes').rglob('*.csv'):
        target = a.output/'partB_external'/source.relative_to(a.external_root/'probes')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    print('Numerical analysis tables written; see README for paper-result mapping.')


if __name__ == '__main__':
    main()
