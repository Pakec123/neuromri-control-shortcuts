"""Honest local CPU provenance for reanalysis of a gated Vega export."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gate(path):
    path = Path(path).resolve()
    gate = json.loads(path.read_text(encoding='utf-8'))
    if gate.get('gate') == 'p2_internal':
        # Generalised: the P2 internal gate (tools/p2_internal_gate.py) has one job of 900 units.
        if (gate.get('status'), gate.get('eligible_units'), gate.get('planned_units')) != ('PASS', 900, 900):
            raise ValueError('Local reanalysis requires a passing complete P2 internal integrity gate')
    elif (gate.get('status'), gate.get('eligible_fits'), gate.get('planned_fits')) != ('PASS', 1350, 1350):
        raise ValueError('Local reanalysis requires a passing complete P1 integrity gate')
    inventory = path.with_name('input_inventory.csv')
    if sha(inventory) != gate['input_inventory_sha256']:
        raise ValueError('Gate inventory hash differs')
    return gate


def provenance(config, output, inputs, gate_path):
    import csv
    gate_path = Path(gate_path).resolve()
    gate = load_gate(gate_path)
    root = Path(gate['input_root']).resolve()
    with gate_path.with_name('input_inventory.csv').open(newline='', encoding='utf-8') as stream:
        inventory = {r['path']: r['sha256'] for r in csv.DictReader(stream)}
    for raw in inputs:
        path = Path(raw).resolve()
        if not path.is_relative_to(root) or sha(path) != inventory.get(path.relative_to(root).as_posix()):
            raise ValueError('Reanalysis input differs from gated inventory: ' + str(path))
    versions = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']}
    sif = gate['provenance_values']['container_sha256']
    record = dict(config=config, result_scope='local_cpu_reanalysis', source_result_scope='vega',
                  source_gate=str(gate_path), source_gate_sha256=sha(gate_path), source_container_sha256=list(sif),
                  execution_host=platform.node(), python=sys.version, executable=sys.executable,
                  platform=platform.platform(), packages=versions, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  git_dirty=subprocess.check_output(['git', 'status', '--porcelain'], text=True),
                  artifacts={str(p): sha(p) for p in inputs})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    (output / 'pip-freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    return record
