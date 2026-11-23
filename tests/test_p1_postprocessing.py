"""Local CPU reanalysis cannot promote or alter ungated experimental inputs."""
import csv
import json

import pytest

from neuromri.result_validation import validate_provenance, metrics_equal
from tools.p1_postprocessing import load_gate, provenance, sha


def fixture(tmp_path):
    root = tmp_path/'read_only_export'
    path = root/'run'/'metrics.csv'
    path.parent.mkdir(parents=True)
    path.write_text('synthetic input\n')
    folder = tmp_path/'audit'
    folder.mkdir()
    inventory = folder/'input_inventory.csv'
    with inventory.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'bytes', 'sha256'])
        writer.writeheader()
        writer.writerow(dict(path='run/metrics.csv', bytes=path.stat().st_size, sha256=sha(path)))
    gate = dict(status='PASS', eligible_fits=1350, planned_fits=1350, input_root=str(root),
                input_inventory_sha256=sha(inventory), provenance_values={'container_sha256': {'a'*64:1350}})
    gate_path = folder/'integrity_gate.json'
    gate_path.write_text(json.dumps(gate))
    return gate_path, gate, path


def test_local_provenance_is_distinct_and_binds_gated_inputs(tmp_path):
    gate_path, gate, path = fixture(tmp_path)
    output = tmp_path/'aggregate'
    record = provenance(dict(scope='vega'), output, [path], gate_path)
    assert record['result_scope'] == 'local_cpu_reanalysis'
    assert record['source_result_scope'] == 'vega'
    assert 'slurm_job' not in record and 'container_sha256' not in record
    validate_provenance(record)
    with pytest.raises(ValueError):
        validate_provenance(record, fit=True)
    record['source_gate_sha256'] = '0'*64
    with pytest.raises(ValueError):
        validate_provenance(record)
    path.write_text('changed input\n')
    with pytest.raises(ValueError, match='differs from gated inventory'):
        provenance(dict(scope='vega'), output, [path], gate_path)
    gate['status'] = 'FAIL'
    gate_path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match='passing complete'):
        load_gate(gate_path)




def test_raw_aggregate_float_roundtrip_tolerance_still_rejects_changed_metrics():
    raw = [dict(encoder='resnet50', auroc='0.14163292605577255', brier='1.9706311956673633e-06')]
    assert metrics_equal(raw,[dict(encoder='resnet50',auroc='0.1416329260557725',brier='1.9706311956673637e-06')])
    assert not metrics_equal(raw,[dict(encoder='resnet50',auroc='0.1417',brier=raw[0]['brier'])])
    assert not metrics_equal(raw,[dict(encoder='vit_b_16',auroc=raw[0]['auroc'],brier=raw[0]['brier'])])
