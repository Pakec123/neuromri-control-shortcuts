"""Provenance and CSV identity checks extracted without changing their functions."""
from pathlib import Path
import re, math, hashlib
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identity(row):
    return (row['encoder'],row['task'],float(row['alpha']),int(float(row['fold'])),
            int(float(row['seed'])),row['policy'])

def validate_provenance(record, fit=False, attested=False):
    if not fit and record.get('result_scope')=='local_cpu_reanalysis':
        from tools.p1_postprocessing import load_gate
        gate_path=Path(record['source_gate'])
        gate=load_gate(gate_path)
        if (record.get('source_result_scope')!='vega' or record.get('config',{}).get('scope')!='vega'
                or sha(gate_path)!=record.get('source_gate_sha256')
                or record.get('source_container_sha256')!=list(gate['provenance_values']['container_sha256'])):
            raise ValueError('Local analysis source provenance differs')
        return
    if record.get('result_scope')!='vega' or record.get('config',{}).get('scope')!='vega':
        raise ValueError('Expected Vega execution scope in provenance')
    if not record.get('slurm_job') or not re.fullmatch('[0-9a-f]{64}',record.get('container_sha256','')):
        raise ValueError('Missing SLURM/container identity')
    if fit and ('A100' not in record.get('gpu','') or (record.get('git_dirty','').strip() and not attested)):
        raise ValueError('Final fits require A100 and a clean recorded checkout')

def metrics_equal(left,right):
    """Allow only CSV floating-point round-trip error, as for summary/raw checks."""
    if len(left)!=len(right):return False
    for a,b in zip(left,right):
        if set(a)!=set(b):return False
        for key,value in a.items():
            if value==b[key]:continue
            if key in {'encoder','task','policy','test_cluster'}:return False
            try:
                if not math.isclose(float(value),float(b[key]),rel_tol=1e-9,abs_tol=1e-12):return False
            except (TypeError,ValueError):return False
    return True
