"""Export the reporting plan and print future aggregate commands; never run jobs."""
import argparse
from itertools import product
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))
from neuromri.p1_stats import ENCODERS, REPAIR, TASKS, MATCH
from neuromri.result_validation import identity
ROOT=Path(__file__).resolve().parents[1]
POLICIES=["pooled_erm","equal_negative_cluster","worst_negative_group","negative_adversarial"]


def export_plan(path):
    # The distributed supplement is the exact dated, versioned prespecification.
    path.write_bytes((ROOT/'docs/P1_ANALYSIS_PLAN.md').read_bytes())


def commands(run_root,aggregate_root):
    q=lambda x:shlex.quote(str(x))
    runs=read_csv(ROOT/'configs/p1_runs.csv');lookup={identity(r):r['run_id'] for r in runs}
    paths=[run_root/r['run_id']/'metrics.csv' for r in runs]
    lines=['#!/usr/bin/env bash','set -euo pipefail',
           'python -m neuromri.aggregate --scope vega --output '+q(aggregate_root)+' --metrics '+' '.join(map(q,paths))]
    pairs=[]
    for m,t in product(ENCODERS,TASKS):pairs.append(((m,t,0.,0,5101,'erm'),(m,t,1.,0,5101,'erm')))
    for m,t,p in product(REPAIR,TASKS,list(POLICIES)[1:]):
        pairs.append(((m,t,-1.,0,5101,'pooled_erm'),(m,t,-1.,0,5101,p)))
    for i,(left,right) in enumerate(pairs):
        query=f'y == 1 or normal_cluster == {MATCH[left[1]]}'
        lines.append('python -m neuromri.aggregate --scope vega --draws 2000 --seed 6101 --left-threshold 0.5 --right-threshold 0.5'
            +' --left '+q(run_root/lookup[left]/'predictions.csv')+' --right '+q(run_root/lookup[right]/'predictions.csv')
            +' --left-query '+q(query)+' --right-query '+q(query)+' --output '+q(aggregate_root/'contrasts'/f'{i:02d}'))
    lines.append('python -m neuromri.p1_stats --run-root '+q(run_root)+' --output '+q(aggregate_root/'inference'))
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commands',type=Path,required=True,help='Write commands only; no execution')
    p.add_argument('--runs',type=Path,default=Path('results/p1'))
    p.add_argument('--aggregate',type=Path,default=Path('results/p1/aggregate'))
    a=p.parse_args();a.commands.parent.mkdir(parents=True,exist_ok=True)
    a.commands.write_text(commands(a.runs,a.aggregate),encoding='utf-8',newline='\n')
    print('Wrote 47 commands without executing them.')
