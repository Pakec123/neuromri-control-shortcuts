"""Local CPU reanalysis of the gated P2 internal Vega export (results plan, Table 2/S7, Figure 2).

Phase `aggregate` runs the unchanged `neuromri.aggregate` over `metrics.csv`
(all_metrics.csv, summary.csv, availability.csv). Phase `contrasts` computes the
prespecified internal paired estimates at the reference fit (fold 0, seed 5101,
source C1): empirical target k=32 versus source specificity90, mean over the 20
calibration draws. Proxy groups are resampled within label using the original
internal `neuromri.aggregate.paired_bootstrap` convention (2000 draws, seed 6101).
Estimation only: no internal test or Holm family.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
from itertools import product
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.p1_postprocessing import load_gate, sha

ENCODERS = ["resnet18", "resnet50", "dinov2_s", "biomedclip"]
TASKS = ["AD", "Tumour", "MS", "PD", "Stroke"]
REFERENCE = dict(fold=0, seed=5101, train_cluster=1)
DRAWS, SEED, K = 2000, 6101, 32


def paired_draw_mean(test, thresholds, baseline, draws=DRAWS, seed=SEED):
    """Same sorted units, RNG and paired label resamples as aggregate.py.

    Averaging each unit's decisions first is algebraically identical to
    averaging the paired endpoint changes over all fixed calibration draws.
    Calibration draws are never resampled or treated as independent groups.
    """
    import numpy as np
    if not test.group_id.is_unique:
        raise ValueError('Duplicate proxy groups in conditional contrast')
    test = test.set_index('group_id').sort_index()
    y, scores = test.y.to_numpy(int), test.score.to_numpy(float)
    thresholds = np.asarray(thresholds, dtype=float)
    if (set(y) != {0, 1} or not np.isfinite(scores).all()
            or not np.isfinite(thresholds).all() or not len(thresholds)
            or not np.isfinite(baseline)):
        raise ValueError('Both labels and finite scores/thresholds are required')
    change = (scores[:, None] >= thresholds).mean(axis=1) - (scores >= baseline)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    def estimate(p, n):
        sn, sp = change[p].mean(), -change[n].mean()
        return [sn, sp, (sn + sp) / 2]
    rng = np.random.default_rng(seed)
    samples = np.asarray([estimate(rng.choice(pos, len(pos), replace=True),
                                   rng.choice(neg, len(neg), replace=True)) for _ in range(draws)])
    lo, hi = np.quantile(samples, [.025, .975], axis=0)
    return {metric: dict(estimate=value, ci_lo=low, ci_hi=high)
            for metric, value, low, high in zip(['sensitivity', 'specificity', 'balanced_accuracy'],
                                               estimate(pos, neg), lo, hi)}


def check_packages():
    packages = {p: importlib.metadata.version(p) for p in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'PyYAML']}
    pins = dict(line.strip().split('==') for line in (ROOT/'env/requirements.lock').read_text().splitlines()
                if '==' in line and not line.startswith('#'))
    if any(value != pins[name] for name, value in packages.items()):
        raise ValueError('Numerical reanalysis packages differ from frozen pins: ' + str(packages))
    return packages


def aggregate(root, output, gate_path):
    from neuromri.aggregate import main as run
    argv = ['neuromri.aggregate', '--metrics', str(root/'metrics.csv'), '--output', str(output),
            '--source-integrity-gate', str(gate_path)]
    sys.argv = argv
    with (output/'aggregate.log').open('w', encoding='utf-8') as log, redirect_stdout(log):
        run()
    return [argv]


def contrasts(root, output):
    import numpy as np
    import pandas as pd
    keys = ['encoder', 'task', 'fold', 'seed', 'train_cluster', 'test_cluster']
    metrics = pd.read_csv(root/'metrics.csv', usecols=keys+['strategy', 'k', 'draw', 'status', 'threshold',
                                                           'sensitivity', 'specificity', 'balanced_accuracy'])
    metrics = metrics.query('fold == 0 and seed == 5101 and train_cluster == 1')
    predictions = pd.read_csv(root/'predictions.csv').query('fold == 0 and seed == 5101 and train_cluster == 1')
    rows = []
    for encoder, task, target in product(ENCODERS, TASKS, range(3)):
        ident = dict(encoder=encoder, task=task, **REFERENCE, test_cluster=target)
        m = metrics[(metrics.encoder == encoder) & (metrics.task == task) & (metrics.test_cluster == target)]
        source = m[m.strategy == 'source_specificity90']
        draws = m[(m.strategy == 'target_quantile') & (m.k == K)].sort_values('draw')
        test = predictions[(predictions.encoder == encoder) & (predictions.task == task)
                           & (predictions.test_cluster == target)][['group_id', 'y', 'score']]
        if len(source) != 1 or len(draws) != 20 or set(draws.status) != {'ok'}:
            raise ValueError(f'Incomplete reference rows: {ident}')
        baseline = float(source.threshold.iloc[0])
        thresholds = draws.threshold.to_numpy(float)
        scores, y = test.score.to_numpy(float), test.y.to_numpy(int)
        # Reconstruction check against the exported per-row decision endpoints.
        def endpoints(t):
            pred = scores >= t
            return pred[y == 1].mean(), (~pred[y == 0]).mean()
        sn, sp = endpoints(baseline)
        if not (np.isclose(sn, source.sensitivity.iloc[0], atol=1e-12) and np.isclose(sp, source.specificity.iloc[0], atol=1e-12)):
            raise ValueError(f'Source operating point does not reproduce exported metrics: {ident}')
        for t, (_, row) in zip(thresholds, draws.iterrows()):
            if not np.allclose(endpoints(t), (row.sensitivity, row.specificity), atol=1e-12):
                raise ValueError(f'Target draw does not reproduce exported metrics: {ident}')
        result = paired_draw_mean(test, thresholds, baseline)
        target_sp = np.mean([endpoints(t)[1] for t in thresholds])
        target_sn = np.mean([endpoints(t)[0] for t in thresholds])
        point = {metric: value['estimate'] for metric, value in result.items()}
        if not np.isclose(point['sensitivity'], target_sn - sn, atol=1e-12):
            raise ValueError('Sensitivity contrast differs from reconstruction')
        for metric in ['sensitivity', 'specificity', 'balanced_accuracy']:
            rows.append(dict(**ident, left_strategy='source_specificity90', right_strategy='target_quantile', k=K,
                             aggregation='mean_over_draws', metric=metric, right_minus_left=point[metric],
                             ci_lo=result[metric]['ci_lo'], ci_hi=result[metric]['ci_hi'],
                             left_value=dict(sensitivity=sn, specificity=sp, balanced_accuracy=(sn+sp)/2)[metric],
                             right_value=dict(sensitivity=target_sn, specificity=target_sp,
                                              balanced_accuracy=(target_sn+target_sp)/2)[metric],
                             n_groups=len(test), positive_groups=int((y == 1).sum()),
                             negative_groups=int((y == 0).sum()), calibration_draws=len(thresholds),
                             bootstrap_draws=DRAWS, bootstrap_seed=SEED, scope='vega'))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'internal_contrasts.csv', index=False)
    return [['internal_contrasts', f'reference fold0/seed5101/C1; k={K}; {DRAWS} draws; seed {SEED}']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['aggregate', 'contrasts'], required=True)
    args = parser.parse_args()
    gate_path = args.gate.resolve()
    gate = load_gate(gate_path)
    if gate.get('gate') != 'p2_internal':
        raise ValueError('Expected the P2 internal gate')
    root, output = Path(gate['input_root']), args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Output must be outside read-only input export')
    for name in ['metrics.csv', 'predictions.csv']:
        recorded = {r.split(',')[0]: r.split(',')[2] for r in gate_path.with_name('input_inventory.csv').read_text().splitlines()[1:]}
        if sha(root/name) != recorded[name]:
            raise ValueError('Input differs from gated inventory: ' + name)
    packages = check_packages()
    output.mkdir(parents=True, exist_ok=True)
    started, tick = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    executed = aggregate(root, output, gate_path) if args.phase == 'aggregate' else contrasts(root, output)
    result = dict(phase=args.phase, started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(),
                  seconds=time.perf_counter()-tick, packages=packages, python=sys.version, executable=sys.executable,
                  platform=platform.platform(), source_gate_sha256=sha(gate_path), commands=executed,
                  result_scope='local_cpu_reanalysis', source_result_scope='vega',
                  source_hashes={str(p.relative_to(ROOT)): sha(p) for p in
                                 [Path(__file__), ROOT/'neuromri/aggregate.py', ROOT/'tools/p1_postprocessing.py']},
                  outputs={str(p.relative_to(output)): sha(p) for p in output.rglob('*')
                           if p.is_file() and not p.name.startswith('runtime_')})
    (output/('runtime_'+args.phase+'.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['phase', 'seconds']}))


if __name__ == '__main__':
    main()
