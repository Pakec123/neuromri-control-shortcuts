"""CPU-only prespecified P1 inference from unchanged per-run CSV outputs.

Run: python -m neuromri.p1_stats --run-root results/p1 --output results/p1/aggregate/inference
No torch import, training, network access, or mutation of input fits.
"""
import argparse
import hashlib
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = ('resnet50', 'convnext_tiny', 'efficientnet_v2_s', 'vit_b_16', 'swin_t', 'dinov2_b')
REPAIR = ('resnet50', 'convnext_tiny', 'vit_b_16')
TASKS = ('AD', 'Tumour', 'MS')
MATCH = dict(AD=1, Tumour=2, MS=0)
SEEDS = (5101, 5102, 5103)
METRICS = ('auroc', 'balanced_accuracy', 'worst_negative_specificity')
DRAWS, SEED = 2000, 6101


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(row):
    return (str(row['encoder']), str(row['task']), float(row['alpha']),
            int(row['fold']), int(row['seed']), str(row['policy']))


def specifications():
    return ([('core', m, t) for m, t in product(ENCODERS, TASKS)] +
            [('repair', m, t) for m, t in product(REPAIR, TASKS)])


def pair_id(family, encoder, task, fold, seed):
    if family == 'core':
        return ((encoder, task, 0., fold, seed, 'erm'),
                (encoder, task, 1., fold, seed, 'erm'))
    return ((encoder, task, -1., fold, seed, 'pooled_erm'),
            (encoder, task, -1., fold, seed, 'equal_negative_cluster'))


def holm(pvalues):
    """Step-down adjusted p values in original input order, ties included."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid p values')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def sign_flip(delta, rng, draws=DRAWS):
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all() or draws < 1:
        raise ValueError('Invalid sign-flip input')
    signs = rng.choice([-1., 1.], size=(draws, len(delta)))
    observed = abs(delta.mean())
    # Round-off tolerance counts mathematically tied sums as ties.
    exceed = np.abs((signs * delta).mean(axis=1)) >= observed - 1e-14
    return float((1 + exceed.sum()) / (draws + 1))


class GroupPair:
    """Aligned full-test scores; weighted endpoint evaluation avoids image expansion."""
    def __init__(self, left, right, cluster):
        required = {'group_id', 'y', 'normal_cluster', 'score'}
        for frame in (left, right):
            if not required <= set(frame) or frame.group_id.isna().any() or frame.group_id.duplicated().any():
                raise ValueError('Missing or duplicate prediction group IDs/columns')
        left = left.sort_values('group_id').reset_index(drop=True)
        right = right.sort_values('group_id').reset_index(drop=True)
        if not left[['group_id', 'y', 'normal_cluster']].equals(right[['group_id', 'y', 'normal_cluster']]):
            raise ValueError('Unpaired group IDs, labels or strata')
        self.y = left.y.to_numpy(dtype=float)
        self.cl = left.normal_cluster.to_numpy(dtype=float)
        if not np.isin(self.y, [0, 1]).all() or not np.isfinite(self.cl).all():
            raise ValueError('Invalid prediction labels/strata')
        if not np.isin(self.cl[self.y == 0], [0, 1, 2]).all():
            raise ValueError('Invalid negative stratum')
        self.cells = [np.flatnonzero(self.y == 1)] + [np.flatnonzero((self.y == 0) & (self.cl == c)) for c in range(3)]
        if any(len(c) == 0 for c in self.cells):
            raise ValueError('Missing positive or negative evaluation stratum')
        self.cluster = cluster
        self.scores = [f.score.to_numpy(dtype=float) for f in (left, right)]
        self.auc_order = []
        matched = np.flatnonzero((self.y == 1) | (self.cl == cluster))
        for scores in self.scores:
            if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                raise ValueError('Invalid prediction scores')
            order = matched[np.argsort(scores[matched], kind='stable')]
            starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
            self.auc_order.append((order, starts))

    def endpoints(self, weights):
        weights = np.atleast_2d(weights)
        values = []
        for scores, (order, starts) in zip(self.scores, self.auc_order):
            w = weights[:, order]
            pos = np.add.reduceat(w * self.y[order], starts, axis=1)
            neg = np.add.reduceat(w * (1 - self.y[order]), starts, axis=1)
            auc = (pos * (np.cumsum(neg, axis=1) - .5 * neg)).sum(axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))
            cell_rates = []
            for i, cell in enumerate(self.cells):
                correct = scores[cell] >= .5 if i == 0 else scores[cell] < .5
                cell_rates.append((weights[:, cell] * correct).sum(axis=1) / weights[:, cell].sum(axis=1))
            values.append(np.column_stack([auc, .5 * (cell_rates[0] + cell_rates[self.cluster + 1]),
                                           np.min(cell_rates[1:], axis=0)]))
        return values

    def resample(self, rng, count):
        weights = np.zeros((count, len(self.y)))
        for cell in self.cells:
            weights[:, cell] = rng.multinomial(len(cell), np.full(len(cell), 1 / len(cell)), size=count)
        left, right = self.endpoints(weights)
        return right - left


def bootstrap(delta, pairs, rng, draws=DRAWS):
    """Resample fit identities then independent paired groups per selected occurrence."""
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or not len(delta) or not np.isfinite(delta).all() or draws < 1:
        raise ValueError('Invalid bootstrap differences')
    n = len(delta)
    selected = rng.integers(n, size=(draws, n))
    if pairs is None:
        means = delta[selected].mean(axis=1)
    else:
        if len(pairs) != n:
            raise ValueError('Bootstrap pair count differs from fit count')
        means = np.zeros((draws, delta.shape[1]))
        for fit, pair in enumerate(pairs):
            rows, _ = np.nonzero(selected == fit)
            # Bound memory for large group counts; repeated occurrences get fresh draws.
            for start in range(0, len(rows), 128):
                subset = rows[start:start + 128]
                values = pair.resample(rng, len(subset))
                np.add.at(means, subset, values[:, :delta.shape[1]])
        means /= n
    return np.quantile(means, [.025, .975], axis=0, method='linear')


def mixed_sensitivity(frame, lock=ROOT / 'env/requirements.lock'):
    """Do not silently use a package outside the frozen environment."""
    pins = [line.strip() for line in Path(lock).read_text().splitlines() if line.strip().startswith('statsmodels==')]
    if not pins:
        return dict(status='unavailable', reason='statsmodels is absent from env/requirements.lock; frozen environment unchanged')
    try:
        import statsmodels.formula.api as smf
        if version('statsmodels') != pins[0].split('==')[1]:
            return dict(status='unavailable', reason='Installed statsmodels differs from pinned version')
    except ImportError:
        return dict(status='unavailable', reason='Pinned statsmodels is not installed')
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = smf.mixedlm('delta ~ 1', frame.assign(overall=1), groups='overall', re_formula='0',
                               vc_formula={'fold': '0 + C(fold)', 'seed': '0 + C(seed)'})
            fit = model.fit(reml=True, method='lbfgs', maxiter=1000)
        ci = fit.conf_int().loc['Intercept'].tolist()
        estimate = float(fit.fe_params['Intercept'])
        if not np.isfinite([estimate, *ci]).all():
            raise ValueError('Nonfinite mixed-model estimate or interval')
        return dict(status='converged' if fit.converged else 'not_converged', estimate=estimate,
                    ci_lo=float(ci[0]), ci_hi=float(ci[1]), variance_components=fit.vcomp.tolist(),
                    warnings=[str(w.message) for w in caught])
    except (ValueError, np.linalg.LinAlgError) as exc:
        return dict(status='failed', reason=str(exc))


def analyze(run_root, runlist=ROOT / 'configs/p1_runs.csv', draws=DRAWS, seed=SEED):
    """Read the actual metrics/predictions schema; require every prespecified pair."""
    run_root = Path(run_root).resolve()
    if draws < 1 or seed < 0:
        raise ValueError('Draw count must be positive and seed nonnegative')
    runs = pd.read_csv(runlist)
    lookup = {}
    for row in runs.to_dict('records'):
        ident = identity(row)
        if ident in lookup:
            raise ValueError('Duplicate runlist identity')
        path = (run_root / row['run_id']).resolve()
        if not path.is_relative_to(run_root):
            raise ValueError('Run path outside run root')
        lookup[ident] = path
    needed = {i for family, m, t in specifications() for f, s in product(range(5), SEEDS)
              for i in pair_id(family, m, t, f, s)}
    if not needed <= lookup.keys():
        raise ValueError('Missing prespecified fit pairs')
    availability = [(lookup[i] / 'predictions.csv').is_file() for i in sorted(needed)]
    if any(availability) and not all(availability):
        raise ValueError('Partially missing predictions; restore the complete export')
    have_groups = all(availability)
    inputs, cache = {}, {}
    for ident in sorted(needed):
        folder = lookup[ident]
        metric = folder / 'metrics.csv'
        inputs[str(metric)] = sha(metric)
        frame = pd.read_csv(metric, dtype={'test_cluster': str})
        if len(frame) != 4 or set(frame.test_cluster) != {'0', '1', '2', 'standardized_pool'}:
            raise ValueError('Missing or duplicate metric strata')
        if {identity(r) for r in frame.to_dict('records')} != {ident}:
            raise ValueError('Metric/runlist identity differs')
        cells = frame[['auroc', 'balanced_accuracy', 'specificity', 'worst_negative_specificity']].to_numpy(dtype=float)
        if not np.isfinite(cells).all() or ((cells < 0) | (cells > 1)).any():
            raise ValueError('Invalid metric values')
        matched = frame[frame.test_cluster == str(MATCH[ident[1]])].iloc[0]
        worst = float(frame[frame.test_cluster != 'standardized_pool'].specificity.min())
        values = np.array([matched.auroc, matched.balanced_accuracy, matched.worst_negative_specificity], dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Invalid metric values')
        if not np.allclose(frame.worst_negative_specificity, worst, atol=1e-9, rtol=0):
            raise ValueError('Worst specificity disagrees with stratum metrics')
        prediction = folder / 'predictions.csv'
        pred = None
        if have_groups:
            inputs[str(prediction)] = sha(prediction)
            pred = pd.read_csv(prediction, dtype={'group_id': str})
        cache[ident] = (values, pred)
    estimates, fit_rows, sensitivity = [], [], []
    streams = np.random.SeedSequence(seed).spawn(len(specifications()) * 2)
    for idx, (family, m, t) in enumerate(specifications()):
        differences, pairs = [], []
        for fold, fit_seed in product(range(5), SEEDS):
            li, ri = pair_id(family, m, t, fold, fit_seed)
            lvalue, left = cache[li]
            rvalue, right = cache[ri]
            if have_groups:
                pair = GroupPair(left, right, MATCH[t])
                computed = pair.endpoints(np.ones(len(pair.y)))
                if not all(np.allclose(v[0], raw, atol=1e-9, rtol=0) for v, raw in zip(computed, [lvalue, rvalue])):
                    raise ValueError('Prediction endpoints disagree with metrics.csv')
                pairs.append(pair)
            delta = rvalue - lvalue
            # Repair uses only worst specificity; group evaluation returns it in column 2.
            differences.append(delta)
            for metric, value in zip(METRICS if family == 'core' else METRICS[2:], delta if family == 'core' else delta[2:]):
                fit_rows.append(dict(family=family, encoder=m, task=t, fold=fold, seed=fit_seed, metric=metric, delta=float(value)))
        delta = np.asarray(differences)
        ci = bootstrap(delta, pairs if have_groups else None, np.random.default_rng(streams[2 * idx]), draws)
        primary_index = 0 if family == 'core' else 2
        p = sign_flip(delta[:, primary_index], np.random.default_rng(streams[2 * idx + 1]), draws)
        for k in (range(3) if family == 'core' else [2]):
            estimates.append(dict(family=family, encoder=m, task=t, metric=METRICS[k], n_fits=15,
                                  estimate=float(delta[:, k].mean()), ci_lo=float(ci[0, k]), ci_hi=float(ci[1, k]),
                                  p_raw=p if k == primary_index else None, p_holm=None))
        frame = pd.DataFrame([r for r in fit_rows if r['family'] == family and r['encoder'] == m and r['task'] == t and r['metric'] == METRICS[primary_index]])
        sensitivity.append(dict(family=family, encoder=m, task=t, metric=METRICS[primary_index], **mixed_sensitivity(frame)))
    for family, metric in [('core', 'auroc'), ('repair', 'worst_negative_specificity')]:
        rows = [r for r in estimates if r['family'] == family and r['metric'] == metric]
        for row, adjusted in zip(rows, holm([r['p_raw'] for r in rows])):
            row['p_holm'] = float(adjusted)
    return dict(schema_version=1, draws=draws, seed=seed, threshold=.5,
                bootstrap_scope='two_level_fits_and_groups' if have_groups else 'fit_only_predictions_absent',
                inputs=inputs, runlist_sha256=sha(runlist), plan_sha256=sha(ROOT / 'docs/P1_ANALYSIS_PLAN.md'),
                module_sha256=sha(__file__), lock_sha256=sha(ROOT / 'env/requirements.lock'),
                packages={p: version(p) for p in ['numpy', 'pandas', 'scipy']},
                estimates=estimates, fit_deltas=fit_rows, sensitivity=sensitivity)


def write_analysis(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('estimates', result['estimates']), ('fit_deltas', result['fit_deltas'])]:
        pd.DataFrame(rows).to_csv(output / (name + '.csv'), index=False)
    (output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--runlist', type=Path, default=ROOT / 'configs/p1_runs.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_root, args.runlist)
    write_analysis(result, args.output)
    print(f"Wrote 18 core and 9 repair tests ({result['bootstrap_scope']}; {DRAWS} draws, seed {SEED}).")


if __name__ == '__main__':
    main()
