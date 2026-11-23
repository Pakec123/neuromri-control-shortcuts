"""Synthetic CPU checks for the prespecified P1 repeated-fit analysis."""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from neuromri import p1_stats as stats


def predictions(shift=0.):
    return pd.DataFrame(dict(group_id=[f'g{i}' for i in range(8)],
                             y=[1, 1, 0, 0, 0, 0, 0, 0],
                             normal_cluster=[-1, -1, 0, 0, 1, 1, 2, 2],
                             score=np.clip(np.array([.6, .5, .2, .7, .5, .8, .3, .6]) + shift, 0, 1)))


def test_weighted_endpoints_match_explicit_group_sampling_with_ties():
    left, right = predictions(), predictions(-.1)
    right.loc[right.y == 1, 'score'] += .2
    pair = stats.GroupPair(left, right.sample(frac=1, random_state=3), 1)
    rng = np.random.default_rng(4)
    weights = np.zeros((40, 8), dtype=int)
    for cell in pair.cells:
        weights[:, cell] = rng.multinomial(len(cell), [1 / len(cell)] * len(cell), size=40)
    endpoints = pair.endpoints(weights)
    for j, scores in enumerate(pair.scores):
        for i, weight in enumerate(weights):
            selected = np.repeat(np.arange(8), weight)
            y, p, cl = pair.y[selected], scores[selected], pair.cl[selected]
            matched = (y == 1) | (cl == 1)
            sp = [np.mean(p[(y == 0) & (cl == c)] < .5) for c in range(3)]
            expected = [roc_auc_score(y[matched], p[matched]),
                        .5 * (np.mean(p[y == 1] >= .5) + sp[1]), min(sp)]
            np.testing.assert_allclose(endpoints[j][i], expected)


def test_two_levels_include_fit_and_group_variability_and_are_reproducible():
    pair = stats.GroupPair(predictions(), predictions(-.1), 1)
    l, r = pair.endpoints(np.ones(8))
    delta = np.repeat(r - l, 15, axis=0)
    fit_only = stats.bootstrap(delta, None, np.random.default_rng(1), 2000)
    both = stats.bootstrap(delta, [pair] * 15, np.random.default_rng(1), 2000)
    np.testing.assert_allclose(fit_only[0], fit_only[1])
    assert np.any(both[1] > both[0])
    np.testing.assert_array_equal(both, stats.bootstrap(delta, [pair] * 15, np.random.default_rng(1), 2000))
    varying = np.linspace(-.2, .2, 15)[:, None]
    ci = stats.bootstrap(varying, None, np.random.default_rng(2), 2000)
    assert ci[0, 0] < 0 < ci[1, 0]


def test_holm_families_and_plus_one_sign_flip():
    np.testing.assert_allclose(stats.holm([.03, .001, .02, .4]), [.06, .004, .06, .4])
    delta = np.arange(1, 16) / 100
    rng = np.random.default_rng(3)
    permuted = (rng.choice([-1., 1.], size=(2000, 15)) * delta).mean(axis=1)
    expected = (1 + np.sum(abs(permuted) >= abs(delta.mean()) - 1e-14)) / 2001
    assert stats.sign_flip(delta, np.random.default_rng(3)) == expected
    assert expected >= 1 / 2001
    assert stats.sign_flip(np.zeros(15), np.random.default_rng(3)) == 1
    with pytest.raises(ValueError):
        stats.holm([np.nan])


@pytest.mark.parametrize('change', ['ids', 'label', 'stratum', 'duplicate', 'nan', 'range', 'missing'])
def test_bad_predictions_are_rejected(change):
    left, right = predictions(), predictions()
    if change == 'ids': right.loc[0, 'group_id'] = 'different'
    if change == 'label': right.loc[0, 'y'] = 0
    if change == 'stratum': right.loc[2, 'normal_cluster'] = 1
    if change == 'duplicate': right.loc[0, 'group_id'] = right.loc[1, 'group_id']
    if change == 'nan': right.loc[0, 'score'] = np.nan
    if change == 'range': right.loc[0, 'score'] = 1.1
    if change == 'missing': left = left[left.normal_cluster != 2]; right = right[right.normal_cluster != 2]
    with pytest.raises(ValueError):
        stats.GroupPair(left, right, 1)


def synthetic_runs(root):
    """Every required comparison, real runner columns, small variable group scores."""
    identities = sorted({i for family, m, t in stats.specifications() for f, s in product(range(5), stats.SEEDS)
                         for i in stats.pair_id(family, m, t, f, s)})
    runs = []
    for n, ident in enumerate(identities):
        m, t, a, f, s, policy = ident
        row = dict(zip(['encoder', 'task', 'alpha', 'fold', 'seed', 'policy'], ident))
        row['run_id'] = f'run{n:04d}'
        folder = root / row['run_id']; folder.mkdir(parents=True)
        pred = predictions(-.05 if a == 1 or policy == 'equal_negative_cluster' else 0.)
        pair = stats.GroupPair(pred, pred, stats.MATCH[t])
        values = pair.endpoints(np.ones(8))[0][0]
        rows = []
        for c in range(3):
            v = stats.GroupPair(pred, pred, c).endpoints(np.ones(8))[0][0]
            spec = (pred[(pred.y == 0) & (pred.normal_cluster == c)].score < .5).mean()
            rows.append(dict(**row, test_cluster=str(c), auroc=v[0], balanced_accuracy=v[1],
                             specificity=spec, worst_negative_specificity=values[2]))
        rows.append({**rows[0], 'test_cluster': 'standardized_pool'})
        pd.DataFrame(rows).drop(columns='run_id').to_csv(folder / 'metrics.csv', index=False)
        pred.to_csv(folder / 'predictions.csv', index=False)
        runs.append(row)
    runlist = root / 'runlist.csv'
    pd.DataFrame(runs).to_csv(runlist, index=False)
    return runlist


def test_complete_analysis_schema_pair_validation_fallback_and_output(tmp_path):
    runlist = synthetic_runs(tmp_path)
    result = stats.analyze(tmp_path, runlist, draws=20)
    assert len(result['estimates']) == 63 and len(result['fit_deltas']) == 945
    assert len(result['sensitivity']) == 27
    assert result['bootstrap_scope'] == 'two_level_fits_and_groups'
    for family, size in [('core', 18), ('repair', 9)]:
        rows = [r for r in result['estimates'] if r['family'] == family and r['p_raw'] is not None]
        assert len(rows) == size
        np.testing.assert_allclose([r['p_holm'] for r in rows], stats.holm([r['p_raw'] for r in rows]))
    assert result == stats.analyze(tmp_path, runlist, draws=20)
    stats.write_analysis(result, tmp_path / 'inference')
    assert json.loads((tmp_path / 'inference/analysis.json').read_text()) == result
    assert len(pd.read_csv(tmp_path / 'inference/estimates.csv')) == 63
    # Raw predictions cannot be substituted without detection.
    path = next(tmp_path.glob('run*/predictions.csv')); before = path.read_bytes()
    frame = pd.read_csv(path); frame['score'] = .1; frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='endpoints disagree'):
        stats.analyze(tmp_path, runlist, draws=2)
    path.write_bytes(before)
    metric = path.with_name('metrics.csv'); original = metric.read_bytes()
    frame = pd.read_csv(metric); frame.loc[0, 'auroc'] = np.nan; frame.to_csv(metric, index=False)
    with pytest.raises(ValueError, match='Invalid metric'):
        stats.analyze(tmp_path, runlist, draws=2)
    metric.write_bytes(original)
    bad_runlist = tmp_path / 'bad-runlist.csv'
    runs = pd.read_csv(runlist)
    pd.concat([runs, runs.iloc[:1]]).to_csv(bad_runlist, index=False)
    with pytest.raises(ValueError, match='Duplicate runlist'):
        stats.analyze(tmp_path, bad_runlist, draws=2)
    runs.iloc[1:].to_csv(bad_runlist, index=False)
    with pytest.raises(ValueError, match='Missing prespecified'):
        stats.analyze(tmp_path, bad_runlist, draws=2)
    path.unlink()
    with pytest.raises(ValueError, match='Partially missing'):
        stats.analyze(tmp_path, runlist, draws=2)
    for p in tmp_path.glob('run*/predictions.csv'): p.unlink()
    fallback = stats.analyze(tmp_path, runlist, draws=20)
    assert fallback['bootstrap_scope'] == 'fit_only_predictions_absent'
    assert fallback['estimates'][0]['estimate'] == result['estimates'][0]['estimate']


def test_mixed_model_unavailability_is_explicit_without_installing(tmp_path):
    lock = tmp_path / 'requirements.lock'; lock.write_text('numpy==1.26.4\n')
    result = stats.mixed_sensitivity(pd.DataFrame(), lock)
    assert result['status'] == 'unavailable' and 'absent' in result['reason']
