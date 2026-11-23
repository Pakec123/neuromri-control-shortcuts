import json

import numpy as np
import pandas as pd
import pytest

from neuromri.common import canonical_hash, digest
from neuromri.external_embeddings import read_cache
from neuromri.p2_external import (Probe, ad_roles, aggregate, assert_disjoint, auc_loss_ci, e2_roles, external_frame,
    e3_contrast, holm, ixi_roles, metrics_ci, operating_point_contrast, permutation_p)


def frame(n=40):
    # Two slices/subject, fully shared metadata and genuinely distinct subjects.
    people = np.repeat(np.arange(n), 2)
    return pd.DataFrame(dict(image_id=np.arange(n*2), group_id=['subject-'+str(g) for g in people],
                             y=people % 2, cohort='NEUROCON', site='Bucharest'))


def test_paired_operating_point_intervals_show_sensitivity_tradeoff():
    people = pd.DataFrame(dict(group_id=[f's{i}' for i in range(30)],
        y=[0]*10+[1]*20, score=np.r_[np.linspace(.1, .9, 10), np.linspace(.2, .95, 20)]))
    result = operating_point_contrast(people, [.6, .7], .5, draws=64)
    assert result['sensitivity_delta'] < 0
    assert result['specificity_delta_ci_low'] >= 0
    assert result['sensitivity_delta_ci_high'] <= 0
    assert result['n_test_subjects'] == 30 and result['n_positive_subjects'] == 20
    assert result == operating_point_contrast(people, [.6, .7], .5, draws=64)
    same = operating_point_contrast(people, [.5, .5], .5, draws=64)
    assert same['sensitivity_delta'] == same['balanced_accuracy_delta'] == 0
    with pytest.raises(ValueError, match='unique'):
        operating_point_contrast(pd.concat([people, people]), [.6], .5)


def test_ad_roles_are_score_blind_and_mci_is_rejected(tmp_path):
    d = pd.DataFrame(dict(group_id=[f's{i:02}' for i in range(54)], cohort='ZENODO3935636',
        site='Alava', diagnosis=['control']*19+['AD']*35, score=np.arange(54)))
    first = ad_roles(d)
    pd.testing.assert_frame_equal(first, ad_roles(d.sample(frac=1, random_state=19).assign(score=-999)))
    assert first.query('role == "target_cal"').y.eq(0).all()
    manifest = tmp_path/'slices.csv'
    pd.DataFrame([dict(image_id=1, subject_id='mci1', cohort='MCI', site='X', diagnosis='MCI')]).to_csv(manifest,index=False)
    with pytest.raises(ValueError, match='Unknown external diagnosis'):
        external_frame(dict(external_manifest=manifest), task='AD')


def test_source_cv_and_target_roles_do_not_leak_subjects():
    d = frame()
    r = e2_roles(d, 5, 0, 5101)
    assert r.groupby('role').y.nunique().eq(2).all()
    assert len(r) == 40 and r.group_id.is_unique
    assert_disjoint(*(d[d.group_id.isin(r[r.role == role].group_id)]
                      for role in ['train', 'source_val', 'source_cal']))
    with pytest.raises(ValueError, match='leakage'):
        assert_disjoint(d, d)
    ix = d.assign(cohort='IXI', site=np.where(d.image_id < 40, 'HH', 'IOP'), y=0)
    roles = ixi_roles(ix)
    assert roles.groupby('site').role.value_counts().eq(10).all()
    pd.testing.assert_frame_equal(roles, ixi_roles(ix.sample(frac=1, random_state=3)))


def test_subject_aggregation_and_negative_only_metrics():
    d = frame(4)
    g = aggregate(d, [.1, .3, .4, .6, .2, .4, .7, .9])
    np.testing.assert_allclose(g.score, [.2, .5, .3, .8])
    negative = g[g.y == 0]
    m = metrics_ci(negative, .25, draws=50)
    assert m['specificity'] == .5
    for key in ['auroc', 'sensitivity', 'balanced_accuracy']:
        assert np.isnan(m[key])
    with pytest.raises(ValueError, match='Duplicate'):
        metrics_ci(pd.concat([g, g]), .5, draws=5)


def test_probe_is_deterministic_and_scaler_uses_train_only():
    d = frame()
    r = e2_roles(d, 5, 0, 5101)
    splits = [d[d.group_id.isin(r[r.role == role].group_id)] for role in ['train', 'source_val', 'source_cal']]
    rng = np.random.default_rng(2)
    x = rng.normal(size=(len(d), 8)).astype('float32')
    x[:, 0] += d.y*2
    position = pd.Series(np.arange(len(d)), index=d.image_id)
    a = Probe({'C_grid': [.01, 1]}, x, position, *splits, 5101)
    b = Probe({'C_grid': [.01, 1]}, x, position, *splits, 5101)
    pd.testing.assert_frame_equal(a.score(d), b.score(d), check_exact=True)
    from sklearn.preprocessing import normalize
    np.testing.assert_allclose(a.scaler.mean_, normalize(x[splits[0].image_id]).mean(axis=0), rtol=1e-5, atol=1e-7)
    assert a.ba == b.ba and a.q == b.q and a.C == b.C
    with pytest.raises(ValueError, match='cover'):
        a.score(d.assign(image_id=9999))


def test_cache_manifest_and_arrays_are_authenticated(tmp_path, monkeypatch):
    manifest = tmp_path/'manifest.csv'
    manifest.write_text('image_id\n1\n2\n')
    np.save(tmp_path/'embeddings.npy', np.ones((2, 3), 'float32'))
    np.save(tmp_path/'image_ids.npy', np.array([1, 2]))
    identity = dict(manifest=digest(manifest), image_ids=[1, 2], encoder='resnet18',
        weights_manifest=digest('weights/manifest.yaml'), encoder_code=digest('neuromri/models.py'),
        data_code=digest('neuromri/data.py'))
    meta = dict(key=canonical_hash(identity), identity=identity,
                sha256=digest(tmp_path/'embeddings.npy'), ids_sha256=digest(tmp_path/'image_ids.npy'),
                scope='smoke', shape=[2, 3], n=2)
    (tmp_path/'cache.json').write_text(json.dumps(meta))
    x, pos, _ = read_cache(tmp_path, manifest)
    assert x.shape == (2, 3) and list(pos.index) == [1, 2]
    x._mmap.close()  # Windows prevents overwriting an open mapping.
    monkeypatch.setenv('NEUROMRI_CONTAINER_SHA256', 'Vega-SIF')
    with pytest.raises(ValueError, match='same Vega SIF'):
        read_cache(tmp_path, manifest, scope='vega')
    with pytest.raises(ValueError, match='encoder identity'):
        read_cache(tmp_path, manifest, encoder_name='dinov2_s')
    np.save(tmp_path/'embeddings.npy', np.zeros((2, 3)))
    with pytest.raises(ValueError, match='Corrupt'):
        read_cache(tmp_path, manifest)


def test_contrast_resamples_subjects_not_calibration_repetitions():
    d = pd.DataFrame(dict(group_id=['a', 'b', 'c', 'd'], y=0, score=[.1, .2, .3, .9]))
    a = e3_contrast(d, [.8, .8], .5, draws=50)
    b = e3_contrast(d, [.8]*20, .5, draws=50)
    for key in ['absolute_error_delta', 'delta_ci_low', 'delta_ci_high', 'specificity_randomization_p']:
        np.testing.assert_allclose(a[key], b[key], atol=1e-14)
    assert a['n_subjects'] == b['n_subjects'] == 4
    np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])


def test_permutation_p_is_seeded_and_two_sided():
    g = pd.DataFrame(dict(group_id=list('abcdefgh'), y=[0]*4+[1]*4, score=np.arange(8)))
    p = permutation_p(g, 50, 1)
    assert p == permutation_p(g, 50, 1) and 0 < p <= 1
    assert p == permutation_p(g.assign(score=-g.score), 50, 1)


def test_auc_loss_uses_independent_subject_units():
    g = pd.DataFrame(dict(group_id=list('abcdefgh'), y=[0]*4+[1]*4, score=np.arange(8)))
    target = g.assign(group_id='external-'+g.group_id, score=-g.score)
    result = auc_loss_ci(g, target, 50, 5101)
    assert result['auroc_loss'] == result['loss_ci_low'] == result['loss_ci_high'] == 1
    with pytest.raises(ValueError, match='unique'):
        auc_loss_ci(g, pd.concat([target, target]), 5, 1)
