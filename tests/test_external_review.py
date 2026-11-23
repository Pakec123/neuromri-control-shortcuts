import numpy as np
import pandas as pd
import pytest

from neuromri.external_demographics import age_matched_roles, demographic_table
from neuromri.external_diagnostics import domain_oof, histogram_distances


def test_score_blind_age_matching_preserves_roles_and_excludes_missing():
    people = pd.DataFrame([
        dict(subject_id='p1', cohort='NEUROCON', diagnosis='PD', age=60, sex='M', site='B'),
        dict(subject_id='p2', cohort='TAOWU', diagnosis='PD', age=66, sex='F', site='T'),
        *[dict(subject_id=f'i{i}', cohort='IXI', diagnosis='control', age=a, sex='F', site='HH')
          for i, a in enumerate([61, 65, 20, np.nan, 59, 68])]])
    roles = pd.DataFrame(dict(group_id=[f'i{i}' for i in range(6)], site='HH', y=0,
                             role=['target_cal']*4+['test']*2))
    result = age_matched_roles(people, roles)
    assert set(result.group_id) == {'i0', 'i1', 'i4', 'i5'}
    assert result.group_id.is_unique and result.age_gap.le(5).all()
    assert result.role.eq(result.group_id.map(roles.set_index('group_id').role)).all()
    assert not result.duplicated(['site', 'role', 'reference_PD']).any()
    pd.testing.assert_frame_equal(result, age_matched_roles(people.sample(frac=1, random_state=3),
                                                          roles.sample(frac=1, random_state=4)))
    table = demographic_table(people, result)
    assert table.query('cohort == "IXI" and analysis == "full"').age_missing.iloc[0] == 1


def test_matching_maximizes_cardinality_before_distance():
    # Greedy p1->a would strand p2. Global assignment matches both within caliper.
    people = pd.DataFrame(dict(subject_id=['p1', 'p2', 'a', 'b'],
        cohort=['NEUROCON', 'TAOWU', 'IXI', 'IXI'], diagnosis=['PD', 'PD', 'control', 'control'],
        age=[60, 65, 61, 56]))
    roles = pd.DataFrame(dict(group_id=['a', 'b'], site='HH', role='test', y=0))
    result = age_matched_roles(people, roles)
    assert len(result) == 2
    assert result.set_index('group_id').reference_PD.to_dict() == {'a': 'p2', 'b': 'p1'}


def test_domain_oof_and_histogram_diagnostics_are_deterministic():
    labels = np.repeat([0, 1], 20)
    features = np.random.default_rng(3).normal(size=(40, 5))+labels[:, None]*3
    a, folds = domain_oof(features, labels)
    b, again = domain_oof(features, labels)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(folds, again)
    assert set(folds) == set(range(5))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(labels, a) > .95
    with pytest.raises(ValueError, match='five'):
        domain_oof(features[:22], labels[:22])
    h = np.eye(32)[[0]]
    assert histogram_distances(h, h) == dict(jensen_shannon_distance=0., wasserstein_intensity_0_1=0.)
    assert histogram_distances(h, np.eye(32)[[-1]])['jensen_shannon_distance'] == 1.
