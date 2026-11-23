"""Validate real manifests and result-table hygiene, without treating repeats as people."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/external_smoke')
    p.add_argument('--manifests', default='manifests/external')
    a = p.parse_args()
    root, manifests = Path(a.results), Path(a.manifests)
    subjects = pd.read_csv(manifests/'subjects.csv')
    slices = pd.read_csv(manifests/'slices.csv')
    metrics = pd.read_csv(root/'metrics.csv')
    predictions = pd.read_csv(root/'predictions.csv')
    roles = pd.read_csv(root/'ixi_roles.csv')
    assert subjects.subject_id.is_unique and slices.image_id.is_unique
    assert set(slices.subject_id) == set(subjects.subject_id)
    assert slices.groupby('subject_id').size().eq(5).all()
    assert slices.groupby('subject_id').band_fraction.nunique().eq(5).all()
    raw_hash = subjects.set_index('subject_id').sha256
    assert slices.raw_sha256.eq(slices.subject_id.map(raw_hash)).all()
    for key in ['cohort', 'site', 'diagnosis']:
        assert slices[key].eq(slices.subject_id.map(subjects.set_index('subject_id')[key])).all()
    assert roles.group_id.is_unique
    e3 = metrics[metrics.experiment == 'E3']
    ok = e3.status.eq('ok')
    assert e3.loc[ok, ['auroc', 'sensitivity', 'balanced_accuracy']].isna().all().all()
    assert e3.loc[~ok, 'specificity'].isna().all()
    internal = metrics[metrics.target == 'NeuroMRI_internal']
    assert internal.unit_type.eq('proxy_group').all() and internal.n_subjects.isna().all()
    keys = ['encoder', 'task', 'fold', 'seed', 'train_cluster', 'train_cohort']
    if 'analysis' in predictions:
        keys += ['analysis']
    assert not predictions.duplicated(keys+['experiment', 'target', 'role', 'group_id']).any()
    ep = predictions[predictions.experiment == 'E3']
    lookup = roles.set_index('group_id')
    assert ep.role.eq(ep.group_id.map(lookup.role)).all()
    assert ep.target.eq(ep.group_id.map(lookup.site)).all() and ep.y.eq(0).all()
    if 'analysis' in ep and ep.analysis.eq('age_matched').any():
        matched = pd.read_csv(root/'ixi_age_matched_roles.csv').set_index('group_id')
        selected = ep[ep.analysis.eq('age_matched')]
        assert selected.group_id.isin(matched.index).all()
        assert selected.role.eq(selected.group_id.map(matched.role)).all()
        assert matched.age_gap.le(5).all()
    determinism = json.loads((root/'determinism.json').read_text())
    e4 = metrics[metrics.experiment.eq('E4')]
    if len(e4):
        ad = pd.read_csv(root/'ad_roles.csv').set_index('group_id')
        assert ad.index.is_unique
        assert ad.loc[ad.role.eq('target_cal'), 'y'].eq(0).all()
        assert set(ad.loc[ad.role.eq('test'), 'y']) == {0, 1}
        p4 = predictions[predictions.experiment.eq('E4')]
        assert p4.role.eq(p4.group_id.map(ad.role)).all()
        assert p4.y.eq(p4.group_id.map(ad.y)).all()
        assert e4.loc[e4.status.eq('ok'), ['auroc', 'auroc_ci_low', 'sensitivity', 'specificity']].notna().all().all()
        allowed = set(ad.index[ad.role.eq('target_cal')])
        assert all(set(value.split(';')) <= allowed for value in e4.calibration_subjects.dropna() if value)
        e4_tests = pd.read_csv(root/'E4_tests.csv')
        assert len(e4_tests) == 2*metrics.encoder.nunique()
        assert e4_tests.p_holm.ge(e4_tests.p_raw-1e-14).all()
    assert determinism['identical_metrics']
    summary = dict(status='passed', subjects=len(subjects), slices=len(slices),
        subjects_by_cohort=subjects.groupby('cohort').size().to_dict(),
        score_fits=len(metrics[[k for k in keys if k != 'analysis']].drop_duplicates()), metric_rows=len(metrics),
        E3_status_rows=e3.status.value_counts().to_dict(), E4_status_rows=e4.status.value_counts().to_dict(),
        independent_IXI_test_subjects=int(roles.role.eq('test').sum()),
        independent_IXI_calibration_subjects=int(roles.role.eq('target_cal').sum()),
        repeated_metric_rows_checked=determinism['rows_compared'])
    (root/'validation.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
