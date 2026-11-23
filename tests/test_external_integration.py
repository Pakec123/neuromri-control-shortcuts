"""Exercise table generation across the full experiment boundary without GPU/data.

Only the frozen NeuroMRI role provider is replaced; fits, cache authentication,
subject splitting, threshold draws, inference and output tables use real code.
"""
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd

from neuromri.common import canonical_hash, digest
from neuromri.p2_external import run


def write_cache(root, manifest, ids, x):
    root.mkdir(parents=True)
    np.save(root/'embeddings.npy', np.asarray(x, dtype='float32'))
    np.save(root/'image_ids.npy', np.asarray(ids))
    identity = dict(manifest=digest(manifest), encoder='resnet18', image_ids=list(map(int, ids)),
        weights_manifest=digest('weights/manifest.yaml'), encoder_code=digest('neuromri/models.py'),
        data_code=digest('neuromri/data.py'))
    meta = dict(identity=identity, key=canonical_hash(identity), scope='smoke', n=len(ids),
        shape=list(x.shape), sha256=digest(root/'embeddings.npy'), ids_sha256=digest(root/'image_ids.npy'))
    (root/'cache.json').write_text(json.dumps(meta))


def test_complete_external_experiments_and_deterministic_repeat(tmp_path, monkeypatch):
    source = []
    for role in ['train', 'source_val', 'source_cal', 'test']:
        for label, cluster in [(1, -1), (0, 0), (0, 1), (0, 2)]:
            for _ in range(4):
                i = len(source)
                source.append(dict(image_id=i, group_id=str(i), y=label, normal_cluster=cluster, role=role))
    source = pd.DataFrame(source).set_index('image_id', drop=False)
    stub = ModuleType('neuromri.data')
    stub.task_roles = lambda c, task, fold: source.copy()
    stub.groups_for = lambda d, role, cluster=None: d[d.role.eq(role) &
        (True if cluster is None else d.y.eq(1) | d.normal_cluster.eq(cluster))].copy()
    def sample(d, n, rng):
        return d[d.group_id.isin(rng.choice(np.sort(d.group_id.unique()), n, replace=False))]
    stub.sample_groups = sample
    monkeypatch.setitem(sys.modules, 'neuromri.data', stub)
    def provenance(c, out, artifacts):
        Path(out).mkdir(parents=True)
    monkeypatch.setattr('neuromri.p2_external.provenance', provenance)

    manifests = tmp_path/'manifests'
    manifests.mkdir()
    source.to_csv(manifests/'images.csv', index=False)
    (manifests/'roles.csv').write_text('synthetic role provider\n')
    rows = []
    for cohort, site, n in [('NEUROCON', 'Bucharest', 40), ('TAOWU', 'Beijing', 40),
                            ('IXI', 'HH', 70), ('IXI', 'Guys', 70), ('IXI', 'IOP', 70),
                            ('ZENODO3935636', 'Alava', 54)]:
        for person in range(n):
            for plane in range(2):
                rows.append(dict(image_id=1000+len(rows), subject_id=f'{cohort}:{site}:{person}',
                    cohort=cohort, site=site, diagnosis=(('AD' if person >= 19 else 'control')
                        if cohort == 'ZENODO3935636' else 'PD' if cohort != 'IXI' and person % 2 else 'control')))
    external = pd.DataFrame(rows)
    manifest = tmp_path/'slices.csv'
    external.to_csv(manifest, index=False)
    people = external.drop_duplicates('subject_id')[['subject_id', 'cohort', 'site', 'diagnosis']].copy()
    people['age'], people['sex'] = 65., 'F'
    subjects_path = tmp_path/'subjects.csv'
    people.to_csv(subjects_path, index=False)
    rng = np.random.default_rng(3)
    sx, ex = rng.normal(size=(len(source), 6)), rng.normal(size=(len(external), 6))
    sx[:, 0] += source.y*2
    ex[:, 0] += external.diagnosis.eq('PD')*2
    write_cache(tmp_path/'source/resnet18', manifests/'images.csv', source.image_id, sx)
    write_cache(tmp_path/'external/resnet18', manifest, external.image_id, ex)
    c = dict(scope='smoke', manifests=str(manifests), source_embeddings=str(tmp_path/'source'),
        external_embeddings=str(tmp_path/'external'), external_manifest=str(manifest),
        encoders=['resnet18'], tasks=['AD', 'PD'], folds=[0], seeds=[5101], train_clusters=[1],
        C_grid=[.1], e2_folds=2, budgets=[0, 4, 32, 64], draws=2, bootstrap_draws=16,
        permutation_draws=16, ixi_split_seed=7101, jobs=1, output=str(tmp_path/'first'),
        external_subjects=str(subjects_path), age_matched_secondary=True, age_match_caliper=5.,
        ad_external=True, ad_split_seed=7201, ad_contrast_k=4)
    first = run(c)
    second = run(dict(c, output=str(tmp_path/'second')))
    pd.testing.assert_frame_equal(first.drop(columns='config_hash'), second.drop(columns='config_hash'), check_exact=True)
    assert len(first) == 292
    assert set(first.loc[first.experiment.eq('E3'), 'analysis']) == {'full', 'age_matched'}
    e3 = first[first.experiment == 'E3']
    assert e3.loc[e3.status == 'ok', ['auroc', 'sensitivity', 'balanced_accuracy']].isna().all().all()
    assert e3.loc[e3.k == 64, 'status'].eq('insufficient_target_subjects').all()
    assert e3.loc[(e3.k == 4) & (e3.strategy == 'target_tolerance'), 'status'].eq('bound_unavailable').all()
    assert e3.loc[e3.status == 'ok', 'worst_site_specificity'].notna().all()
    assert first.temperature.gt(0).all()
    internal = first[first.target == 'NeuroMRI_internal']
    assert internal.n_subjects.isna().all() and internal.n_proxy_groups.notna().all()
    assert first[first.target != 'NeuroMRI_internal'].loc[first.status == 'ok', 'unit_type'].eq('subject').all()
    e4 = first[first.experiment.eq('E4')]
    assert e4.loc[e4.status.eq('ok'), ['auroc', 'auroc_ci_low', 'sensitivity', 'specificity']].notna().all().all()
    assert e4.loc[e4.k.ge(32), 'status'].eq('insufficient_target_subjects').all()
    roles = pd.read_csv(tmp_path/'first/ad_roles.csv')
    assert roles.group_id.is_unique
    assert len(roles.query('role == "target_cal"')) == 9
    assert roles.query('role == "target_cal"').y.eq(0).all()
    assert roles.query('role == "test"').y.value_counts().to_dict() == {1: 35, 0: 10}
    cal_ids = set(roles.query('role == "target_cal"').group_id)
    assert all(set(value.split(';')) <= cal_ids for value in e4.calibration_subjects.dropna() if value)
    tests = pd.read_csv(tmp_path/'first/E4_tests.csv')
    assert set(tests.hypothesis) == {'discrimination', 'specificity_change'}
    from neuromri.p2_external import holm
    np.testing.assert_array_equal(tests.p_holm, holm(tests.p_raw))
    for name in ['E1_metrics.csv', 'E2_metrics.csv', 'E3_metrics.csv', 'E1_contrasts.csv',
                 'E2_tests.csv', 'E3_contrasts.csv', 'E4_metrics.csv', 'E4_tests.csv',
                 'development_roles.csv', 'ixi_roles.csv', 'predictions.csv']:
        assert (tmp_path/'first'/name).stat().st_size > 0
