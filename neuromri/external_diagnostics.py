"""Descriptive joint modality/rendering/cohort shift checks; no causal attribution."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

from .common import config_cli, provenance
from .external_embeddings import read_cache


def domain_oof(x, labels, seed=9101):
    """Fixed C=.1, five stratified folds of units, no slice-level split or tuning."""
    labels = np.asarray(labels)
    if np.bincount(labels).min() < 5:
        raise ValueError('Domain diagnostic needs five units in each domain')
    scores, fold_ids = np.empty(len(labels)), np.empty(len(labels), dtype=int)
    for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(x, labels)):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight='balanced',
                              max_iter=1000, random_state=seed))
        model.fit(x[train], labels[train])
        scores[test] = model.predict_proba(x[test])[:, 1]
        fold_ids[test] = fold
    return scores, fold_ids


def group_features(frame, x, position):
    values = normalize(np.asarray(x[position.loc[frame.image_id].to_numpy()]))
    d = pd.DataFrame(values, index=frame.group_id)
    return d.groupby(level=0, sort=True).mean()


def image_histogram(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('L'))
    foreground = pixels[pixels > 0]
    if not len(foreground):
        raise ValueError(f'Empty foreground: {path}')
    counts = np.histogram(foreground, bins=32, range=(0, 256))[0].astype(float)
    return counts/counts.sum()


def histogram_distances(left, right):
    # Unit means first: neither slice count nor pixel/brain size weights people.
    p, q = np.mean(left, axis=0), np.mean(right, axis=0)
    centers = (np.arange(32)+.5)/32
    return dict(jensen_shannon_distance=float(jensenshannon(p, q, base=2)),
                wasserstein_intensity_0_1=float(wasserstein_distance(centers, centers, p, q)))


def run(c):
    from .data import task_roles
    from .p2_external import external_frame
    threadpool_limits(1)
    out = Path(c['output'])/'diagnostics'
    if (out/'domain_metrics.csv').exists():
        raise FileExistsError(out)
    source = task_roles(c, 'PD', 0).query('y == 1').copy()
    source['group_id'] = 'NeuroMRI:proxy:'+source.group_id.astype(str)
    source['cohort'] = 'NeuroMRI_PD'
    external = external_frame(c).query('y == 1').copy()
    provenance(c, out, [c['external_manifest'], Path(c['manifests'])/'images.csv'])
    metrics, predictions = [], []
    for name in c['encoders']:
        sx, si, _ = read_cache(Path(c['source_embeddings'])/name, Path(c['manifests'])/'images.csv', c['scope'], name)
        ex, ei, _ = read_cache(Path(c['external_embeddings'])/name, c['external_manifest'], c['scope'], name)
        sf, ef = group_features(source, sx, si), group_features(external, ex, ei)
        for cohort in ['NEUROCON', 'TAOWU']:
            target = ef.loc[sorted(external.loc[external.cohort.eq(cohort), 'group_id'].unique())]
            x = np.concatenate([sf.to_numpy(), target.to_numpy()])
            y = np.r_[np.zeros(len(sf), dtype=int), np.ones(len(target), dtype=int)]
            score, folds = domain_oof(x, y)
            metrics.append(dict(encoder=name, target=cohort, domain_auroc=roc_auc_score(y, score),
                n_source_proxy_groups=len(sf), n_external_PD_subjects=len(target), seed=9101,
                interpretation='Joint modality/rendering/cohort/demographic separability; descriptive'))
            predictions.append(pd.DataFrame(dict(encoder=name, target=cohort,
                group_id=list(sf.index)+list(target.index), external_domain=y, score=score, cv_fold=folds)))
    pd.DataFrame(metrics).to_csv(out/'domain_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out/'domain_predictions.csv', index=False)
    hists = []
    for frame, root in [(source, c['data_root']), (external, c['external_root'])]:
        for person, part in frame.groupby('group_id', sort=True):
            h = np.mean([image_histogram(Path(root)/file) for file in part.file], axis=0)
            hists.append(dict(group_id=person, cohort=part.cohort.iloc[0],
                              **{f'bin_{i:02d}': value for i, value in enumerate(h)}))
    hists = pd.DataFrame(hists)
    hists.to_csv(out/'intensity_histograms.csv', index=False)
    bins = [f'bin_{i:02d}' for i in range(32)]
    rows = []
    for left, right in [('NeuroMRI_PD', 'NEUROCON'), ('NeuroMRI_PD', 'TAOWU'), ('NEUROCON', 'TAOWU')]:
        a, b = [hists.loc[hists.cohort.eq(cohort), bins].to_numpy() for cohort in [left, right]]
        rows.append(dict(source=left, target=right, n_source_units=len(a), n_target_units=len(b),
                         **histogram_distances(a, b)))
    pd.DataFrame(rows).to_csv(out/'intensity_distances.csv', index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
    return pd.DataFrame(metrics)


if __name__ == '__main__':
    run(config_cli('configs/p2_external.yaml'))
