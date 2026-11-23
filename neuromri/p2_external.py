"""Subject-level E1–E4 frozen-probe experiments, separate from internal P2."""
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from threadpoolctl import threadpool_limits

from .common import canonical_hash, config_cli, config_hash, digest, provenance
from .external_embeddings import read_cache
from .statistics import ba_threshold, quantile_threshold, temperature


def aggregate(frame, scores):
    """One score per real external subject or explicitly labelled source proxy."""
    if not frame.groupby("group_id").y.nunique().eq(1).all():
        raise ValueError("A subject has inconsistent diagnoses")
    return frame.assign(score=np.asarray(scores)).groupby("group_id", sort=True).agg(
        y=("y", "first"), score=("score", "mean"), n_images=("image_id", "size")).reset_index()


def assert_disjoint(*frames):
    seen = set()
    for frame in frames:
        groups = set(frame.group_id)
        if seen & groups:
            raise ValueError("Subject/group leakage across development roles")
        seen |= groups


class Probe:
    def __init__(self, c, x, position, train, val, cal, seed):
        assert_disjoint(train, val, cal)
        for frame in [train, val, cal]:
            if frame.y.nunique() != 2:
                raise ValueError("Both diagnoses required in each development role")
        self.x, self.position = x, position
        self.scaler = StandardScaler().fit(self.raw(train))
        ng = train.drop_duplicates("group_id").y.value_counts()
        sizes = train.groupby("group_id").size()
        weights = np.array([len(train)/(2*ng[y]*sizes[g]) for y, g in zip(train.y, train.group_id)])
        candidates = []
        for C in c.get("C_grid", [.01, .1, 1.]):
            model = LogisticRegression(C=C, solver="lbfgs", max_iter=c.get("max_iter", 1000), random_state=seed)
            model.fit(self.scaler.transform(self.raw(train)), train.y, sample_weight=weights)
            score = aggregate(val, model.predict_proba(self.scaler.transform(self.raw(val)))[:, 1])
            candidates.append((log_loss(score.y, score.score, labels=[0, 1]), C, model))
        _, self.C, self.model = min(candidates, key=lambda r: (r[0], r[1]))
        sc = self.score(cal)
        self.ba = ba_threshold(sc.y, sc.score)
        self.q = quantile_threshold(sc.loc[sc.y == 0, "score"])
        self.temp = temperature(sc.y, sc.score)

    def raw(self, frame, x=None, position=None):
        x = self.x if x is None else x
        position = self.position if position is None else position
        if not frame.image_id.isin(position.index).all():
            raise ValueError("Embedding IDs do not cover the complete requested groups")
        return normalize(np.asarray(x[position.loc[frame.image_id].to_numpy()]), norm="l2")

    def score(self, frame, x=None, position=None):
        p = self.model.predict_proba(self.scaler.transform(self.raw(frame, x, position)))[:, 1]
        return aggregate(frame, p)

    def strategies(self, scores):
        return [("fixed_0.5", .5, scores), ("source_ba", self.ba, scores),
                ("source_specificity90", self.q, scores),
                ("source_temperature", .5, expit(logit(np.clip(scores, 1e-7, 1-1e-7))/self.temp))]


def metrics_ci(frame, threshold, draws=2000, seed=8101, ranking=True, unit="subject"):
    """Conditional percentile CI over unique subjects or declared source proxies."""
    if not frame.group_id.is_unique:
        raise ValueError("Duplicate subjects: select one model and evaluation")
    y, p = frame.y.to_numpy(), frame.score.to_numpy()
    if not len(y):
        raise ValueError("Empty evaluation cohort")
    pred = p >= threshold
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if unit not in {"subject", "proxy_group"}:
        raise ValueError("Unknown evaluation unit")
    result = dict(n_units=len(y), unit_type=unit,
                  n_subjects=len(y) if unit == "subject" else np.nan,
                  n_proxy_groups=len(y) if unit == "proxy_group" else np.nan,
                  n_positive=len(pos), n_negative=len(neg),
                  auroc=float(roc_auc_score(y, p)) if len(pos) and len(neg) else np.nan,
                  sensitivity=float(pred[pos].mean()) if len(pos) else np.nan,
                  specificity=float((~pred[neg]).mean()) if len(neg) else np.nan)
    result["balanced_accuracy"] = (result["sensitivity"]+result["specificity"])/2
    result["absolute_specificity_error"] = abs(result["specificity"]-.9)
    result["brier"] = float(np.mean((p-y)**2))
    result["constructed_prevalence"] = float(y.mean())
    result["auprc_constructed_prevalence"] = float(average_precision_score(y, p)) if len(pos) and len(neg) else np.nan
    rng = np.random.default_rng(seed)
    pi = rng.choice(pos, (draws, len(pos)), replace=True) if len(pos) else np.empty((draws, 0), int)
    ni = rng.choice(neg, (draws, len(neg)), replace=True) if len(neg) else np.empty((draws, 0), int)
    sens = pred[pi].mean(axis=1) if len(pos) else np.full(draws, np.nan)
    spec = (~pred[ni]).mean(axis=1) if len(neg) else np.full(draws, np.nan)
    samples = dict(sensitivity=sens, specificity=spec, balanced_accuracy=(sens+spec)/2)
    if ranking and len(pos) and len(neg):
        samples["auroc"] = np.array([roc_auc_score(y[idx], p[idx]) for idx in np.concatenate([pi, ni], axis=1)])
    for key, values in samples.items():
        low, high = np.quantile(values, [.025, .975])
        result[key+"_ci_low"], result[key+"_ci_high"] = low, high
    return result


def source_evaluations(probe, frame, draws, seed, unit="subject"):
    """Share ranking bootstrap only when ranks/ties are exactly preserved.

    Temperature itself is monotone, but P2's numerical clipping can merge
    extreme probabilities; those transformed scores get their own ranking CI.
    """
    cache = {}
    for strategy, threshold, scores in probe.strategies(frame.score.to_numpy()):
        rank_key = tuple(pd.Series(np.asarray(scores)).rank(method="average"))
        values = metrics_ci(frame.assign(score=scores), threshold, draws, seed, ranking=rank_key not in cache, unit=unit)
        if rank_key not in cache:
            cache[rank_key] = {k: values[k] for k in ["auroc_ci_low", "auroc_ci_high"]}
        values.update(cache[rank_key])
        yield strategy, threshold, values


def ixi_roles(frame, seed=7101):
    people = frame[frame.cohort == "IXI"][["group_id", "site", "y"]].drop_duplicates().sort_values("group_id")
    if not people.group_id.is_unique or not people.y.eq(0).all():
        raise ValueError("IXI requires unique healthy-control subjects and fixed sites")
    parts = []
    for site, part in people.groupby("site", sort=True):
        ids = part.group_id.to_numpy().copy()
        # Independent stable site seed, unaffected by another site's sample count.
        rng = np.random.default_rng(np.random.SeedSequence([seed, sum(map(ord, site))]))
        rng.shuffle(ids)
        pool = set(ids[:len(ids)//2])
        parts.append(part.assign(role=np.where(part.group_id.isin(pool), "target_cal", "test")))
    return pd.concat(parts, ignore_index=True)


def e2_roles(frame, folds, fold, seed):
    people = frame[["group_id", "y"]].drop_duplicates().sort_values("group_id")
    if not people.group_id.is_unique or people.y.value_counts().min() < folds:
        raise ValueError("Insufficient independent source subjects for E2 folds")
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    dev, calibration = list(cv.split(people, people.y))[fold]
    train, val = train_test_split(dev, test_size=.25, stratify=people.iloc[dev].y, random_state=seed)
    result = people.assign(role="")
    for indices, role in [(train, "train"), (val, "source_val"), (calibration, "source_cal")]:
        result.iloc[indices, result.columns.get_loc("role")] = role
    return result


def external_frame(c, task='PD'):
    frame = pd.read_csv(c["external_manifest"])
    if not frame.image_id.is_unique:
        raise ValueError("Duplicate external image IDs")
    frame = frame.assign(group_id=frame.subject_id, y=frame.diagnosis.eq(task).astype(int))
    if not set(frame.diagnosis) <= {"PD", "AD", "control"}:
        raise ValueError("Unknown external diagnosis")
    for key in ["y", "cohort", "site"]:
        if not frame.groupby("group_id")[key].nunique().eq(1).all():
            raise ValueError(f"Inconsistent subject {key}")
    return frame


def ad_roles(frame, seed=7201):
    """All AD cases test-only; half of controls per site calibrate, sorted IDs first."""
    people = frame[frame.cohort.eq('ZENODO3935636')][['group_id', 'site', 'diagnosis']].drop_duplicates().sort_values('group_id')
    if not people.group_id.is_unique or not set(people.diagnosis) <= {'AD', 'control'}:
        raise ValueError('E4 requires unique AD/control subjects')
    people = people.assign(y=people.diagnosis.eq('AD').astype(int), role='test')
    for site, part in people[people.y.eq(0)].groupby('site', sort=True):
        rng = np.random.default_rng(np.random.SeedSequence([seed, sum(map(ord, site))]))
        ids = part.group_id.to_numpy().copy()
        rng.shuffle(ids)
        people.loc[people.group_id.isin(ids[:len(ids)//2]), 'role'] = 'target_cal'
    return people.reset_index(drop=True)


def e4_evaluations(c, probe, external, ex, ei, ident):
    """Frozen source AD probe; target controls alone may change thresholds."""
    roles = ad_roles(external, c.get('ad_split_seed', 7201))
    if roles.empty:
        return [], [], []
    external = external.assign(y=external.diagnosis.eq('AD').astype(int))
    pool = probe.score(external[external.group_id.isin(roles.loc[roles.role.eq('target_cal'), 'group_id'])], ex, ei)
    test = probe.score(external[external.group_id.isin(roles.loc[roles.role.eq('test'), 'group_id'])], ex, ei)
    assert_disjoint(pool, test)
    if not pool.y.eq(0).all() or test.y.nunique() != 2:
        raise ValueError('E4 needs target-negative calibration and AD/control test subjects')
    ident = dict(ident, experiment='E4', target='ZENODO3935636', analysis='full')
    predictions = [f.assign(**ident, role=r) for f, r in [(pool, 'target_cal'), (test, 'test')]]
    rows, contrasts, thresholds = [], [], []
    ranking = None
    for strategy, threshold, values in source_evaluations(probe, test, c['bootstrap_draws'], ident['seed']):
        rows.append(dict(**ident, strategy=strategy, threshold=threshold, k=0, draw=-1,
            status='ok', temperature=probe.temp if strategy == 'source_temperature' else 1.,
            n_target_pool=len(pool), calibration_subjects='', **values))
        if strategy == 'fixed_0.5':
            ranking = {key: values[key] for key in ['auroc_ci_low', 'auroc_ci_high']}
    for k, draw in product(c['budgets'], range(c['draws'])):
        selected = pool.iloc[[]]
        if k <= len(pool):
            rng = np.random.default_rng(np.random.SeedSequence([c.get('ad_split_seed', 7201), k, draw]))
            selected = pool.iloc[rng.choice(len(pool), k, replace=False)]
        for strategy, tolerance in [('target_quantile', False), ('target_tolerance', True)]:
            threshold = (quantile_threshold(selected.score, tolerance=tolerance) if k else probe.q) if k <= len(pool) else np.nan
            status = 'insufficient_target_subjects' if k > len(pool) else 'ok' if np.isfinite(threshold) else 'bound_unavailable'
            row = dict(**ident, strategy=strategy, threshold=threshold, k=k, draw=draw, status=status,
                temperature=1., n_target_pool=len(pool), calibration_subjects=';'.join(selected.group_id))
            if status == 'ok':
                row.update(metrics_ci(test, threshold, c['bootstrap_draws'], ident['seed'], ranking=False), **ranking)
                if k == c.get('ad_contrast_k', 8) and not tolerance:
                    thresholds.append(threshold)
            rows.append(row)
    reference = ident['fold'] == 0 and ident['seed'] == c['seeds'][0] and ident['train_cluster'] == 1
    if reference:
        contrasts.append(dict(**ident, hypothesis='discrimination',
            p_raw=permutation_p(test, c['permutation_draws'], ident['seed']),
            auroc=roc_auc_score(test.y, test.score), **ranking))
        if len(thresholds) == c['draws']:
            contrast = e3_contrast(test[test.y.eq(0)], thresholds, probe.q,
                c['bootstrap_draws'], ident['seed'], c['permutation_draws'])
            contrasts.append(dict(**ident, hypothesis='specificity_change', k=c.get('ad_contrast_k', 8),
                p_raw=contrast['specificity_randomization_p'], **contrast,
                **operating_point_contrast(test, thresholds, probe.q, c['bootstrap_draws'], ident['seed'])))
    return rows, predictions, contrasts


def operating_point_contrast(test, thresholds, baseline, draws=2000, seed=8101):
    """Paired mean-over-calibration-draw sensitivity/BA change; resample people."""
    if not test.group_id.is_unique or test.y.nunique() != 2:
        raise ValueError('Paired operating-point CI requires unique AD/control subjects')
    rng = np.random.default_rng(seed)
    samples, observed = {}, {}
    for label, metric in [(1, 'sensitivity_delta'), (0, 'specificity_delta')]:
        scores = test.loc[test.y.eq(label), 'score'].to_numpy()
        old = scores >= baseline
        new = scores[None, :] >= np.asarray(thresholds)[:, None]
        delta = (new.mean(axis=0)-old)*(1 if label else -1)
        indices = rng.integers(0, len(scores), (draws, len(scores)))
        observed[metric], samples[metric] = float(delta.mean()), delta[indices].mean(axis=1)
    observed['balanced_accuracy_delta'] = (observed['sensitivity_delta']+observed['specificity_delta'])/2
    samples['balanced_accuracy_delta'] = (samples['sensitivity_delta']+samples['specificity_delta'])/2
    # Specificity point delta is already present in e3_contrast; retain its CI too.
    result = {key: value for key, value in observed.items() if key != 'specificity_delta'}
    for key, values in samples.items():
        result[key+'_ci_low'], result[key+'_ci_high'] = np.quantile(values, [.025, .975])
    result['n_test_subjects'] = len(test)
    result['n_positive_subjects'] = int(test.y.eq(1).sum())
    return result


def permutation_p(frame, draws, seed):
    rng = np.random.default_rng(seed)
    observed = abs(roc_auc_score(frame.y, frame.score)-.5)
    values = [abs(roc_auc_score(rng.permutation(frame.y), frame.score)-.5) for _ in range(draws)]
    return (1+np.sum(np.asarray(values) >= observed-1e-14))/(draws+1)


def auc_loss_ci(source, target, draws, seed):
    """Independent source-proxy and external-subject bootstrap, conditional on fit."""
    for frame in [source, target]:
        if not frame.group_id.is_unique or frame.y.nunique() != 2:
            raise ValueError("AUROC loss needs unique groups and both labels per cohort")
    rng = np.random.default_rng(seed)
    samples = []
    observed = []
    for frame in [source, target]:
        y, p = frame.y.to_numpy(), frame.score.to_numpy()
        observed.append(roc_auc_score(y, p))
        indices = np.concatenate([rng.choice(np.flatnonzero(y == label),
            (draws, int((y == label).sum())), replace=True) for label in [0, 1]], axis=1)
        samples.append(np.array([roc_auc_score(y[idx], p[idx]) for idx in indices]))
    low, high = np.quantile(samples[0]-samples[1], [.025, .975])
    return dict(internal_auroc=observed[0], external_auroc=observed[1],
                auroc_loss=observed[0]-observed[1], loss_ci_low=low, loss_ci_high=high,
                source_units="proxy_groups_unknown_patients", target_units="subjects")


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return adjusted


def e3_contrast(test, thresholds, baseline, draws=2000, seed=8101, permutations=None):
    """Average calibration-draw contrast; bootstrap final subjects, never draws."""
    scores = test.score.to_numpy()
    left = (scores < baseline).astype(float)
    right = scores[None, :] < np.asarray(thresholds)[:, None]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scores), (draws, len(scores)))
    lp = left[indices].mean(axis=1)
    rp = right[:, indices].mean(axis=2)
    delta = np.abs(rp-.9).mean(axis=0)-np.abs(lp-.9)
    lo, hi = np.quantile(delta, [.025, .975])
    differences = right.mean(axis=0)-left
    permutations = draws if permutations is None else permutations
    signs = rng.choice([-1, 1], (permutations, len(scores)))
    observed = abs(differences.mean())
    p = (1+np.sum(abs((signs*differences).mean(axis=1)) >= observed-1e-14))/(permutations+1)
    return dict(absolute_error_delta=float(np.abs(right.mean(axis=1)-.9).mean()-abs(left.mean()-.9)),
                delta_ci_low=lo, delta_ci_high=hi, specificity_delta=float(differences.mean()),
                specificity_randomization_p=p, n_subjects=len(scores),
                n_randomization_draws=permutations, n_calibration_draws=len(thresholds))


def e1_e3_unit(c, name, task, fold, seed, source):
    from .data import task_roles, groups_for, sample_groups
    threadpool_limits(c.get("threads_per_fit", 1))
    d = task_roles(c, task, fold)
    # Source identity is a proxy, never advertised as a recovered patient ID.
    d["group_id"] = "NeuroMRI:proxy:"+d.group_id.astype(str)
    sx, si, _ = read_cache(Path(c["source_embeddings"])/name, Path(c["manifests"])/"images.csv", c["scope"], name)
    ex, ei, _ = read_cache(Path(c["external_embeddings"])/name, c["external_manifest"], c["scope"], name)
    external = external_frame(c)
    tr = groups_for(d, "train", source)
    n = min(groups_for(d, "train", i).query("y == 0").group_id.nunique() for i in range(3))
    tr = pd.concat([tr[tr.y == 1], sample_groups(tr[tr.y == 0], n, np.random.default_rng(seed))]).sort_index()
    val, cal = groups_for(d, "source_val", source), groups_for(d, "source_cal", source)
    probe = Probe(c, sx, si, tr, val, cal, seed)
    ident = dict(encoder=name, task=task, fold=fold, seed=seed, train_cluster=source, C=probe.C,
                 source_grouping="proxy_unknown_patient", scope=c["scope"])
    all_metrics, all_predictions, contrasts, selections = [], [], [], []
    internal = probe.score(groups_for(d, "test", source))
    for label, frame in [("NeuroMRI_internal", internal)] + [
            (cohort, probe.score(external[external.cohort == cohort], ex, ei))
            for cohort in ["NEUROCON", "TAOWU"] if task == "PD"]:
        experiment = "E1" if task == "PD" else "source_context"
        all_predictions.append(frame.assign(**ident, experiment=experiment, target=label, role="test"))
        unit = "proxy_group" if label == "NeuroMRI_internal" else "subject"
        for strategy, threshold, values in source_evaluations(probe, frame, c["bootstrap_draws"], seed, unit):
            all_metrics.append(dict(**ident, experiment=experiment, target=label,
                strategy=strategy, threshold=threshold, temperature=probe.temp if strategy == "source_temperature" else 1.,
                k=0, draw=-1, status="ok", **values))
        if task == "PD" and label != "NeuroMRI_internal":
            contrasts.append(dict(**ident, experiment="E1", target=label,
                **auc_loss_ci(internal, frame, c["bootstrap_draws"], seed)))
    full_roles = ixi_roles(external, c["ixi_split_seed"])
    analyses = [("full", full_roles)]
    if c.get("age_matched_secondary", False):
        from .external_demographics import age_matched_roles
        people = pd.read_csv(c["external_subjects"])
        analyses.append(("age_matched", age_matched_roles(people, full_roles, c["age_match_caliper"])))
    for analysis, roles in analyses:
        e3_ident = dict(ident, analysis=analysis)
        for site in sorted(full_roles.site.unique()):
            sr = roles[roles.site == site]
            if sr[sr.role == "target_cal"].empty or sr[sr.role == "test"].empty:
                all_metrics.append(dict(**e3_ident, experiment="E3", target=site,
                    strategy="all", status="empty_age_matched_role"))
                continue
            pool = probe.score(external[external.group_id.isin(sr[sr.role == "target_cal"].group_id)], ex, ei)
            test = probe.score(external[external.group_id.isin(sr[sr.role == "test"].group_id)], ex, ei)
            assert_disjoint(pool, test)
            all_predictions.append(test.assign(**e3_ident, experiment="E3", target=site, role="test"))
            all_predictions.append(pool.assign(**e3_ident, experiment="E3", target=site, role="target_cal"))
            chosen_thresholds = []
            def record(strategy, threshold, scores, k=0, draw=-1, status="ok", selected=()):
                row = dict(**e3_ident, experiment="E3", target=site, strategy=strategy, threshold=threshold,
                           k=k, draw=draw, status=status, n_target_pool=len(pool),
                           temperature=probe.temp if strategy == "source_temperature" else 1.,
                           calibration_subjects=";".join(selected))
                if status == "ok":
                    row.update(metrics_ci(test.assign(score=scores), threshold, c["bootstrap_draws"], seed, ranking=False))
                all_metrics.append(row)
            for strategy, threshold, scores in probe.strategies(test.score.to_numpy()):
                record(strategy, threshold, scores)
            for k, draw in product(c["budgets"], range(c["draws"])):
                if k > len(pool):
                    for strategy in ["target_quantile", "target_tolerance"]:
                        record(strategy, np.nan, test.score, k, draw, "insufficient_target_subjects")
                    continue
                rng = np.random.default_rng(np.random.SeedSequence([c["ixi_split_seed"], sum(map(ord, site)), k, draw]))
                selected = pool.iloc[rng.choice(len(pool), k, replace=False)]
                for strategy, tolerance in [("target_quantile", False), ("target_tolerance", True)]:
                    threshold = quantile_threshold(selected.score, tolerance=tolerance) if k else probe.q
                    record(strategy, threshold, test.score, k, draw,
                           "ok" if np.isfinite(threshold) else "bound_unavailable", selected.group_id.tolist())
                    if k == 32 and not tolerance and np.isfinite(threshold):
                        chosen_thresholds.append(threshold)
            if source == 1 and len(chosen_thresholds) == c["draws"]:
                contrasts.append(dict(**e3_ident, experiment="E3", target=site,
                    **e3_contrast(test, chosen_thresholds, probe.q, c["bootstrap_draws"], seed, c["permutation_draws"])))
    for role, frame in [("train", tr), ("source_val", val), ("source_cal", cal)]:
        selections.append(frame[["group_id"]].drop_duplicates().assign(**ident, experiment="E1_E3_E4", role=role))
    if task == 'AD' and c.get('ad_external', False):
        e4_rows, e4_predictions, e4_contrasts = e4_evaluations(c, probe, external, ex, ei, ident)
        all_metrics.extend(e4_rows)
        all_predictions.extend(e4_predictions)
        contrasts.extend(e4_contrasts)
    return all_metrics, all_predictions, contrasts, selections


def e2_unit(c, name, cohort, fold, seed):
    threadpool_limits(c.get("threads_per_fit", 1))
    external = external_frame(c)
    source = external[external.cohort == cohort]
    target_name = "TAOWU" if cohort == "NEUROCON" else "NEUROCON"
    target = external[external.cohort == target_name]
    roles = e2_roles(source, c["e2_folds"], fold, seed)
    def split(role):
        return source[source.group_id.isin(roles.loc[roles.role == role, "group_id"])]
    x, position, _ = read_cache(Path(c["external_embeddings"])/name, c["external_manifest"], c["scope"], name)
    tr, val, cal = [split(role) for role in ["train", "source_val", "source_cal"]]
    assert_disjoint(tr, val, cal, target)
    probe = Probe(c, x, position, tr, val, cal, seed)
    test = probe.score(target)
    ident = dict(encoder=name, task="PD", fold=fold, seed=seed, train_cohort=cohort,
                 target=target_name, experiment="E2", C=probe.C, scope=c["scope"])
    rows = []
    # Conditional target-label permutation p is descriptive per fit. Holm analysis
    # uses only the prespecified fold 0 / first seed, never the most favourable fit.
    pvalue = permutation_p(test, c["permutation_draws"], seed) if fold == 0 and seed == c["seeds"][0] else np.nan
    for strategy, threshold, values in source_evaluations(probe, test, c["bootstrap_draws"], seed):
        rows.append(dict(**ident, strategy=strategy, threshold=threshold, k=0, draw=-1, status="ok",
            temperature=probe.temp if strategy == "source_temperature" else 1.,
            permutation_p=pvalue if strategy == "fixed_0.5" else np.nan, **values))
    return rows, [test.assign(**ident, role="test")], [], [roles.assign(**ident)]


def run(c):
    out = Path(c["output"])
    if (out/"metrics.csv").exists():
        raise FileExistsError(f"Refusing completed external run overwrite: {out}")
    external = external_frame(c)
    roles = ixi_roles(external, c["ixi_split_seed"])
    artifacts = [Path(c["manifests"])/"images.csv", Path(c["manifests"])/"roles.csv", c["external_manifest"]]
    for name in c["encoders"]:
        for root, manifest in [(c["source_embeddings"], artifacts[0]), (c["external_embeddings"], c["external_manifest"])]:
            read_cache(Path(root)/name, manifest, c["scope"], name)
            artifacts.append(Path(root)/name/"cache.json")
    provenance(c, out, artifacts)
    roles.to_csv(out/"ixi_roles.csv", index=False)
    if c.get('ad_external', False):
        ad = ad_roles(external, c.get('ad_split_seed', 7201))
        if ad.empty or c.get('ad_contrast_k', 8) not in c['budgets']:
            raise ValueError('E4 requires its AD cohort and prespecified contrast budget')
        ad.to_csv(out/'ad_roles.csv', index=False)
    if c.get("age_matched_secondary", False):
        from .external_demographics import export
        export(pd.read_csv(c["external_subjects"]), roles, out, c["age_match_caliper"])
    units = [delayed(e1_e3_unit)(c, *args) for args in product(c["encoders"], c["tasks"], c["folds"], c["seeds"], c["train_clusters"])]
    units += [delayed(e2_unit)(c, *args) for args in product(c["encoders"], ["NEUROCON", "TAOWU"], range(c["e2_folds"]), c["seeds"])]
    results = Parallel(n_jobs=c["jobs"], verbose=5)(units)
    metrics = pd.DataFrame([row for rows, _, _, _ in results for row in rows])
    metrics["analysis"] = metrics.get("analysis", pd.Series(index=metrics.index, dtype=str)).fillna("full")
    metrics["config_hash"] = config_hash(c)
    e3_mask = metrics.experiment.eq("E3")
    keys = ["analysis", "encoder", "task", "fold", "seed", "train_cluster", "strategy", "k", "draw"]
    grouped = metrics.loc[e3_mask].groupby(keys, dropna=False).specificity
    metrics.loc[e3_mask, "worst_site_specificity"] = grouped.transform("min").where(grouped.transform("count") == roles.site.nunique())
    metrics.to_csv(out/"metrics.csv", index=False)
    for experiment in ["E1", "E2", "E3", "E4"]:
        metrics[metrics.experiment == experiment].to_csv(out/f"{experiment}_metrics.csv", index=False)
    predictions = pd.concat([frame for _, frames, _, _ in results for frame in frames], ignore_index=True)
    predictions["analysis"] = predictions.get("analysis", pd.Series(index=predictions.index, dtype=str)).fillna("full")
    predictions.to_csv(out/"predictions.csv", index=False)
    pd.concat([frame for _, _, _, frames in results for frame in frames], ignore_index=True).to_csv(out/"development_roles.csv", index=False)
    # One fit per encoder/direction is the prespecified conditional testing family.
    e2 = metrics[(metrics.experiment == "E2") & (metrics.strategy == "fixed_0.5") & metrics.permutation_p.notna()].copy()
    if len(e2):
        e2["p_holm"] = holm(e2.permutation_p)
        e2.to_csv(out/"E2_tests.csv", index=False)
    contrasts = pd.DataFrame([row for _, _, rows, _ in results for row in rows])
    if len(contrasts):
        contrasts[contrasts.experiment == "E1"].to_csv(out/"E1_contrasts.csv", index=False)
        e4 = contrasts[contrasts.experiment.eq('E4')].copy()
        if len(e4):
            # One prespecified family: four AUROC tests + four k8 specificity tests.
            e4['p_holm'] = holm(e4.p_raw)
            e4.to_csv(out/'E4_tests.csv', index=False)
        contrasts = contrasts[contrasts.experiment == "E3"].copy()
        # Do not multiply sample size by training repetitions. Test only fold 0,
        # first seed, reference source cluster 1; retain other contrast CIs.
        if len(contrasts):
            selected = contrasts.fold.eq(0) & contrasts.seed.eq(c["seeds"][0]) & contrasts.analysis.eq("full")
            contrasts.loc[selected, "p_holm"] = holm(contrasts.loc[selected, "specificity_randomization_p"])
            contrasts.to_csv(out/"E3_contrasts.csv", index=False)
    print(f"{len(units)} probe fits; {len(metrics)} rows -> {out}")
    return metrics


if __name__ == "__main__":
    # Call through the package module so joblib/loky workers unpickle fit functions by
    # module reference; pickling them from __main__ breaks on scipy ufuncs (e.g. expit).
    from neuromri.p2_external import run as _run
    _run(config_cli("configs/p2_external.yaml"))
