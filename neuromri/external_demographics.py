"""Score-blind external demographic summaries and prespecified IXI age matching."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def age_matched_roles(subjects, roles, caliper=5.):
    """Match pooled PD ages separately within each fixed IXI site/role.

    Maximum-cardinality, minimum-total-absolute-age assignment, without replacement;
    each reference PD person contributes at most once per site/role. Sorted IDs
    provide deterministic ties. No sex, features, scores or outcomes enter matching.
    """
    people = subjects.sort_values('subject_id').copy()
    people['age'] = pd.to_numeric(people.age, errors='coerce')
    reference = people[people.cohort.isin(['NEUROCON', 'TAOWU']) &
                       people.diagnosis.eq('PD') & people.age.notna()]
    if reference.empty:
        raise ValueError('No PD ages for matching')
    candidates = roles.merge(people[['subject_id', 'age']], left_on='group_id',
                             right_on='subject_id', validate='one_to_one').dropna(subset=['age'])
    rows = []
    for (site, role), part in candidates.groupby(['site', 'role'], sort=True):
        part = part.sort_values('group_id')
        distance = abs(reference.age.to_numpy()[:, None]-part.age.to_numpy()[None, :])
        # One unmatched dummy per reference; a match always beats all possible
        # total real distances, hence cardinality takes precedence over distance.
        penalty = (len(reference)+1)*(caliper+1)
        cost = np.where(distance <= caliper, distance, penalty*3)
        cost = np.concatenate([cost, np.full((len(reference), len(reference)), penalty)], axis=1)
        rr, cc = linear_sum_assignment(cost)
        for r, col in zip(rr, cc):
            if col < len(part) and distance[r, col] <= caliper:
                item = part.iloc[col]
                rows.append(dict(group_id=item.group_id, site=site, role=role, y=0,
                    age=item.age, reference_PD=reference.iloc[r].subject_id,
                    reference_age=reference.iloc[r].age, age_gap=distance[r, col]))
    return pd.DataFrame(rows, columns=['group_id', 'site', 'role', 'y', 'age',
                                      'reference_PD', 'reference_age', 'age_gap'])


def demographic_table(subjects, matched):
    parts = [('full', subjects)]
    selected = subjects[subjects.subject_id.isin(matched.group_id)]
    parts.append(('age_matched', selected))
    rows = []
    for analysis, frame in parts:
        for (cohort, site, diagnosis), part in frame.groupby(['cohort', 'site', 'diagnosis'], sort=True):
            age = pd.to_numeric(part.age, errors='coerce')
            sex = part.sex.fillna('').astype(str).str.upper().str.strip()
            gender = pd.to_numeric(part.get('released_gender_code', pd.Series(index=part.index, dtype=float)), errors='coerce')
            rows.append(dict(analysis=analysis, cohort=cohort, site=site, diagnosis=diagnosis,
                n=len(part), n_age=int(age.notna().sum()), age_missing=int(age.isna().sum()),
                age_mean=age.mean(), age_sd=age.std(), age_median=age.median(),
                age_q25=age.quantile(.25), age_q75=age.quantile(.75), age_min=age.min(), age_max=age.max(),
                female=int(sex.eq('F').sum()), male=int(sex.eq('M').sum()),
                sex_missing_or_other=int((~sex.isin(['F', 'M'])).sum()),
                released_gender_code_1=int(gender.eq(1).sum()),
                released_gender_code_2=int(gender.eq(2).sum()),
                released_gender_code_missing=int(gender.isna().sum())))
    return pd.DataFrame(rows)


def export(subjects, roles, output, caliper=5.):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matched = age_matched_roles(subjects, roles, caliper)
    matched.to_csv(output/'ixi_age_matched_roles.csv', index=False)
    demographic_table(subjects, matched).to_csv(output/'cohort_demographics.csv', index=False)
    return matched
