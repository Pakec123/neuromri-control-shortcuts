"""Export deterministic subject split definitions, with no model fitting."""
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from neuromri.common import config_cli, digest
from neuromri.p2_external import ad_roles, e2_roles, external_frame, ixi_roles


def main():
    c = config_cli('configs/p2_external.yaml')
    d = external_frame(c)
    out = Path(c.get('export_manifest_dir', 'manifests/external'))
    out.mkdir(parents=True, exist_ok=True)
    ixi_roles(d, c['ixi_split_seed']).to_csv(out/'ixi_roles.csv', index=False, lineterminator='\n')
    if c.get('ad_external', False):
        ad_roles(d, c['ad_split_seed']).to_csv(out/'ad_roles.csv', index=False, lineterminator='\n')
    if c.get('age_matched_secondary', False):
        from neuromri.external_demographics import export
        export(pd.read_csv(c['external_subjects']), ixi_roles(d, c['ixi_split_seed']),
               out, c['age_match_caliper'])
    frames = []
    for cohort in ['NEUROCON', 'TAOWU']:
        for seed in c['seeds']:
            for fold in range(c['e2_folds']):
                roles = e2_roles(d[d.cohort == cohort], c['e2_folds'], fold, seed)
                frames.append(roles.assign(train_cohort=cohort, seed=seed, fold=fold))
    roles = pd.concat(frames, ignore_index=True)
    roles.to_csv(out/'e2_roles.csv', index=False, lineterminator='\n')
    definition = dict(scope='precomputed subject split definitions; no model fits',
        e2_folds=c['e2_folds'], seeds=c['seeds'], ixi_split_seed=c['ixi_split_seed'],
        slice_manifest_sha256=digest(c['external_manifest']),
        role_code_sha256=digest(Path('neuromri/p2_external.py')),
        age_matching_code_sha256=digest(Path('neuromri/external_demographics.py')),
        age_match_caliper=c.get('age_match_caliper'),
        e2_roles_sha256=digest(out/'e2_roles.csv'), ixi_roles_sha256=digest(out/'ixi_roles.csv'))
    if c.get('ad_external', False):
        definition.update(ad_split_seed=c['ad_split_seed'], ad_roles_sha256=digest(out/'ad_roles.csv'))
    (out/'split_definition.json').write_text(json.dumps(definition, indent=2), encoding='utf-8', newline='\n')
    print(f'IXI {d[d.cohort == "IXI"].group_id.nunique()} subjects; E2 {len(roles)} subject/fold/seed rows')


if __name__ == '__main__':
    main()
