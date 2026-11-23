"""Audit raw archive layout and derived adult-brain masks without changing data."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import pandas as pd
from neuromri.common import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--processed-root')
    p.add_argument('--manifest-dir', default='manifests/external')
    p.add_argument('--output', default='results/external_audit')
    a = p.parse_args()
    root, out = Path(a.root), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    layout = []
    for cohort, file in [('NEUROCON', 'neurocon.tar.gz'), ('TAOWU', 'taowu.tar.gz'), ('IXI', 'IXI-T1.tar')]:
        archive = root/cohort/'raw'/file
        counts = dict(cohort=cohort, archive=file, sha256=digest(archive), members=0, files=0,
                      t1_volumes=0, functional_volumes=0, other_files=0)
        with tarfile.open(archive) as tar:
            for member in tar:
                counts['members'] += 1
                if not member.isfile():
                    continue
                counts['files'] += 1
                name = member.name
                key = ('t1_volumes' if name.endswith(('_T1w.nii.gz', '-T1.nii.gz')) else
                       'functional_volumes' if name.endswith('_bold.nii.gz') else 'other_files')
                counts[key] += 1
        layout.append(counts)
    target = Path(a.manifest_dir)/'archive_layout.csv'
    pd.DataFrame(layout).to_csv(target, index=False)
    print(pd.DataFrame(layout).drop(columns='sha256').to_string(index=False))
    if a.processed_root:
        rows = []
        for path in sorted(Path(a.processed_root).glob('*/*/mask.nii.gz')):
            mask = nib.load(path)
            volume = float((np.asarray(mask.dataobj) > .5).sum()*abs(np.linalg.det(mask.affine[:3, :3]))/1000)
            rows.append(dict(cohort=path.parent.parent.name, subject=path.parent.name,
                             brain_ml=volume, review=volume < 600 or volume > 2000))
        frame = pd.DataFrame(rows)
        frame.to_csv(out/'mask_qc.csv', index=False)
        (out/'mask_qc_summary.json').write_text(json.dumps(dict(subjects=len(frame),
            review_count=int(frame.review.sum()), criterion='adult mask volume outside 600–2000mL; flag only, no exclusion'), indent=2))
        print(frame.groupby('cohort').brain_ml.agg(['count', 'min', 'median', 'max']).to_string())


if __name__ == '__main__':
    main()
