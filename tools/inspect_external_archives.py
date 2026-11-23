"""Audit every PD archive member and anatomical sequence sidecar without extraction."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tarfile


def inspect(root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary, rows = {}, []
    for cohort, file in [('NEUROCON', 'neurocon.tar.gz'), ('TAOWU', 'taowu.tar.gz')]:
        path = Path(root)/cohort/'raw'/file
        counts, sequences, other_images = Counter(), Counter(), []
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = member.name
                kind = ('T1' if name.endswith('_T1w.nii.gz') else 'functional' if name.endswith('_bold.nii.gz')
                        else 'JSON' if name.endswith('.json') else 'TSV' if name.endswith('.tsv') else 'other')
                counts[kind] += 1
                rows.append(dict(cohort=cohort, member=name, bytes=member.size, kind=kind))
                if kind == 'other' and name.endswith(('.nii', '.nii.gz', '.img', '.hdr', '.dcm')):
                    other_images.append(name)
                if name.endswith('_T1w.json'):
                    sidecar = json.load(archive.extractfile(member))
                    sequences[str(sidecar.get('Sequence', 'unknown'))] += 1
        summary[cohort] = dict(counts=dict(counts), T1_sidecar_sequences=dict(sequences),
                               additional_anatomical_candidates=other_images)
    with (output/'archive_members.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['cohort', 'member', 'bytes', 'kind'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    has_other = any(v['additional_anatomical_candidates'] or v['counts'].get('other', 0)
                    for v in summary.values())
    summary['conclusion'] = ('Additional members require manual modality review before an E1 variant can be specified.'
        if has_other else 'No matched-modality E1 possible in these archives: only T1 MPRAGE anatomy and BOLD fMRI; '
        'the other files are metadata. BOLD is not substituted for T2/PD/FLAIR anatomy.')
    summary['archive_hash_reference'] = 'manifests/external/downloads.csv'
    summary['members_csv_sha256'] = hashlib.sha256((output/'archive_members.csv').read_bytes()).hexdigest()
    (output/'archive_modalities.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', default='manifests/external')
    a = p.parse_args()
    inspect(a.root, a.output)
