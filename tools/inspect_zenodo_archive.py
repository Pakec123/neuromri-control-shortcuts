"""Record the actual Zenodo ZIP layout, including excluded tissue derivatives."""
import argparse
import csv
from pathlib import Path, PurePosixPath
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--output', default='manifests/external/zenodo_archive_members.csv', type=Path)
    a = p.parse_args()
    rows = []
    for name in ['Clinical_data.zip', 'T1_preprocessed.zip']:
        with zipfile.ZipFile(a.raw/name) as archive:
            for item in archive.infolist():
                path = PurePosixPath(item.filename)
                selected = (path.parent.as_posix() == 'T1_preprocessed_SPM_1.5mm/T1_brain'
                            and path.suffix == '.nii') or item.filename == 'clinical_data_id_age_gender.csv'
                rows.append(dict(archive=name, member=item.filename, bytes=item.file_size,
                    crc32=f'{item.CRC:08x}', selected_for_extraction=int(selected)))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} members; {sum(r["selected_for_extraction"] for r in rows)} selected (includes bipolar T1, subsequently excluded by diagnosis)')


if __name__ == '__main__':
    main()
