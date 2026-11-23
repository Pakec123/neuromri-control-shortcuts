"""Side-by-side deterministic input sanity check. Run on the data host."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True)
    p.add_argument("--external-root", required=True)
    p.add_argument("--manifest", default="manifests/external/slices.csv")
    p.add_argument("--output", default="results/external_sanity.png")
    a = p.parse_args()
    source = pd.read_csv('manifests/images.csv')
    ext = pd.read_csv(a.manifest)
    groups = []
    for family in ['AD', 'PD', 'Normal']:
        d = source[source.family == family].sort_values('image_id')
        chosen = d.iloc[[int(i*(len(d)-1)/4) for i in range(5)]]
        groups.append([(Path(a.source_root)/r.file,
                        f'NeuroMRI {family}'+(f' C{r.normal_cluster}' if family == 'Normal' else ''),
                        f'proxy {r.group_primary}') for r in chosen.itertuples()])
    subsets = [(cohort, diagnosis, None) for cohort in ['NEUROCON', 'TAOWU'] for diagnosis in ['control', 'PD']]
    subsets += [('IXI', 'control', site) for site in ['Guys', 'HH', 'IOP']]
    subsets += [('ZENODO3935636', diagnosis, None) for diagnosis in ['control', 'AD']]
    for cohort, diagnosis, site in subsets:
        # Five planes of a fixed subject in every diagnosis/site; never select by score.
        d = ext[(ext.cohort == cohort) & (ext.diagnosis == diagnosis)]
        if site is not None:
            d = d[d.site == site]
        d = d.sort_values(['subject_id', 'slice_index'])
        sid = d.subject_id.iloc[0]
        groups.append([(Path(a.external_root)/r.file,
                        f'{cohort} {site or diagnosis} z={r.world_z_mm:g}', sid)
                       for r in d[d.subject_id == sid].itertuples()])
    canvas = Image.new('RGB', (5*224, len(groups)*256), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for y, group in enumerate(groups):
        for x, (path, label, identity) in enumerate(group):
            with Image.open(path) as image:
                canvas.paste(ImageOps.pad(image.convert('RGB'), (224, 224)), (x*224, y*256))
            draw.text((x*224+3, y*256+226), label, fill='white')
            draw.text((x*224+3, y*256+240), identity, fill='white')
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


if __name__ == '__main__':
    main()
