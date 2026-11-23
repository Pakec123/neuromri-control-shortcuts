"""Public external T1 acquisition, subject inventory and deterministic axial slicing.

Raw archives are immutable. Extract only anatomical/metadata files, without trusting
archive paths, links or diagnosis guesses. No image is ever written into git.
"""
import argparse
import json
import re
import subprocess
import sys
import tarfile
import zipfile
import shutil
from pathlib import Path, PurePosixPath

import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .common import canonical_hash, digest, provenance

ACCESS = Path("manifests/external/access.json")
BAND = (.40, .45, .50, .55, .60)
PIPELINE = dict(version=1, orientation="RAS+ world-axial, neurological display",
                spacing_mm=1., band=BAND, percentiles=[1, 99], size=224,
                crop="one subject-wide brain bounding box, 5mm margin",
                skullstrip="pinned SynthStrip model 1; border 1mm; retain CSF",
                registration="none: world-axis resampling; residual head tilt retained")


def pd_demographics(extracted):
    """Released *_patients.tsv covers controls too; IDs omit the BIDS sub- prefix."""
    metadata = {}
    for tsv in sorted(Path(extracted).rglob("*.tsv")):
        if not tsv.name.endswith(("participants.tsv", "_patients.tsv")):
            continue
        frame = pd.read_csv(tsv, sep="\t")  # also handles Tao Wu's CR line endings
        key = next((k for k in ["participant_id", "code", "name"] if k in frame), None)
        if key is None:
            raise ValueError(f"Unknown clinical subject key: {tsv}")
        for row in frame.to_dict("records"):
            original = str(row[key])
            subject = original if original.startswith("sub-") else "sub-"+original
            if subject in metadata:
                raise ValueError(f"Duplicate demographic subject: {subject}")
            status = str(row.get("status", "")).strip().upper()
            if status and status != "NAN":
                expected = "control" if status == "NORMAL" else "patient" if status in {"PD", "PARKINSON"} else None
                if expected is None or not subject.startswith("sub-"+expected):
                    raise ValueError(f"Clinical status disagrees with released subject ID: {subject}/{status}")
            metadata[subject] = row
    return metadata


def download(root):
    from .external_download import download as fetch
    fetch(root)


def safe_extract_zip(archive, destination):
    """The publisher ZIPs contain a '/' directory entry; no absolute files allowed."""
    destination = Path(destination)
    count = 0
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.filename == '/' and item.is_dir():
                continue
            path = PurePosixPath(item.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in item.filename:
                raise ValueError(f'Unsafe archive path: {item.filename}')
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive link rejected')
            # The archive label/record says 1mm T1, but its 1mm DARTEL images
            # are smoothed GM maps. Select only the actual warped T1 contrast.
            keep_t1 = (path.parent.as_posix() == 'T1_preprocessed_SPM_1.5mm/T1_brain'
                       and path.name.endswith('.nii'))
            if item.is_dir() or not (keep_t1 or item.filename == 'clinical_data_id_age_gender.csv'):
                continue
            target = destination.joinpath(*path.parts)
            target.resolve().relative_to(destination.resolve())
            # Stream volumes; compare a temporary extracted copy before any overwrite.
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name+'.extracting')
            with z.open(item) as source, temporary.open('wb') as stream:
                shutil.copyfileobj(source, stream)
            if target.exists():
                if digest(target) != digest(temporary):
                    temporary.unlink()
                    raise RuntimeError(f'Refusing raw overwrite: {target}')
                temporary.unlink()
            else:
                temporary.rename(target)
            count += 1
    return count


def zenodo_demographics(extracted):
    paths = list(Path(extracted).rglob('clinical_data_id_age_gender.csv'))
    if len(paths) != 1:
        raise ValueError('Expected one released Zenodo clinical table')
    frame = pd.read_csv(paths[0])
    if not set(frame.Subject) <= {'crl', 'ea', 'tb'} or frame.id.duplicated().any():
        raise ValueError('Unknown or duplicate Zenodo clinical identities')
    return {f'{r.Subject}_{r.id}': dict(age=r.Age, gender_code=r.gender)
            for r in frame.itertuples()}


def safe_extract(archive, destination):
    """Extract anatomy and metadata only; reject malicious members before writes."""
    destination = Path(destination)
    count = 0
    with tarfile.open(archive) as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Archive link rejected: {member.name}")
            if not member.isfile():
                continue
            keep = (member.name.endswith(("_T1w.nii.gz", "-T1.nii.gz")) or
                    member.name.endswith((".tsv", ".txt", ".json", "README", "LICENSE")))
            if not keep:
                continue
            target = destination.joinpath(*path.parts)
            target.resolve().relative_to(destination.resolve())
            data = tar.extractfile(member).read()
            if target.exists():
                if target.read_bytes() != data:
                    raise RuntimeError(f"Refusing raw overwrite: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as f:
                    f.write(data)
            count += 1
    return count


def inventory(root, output="manifests/external", extracted_root=None):
    root, out = Path(root), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    access = json.loads(ACCESS.read_text())
    archives, rows, counts = [], [], {}
    for cohort, spec in access["cohorts"].items():
        raw = root/cohort/"raw"
        for file, url in spec["files"].items():
            path = raw/file
            archives.append(dict(cohort=cohort, file=path.relative_to(root).as_posix(),
                                 bytes=path.stat().st_size, sha256=digest(path), url=url,
                                 license=spec["license"]))
            if tarfile.is_tarfile(path):
                destination = Path(extracted_root)/cohort if extracted_root else raw/"extracted"
                counts[cohort] = safe_extract(path, destination)
            elif zipfile.is_zipfile(path):
                destination = Path(extracted_root)/cohort if extracted_root else raw/'extracted'
                counts[cohort] = counts.get(cohort, 0)+safe_extract_zip(path, destination)
        extracted = Path(extracted_root)/cohort if extracted_root else raw/"extracted"
        metadata = {}
        if cohort == "IXI":
            sheet = pd.read_excel(raw/"IXI.xls", engine="xlrd")
            # Keep only requested covariates in the portable committed manifest.
            sheet[["IXI_ID", "AGE", "SEX_ID (1=m, 2=f)"]].to_csv(out/"ixi_demographics.csv", index=False)
            metadata = {int(r["IXI_ID"]): r for r in sheet.to_dict("records")}
        elif cohort == 'ZENODO3935636':
            metadata = zenodo_demographics(extracted)
        else:
            metadata = pd_demographics(extracted)
        pattern = 'wmrf*.nii' if cohort == 'ZENODO3935636' else '*-T1.nii.gz' if cohort == 'IXI' else '*_T1w.nii.gz'
        volumes = sorted(extracted.rglob(pattern))
        for path in volumes:
            if cohort == 'ZENODO3935636':
                match = re.fullmatch(r'wmrf(crl|ea|tb)_(\d+)\.nii', path.name)
                if not match:
                    raise ValueError(f'Unknown Zenodo image identifier: {path}')
                code, number = match.groups()
                subject = f'{code}_{int(number)}'
                if subject not in metadata:
                    raise ValueError(f'Zenodo image lacks clinical row: {subject}')
                if code == 'tb':
                    continue  # Bipolar disorder is neither AD nor a healthy control.
                meta, site = metadata[subject], spec['site']
                diagnosis = {'crl': 'control', 'ea': 'AD'}[code]
            elif cohort == "IXI":
                match = re.fullmatch(r"(IXI\d+)-(HH|Guys|IOP)-\d+-T1.nii.gz", path.name)
                if not match:
                    raise ValueError(f"Unknown IXI filename: {path}")
                subject, site = match.groups()
                meta = metadata.get(int(subject[3:]), {})
                diagnosis = "control"
            else:
                subject = path.name.removesuffix("_T1w.nii.gz")
                site = spec["site"]
                # These released BIDS identifiers explicitly encode diagnosis.
                if subject.startswith("sub-control"):
                    diagnosis = "control"
                elif subject.startswith("sub-patient") or subject.startswith("sub-pd"):
                    diagnosis = "PD"
                else:
                    raise ValueError(f"Unrecognised diagnosis identifier: {subject}")
                meta = metadata.get(subject, {})
            image = nib.load(path)
            if len(image.shape) != 3:
                raise ValueError(f"Expected structural 3D volume: {path}")
            rows.append(dict(subject_id=f"{cohort}:{subject}", original_id=subject,
                cohort=cohort, site=site, diagnosis=diagnosis,
                age=meta.get("age", meta.get("AGE", "")),
                sex=meta.get("sex", {1: "M", 2: "F"}.get(meta.get("SEX_ID (1=m, 2=f)"), "")),
                released_gender_code=meta.get('gender_code', ''),
                file=path.relative_to(Path(extracted_root) if extracted_root else root).as_posix(), sha256=digest(path),
                shape="x".join(map(str, image.shape)),
                axis_codes="".join(nib.aff2axcodes(image.affine)),
                voxel_mm="x".join(f"{v:g}" for v in image.header.get_zooms())))
        if cohort != "IXI":
            observed = pd.Series([r["diagnosis"] for r in rows if r["cohort"] == cohort]).value_counts()
            disease = 'AD' if cohort == 'ZENODO3935636' else 'PD'
            if observed.get(disease, 0) != spec['expected_'+disease] or observed.get("control", 0) != spec["expected_controls"]:
                raise ValueError(f"Unexpected released {cohort} diagnosis counts: {observed.to_dict()}")
    frame = pd.DataFrame(rows).sort_values(["cohort", "subject_id"])
    if not frame.subject_id.is_unique or not frame.sha256.is_unique:
        raise ValueError("Duplicate subjects or identical volumes across cohorts")
    frame.to_csv(out/"subjects.csv", index=False)
    pd.DataFrame(archives).to_csv(out/"downloads.csv", index=False)
    summary = dict(extracted_files=counts, subjects=frame.groupby("cohort").size().to_dict(),
                   diagnoses=frame.groupby(["cohort", "diagnosis"]).size().to_dict())
    summary["diagnoses"] = {"/".join(k): v for k, v in summary["diagnoses"].items()}
    (out/"inventory.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return frame


def axial_slices(image, mask, band=BAND, already_mni=False):
    """World-axis 1mm interpolation, subject-wide crop and volume normalization.

    band is a fraction of inferior-to-superior *brain-mask* extent, not an MNI
    coordinate. Returns pixels and reproducible world-plane/crop information.
    """
    if len(image.shape) != 3 or len(mask.shape) != 3:
        raise ValueError("Only 3D volumes are supported")
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine):
        raise ValueError("Mask and input geometry differ")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError("Invalid spatial affine")
    canonical = nib.as_closest_canonical(image)
    canonical_mask = nib.as_closest_canonical(mask)
    # Resampling removes oblique voxel axes; array axis 2 is now world superior.
    if already_mni:
        # Reorientation only (voxel permutation/flips), no interpolation or registration.
        if not np.allclose(canonical.affine[:3, :3], np.diag(canonical.header.get_zooms()), atol=1e-4):
            raise ValueError('Released MNI input must have axis-aligned RAS geometry')
        if not np.allclose(canonical.header.get_zooms(), (1.5, 1.5, 1.5), atol=1e-4):
            raise ValueError('Released MNI T1 must be the verified 1.5mm full-T1 variant')
        volume, brain = canonical, canonical_mask.get_fdata() > .5
    else:
        volume = resample_to_output(canonical, voxel_sizes=(1, 1, 1), order=1)
        from nibabel.processing import resample_from_to
        brain = resample_from_to(canonical_mask, volume, order=0).get_fdata() > .5
    data = volume.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite structural intensity")
    points = np.array(np.where(brain))
    if points.shape[1] < 100:
        raise ValueError("Empty or implausibly small brain mask")
    lo, hi = points.min(axis=1), points.max(axis=1)
    low, high = np.percentile(data[brain], [1, 99])
    if high <= low:
        raise ValueError("Constant brain intensity")
    data = np.rint(np.clip((data-low)/(high-low), 0, 1)*255).astype(np.uint8)
    data[~brain] = 0
    margin = np.ceil(5/np.asarray(volume.header.get_zooms()[:2])).astype(int)
    x0, y0 = np.maximum(lo[:2]-margin, 0)
    x1, y1 = np.minimum(hi[:2]+margin+1, brain.shape[:2])
    result = []
    for fraction in band:
        if not 0 < fraction < 1:
            raise ValueError("Slice band must lie strictly inside brain extent")
        z = int(np.rint(lo[2]+fraction*(hi[2]-lo[2])))
        # RAS: x=right, y=anterior. Rotate for anterior at top, right at right.
        plane = np.rot90(data[x0:x1, y0:y1, z])
        png = ImageOps.pad(Image.fromarray(plane), (224, 224),
                           method=Image.Resampling.BILINEAR, color=0).convert("RGB")
        info = dict(slice_index=z, band_fraction=fraction,
                    world_z_mm=float(nib.affines.apply_affine(volume.affine, [0, 0, z])[2]),
                    intensity_low=float(low), intensity_high=float(high),
                    crop_xy=f"{x0}:{x1},{y0}:{y1}")
        result.append((png, info))
    if len({r[1]["slice_index"] for r in result}) != len(band):
        raise ValueError("Slice band produced duplicate planes")
    return result


def run_synthstrip(source, mask, tools, gpu=True):
    """Hash-check upstream tool/model; apply the documented determinism-only patch."""
    lock = json.loads(Path("manifests/external/tools.json").read_text())["synthstrip"]
    script, weights = Path(tools)/"mri_synthstrip", Path(tools)/"synthstrip.1.pt"
    for p, key in [(script, "script_sha256"), (weights, "weights_sha256")]:
        if digest(p) != lock[key]:
            raise ValueError(f"Unpinned SynthStrip artifact: {p}")
    code = script.read_text()
    old = "torch.backends.cudnn.benchmark = True"
    if code.count(old) != 1:
        raise ValueError("Unexpected SynthStrip source")
    code = "from neuromri.common import seed_all\nseed_all(5101,4)\n"+code.replace(old, "torch.backends.cudnn.benchmark = False")
    args = [sys.executable, "-c", code, "-i", str(source), "-m", str(mask),
            "--model", str(weights), "--threads", "4"]
    if gpu:
        args.append("--gpu")
    with mask.with_suffix(".log").open("w") as log:
        subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT)


def preprocess(root, subjects, output_root, output_manifest, tools, limit=None, gpu=True, scope="smoke"):
    root, dest = Path(root), Path(output_root)
    frame = pd.read_csv(subjects).sort_values(["cohort", "subject_id"])
    if limit:
        # Development limit is stratified; never use as the publication cohort.
        frame = frame.groupby(["cohort", "site", "diagnosis"], sort=True).head(limit)
    dest.mkdir(parents=True, exist_ok=True)
    config = dict(pipeline=PIPELINE, limit=limit, tool_lock=digest("manifests/external/tools.json"),
                  synthstrip_device="cuda" if gpu else "cpu",
                  zenodo='released SPM warped bias-corrected T1 1.5mm; no spatial interpolation; same SynthStrip/band/intensity; ceil(5mm/voxel) crop margin')
    config["code_sha256"] = digest(__file__)
    key = canonical_hash(config)
    identity_file = dest/"pipeline.json"
    if identity_file.exists() and json.loads(identity_file.read_text())["key"] != key:
        raise ValueError("Preprocessing identity changed: use a new output root")
    identity_file.write_text(json.dumps(dict(key=key, config=config), indent=2))
    provenance(dict(scope=scope, **config), dest, [subjects, "manifests/external/tools.json"])
    rows = []
    for n, row in enumerate(frame.to_dict("records")):
        source = root/row["file"]
        if digest(source) != row["sha256"]:
            raise ValueError(f"Raw volume changed: {source}")
        folder = dest/row["cohort"]/row["original_id"]
        folder.mkdir(parents=True, exist_ok=True)
        mask = folder/"mask.nii.gz"
        if not mask.exists():
            run_synthstrip(source, mask, tools, gpu=gpu)
        already_mni = row['cohort'] == 'ZENODO3935636'
        for image, info in axial_slices(nib.load(source), nib.load(mask), already_mni=already_mni):
            path = folder/f"axial_{info['slice_index']:03d}.png"
            image.save(path, compress_level=6)
            rows.append({k: row[k] for k in ["subject_id", "cohort", "site", "diagnosis"]} |
                dict(image_id=1000000+len(rows), file=path.relative_to(dest).as_posix(),
                     sha256=digest(path), raw_sha256=row["sha256"], mask_sha256=digest(mask),
                     pipeline_hash=key, spatial_processing='released_MNI_no_resampling' if already_mni else 'world_1mm_resampling', **info))
        print(f"{n+1}/{len(frame)} {row['subject_id']}", flush=True)
    result = pd.DataFrame(rows)
    Path(output_manifest).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_manifest, index=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["download", "inventory", "preprocess"])
    p.add_argument("--root", required=True)
    p.add_argument("--manifest-dir", default="manifests/external")
    p.add_argument("--extracted-root", help="Extract outside immutable raw archives; use this as preprocess --root")
    p.add_argument("--output-root")
    p.add_argument("--tools", default="weights/files/external")
    p.add_argument("--limit", type=int)
    p.add_argument("--cpu", action="store_true", help="Run SynthStrip without CUDA on CPU partitions")
    p.add_argument("--scope", choices=["smoke", "vega"], default="smoke")
    a = p.parse_args()
    if a.action == "download":
        download(a.root)
    elif a.action == "inventory":
        inventory(a.root, a.manifest_dir, a.extracted_root)
    else:
        if not a.output_root:
            p.error("preprocess requires --output-root")
        preprocess(a.root, Path(a.manifest_dir)/"subjects.csv", a.output_root,
                   Path(a.manifest_dir)/"slices.csv", a.tools, a.limit, gpu=not a.cpu, scope=a.scope)


if __name__ == "__main__":
    main()
