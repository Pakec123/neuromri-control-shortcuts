import io
import json
import tarfile
import zipfile

import numpy as np
import pytest

nib = pytest.importorskip('nibabel')

from neuromri.external_data import axial_slices, inventory, pd_demographics, safe_extract, safe_extract_zip


def phantom():
    x, y, z = np.indices((40, 48, 44))
    mask = ((x-20)/16)**2+((y-24)/19)**2+((z-22)/18)**2 < 1
    image = (20+x+2*y+3*z).astype(np.float32)*mask
    return nib.Nifti1Image(image, np.eye(4)), nib.Nifti1Image(mask.astype('uint8'), np.eye(4))


def test_world_orientation_is_invariant_to_voxel_storage():
    image, mask = phantom()
    # Store the same anatomy with x flipped and anisotropic-looking handedness.
    affine = np.eye(4)
    affine[0, 0], affine[0, 3] = -1, image.shape[0]-1
    flipped = nib.Nifti1Image(image.get_fdata()[::-1], affine)
    flipped_mask = nib.Nifti1Image(mask.get_fdata()[::-1], affine)
    a, b = axial_slices(image, mask), axial_slices(flipped, flipped_mask)
    assert len(a) == 5
    for (pa, ia), (pb, ib) in zip(a, b):
        assert pa.size == (224, 224) and pa.mode == 'RGB'
        np.testing.assert_array_equal(pa, pb)
        assert ia == ib


def test_deterministic_volume_scaling_and_axial_band():
    image, mask = phantom()
    a = axial_slices(image, mask)
    b = axial_slices(image, mask)
    assert [r[1]['slice_index'] for r in a] == sorted(r[1]['slice_index'] for r in a)
    assert len({r[1]['crop_xy'] for r in a}) == 1
    assert len({r[1]['intensity_high'] for r in a}) == 1
    for (p, info), (q, again) in zip(a, b):
        np.testing.assert_array_equal(p, q)
        assert info == again
        assert np.asarray(p)[0, 0].sum() == 0


def test_mni_branch_never_resamples_and_rejects_obliquity(monkeypatch):
    image, mask = phantom()
    affine = np.diag([1.5, 1.5, 1.5, 1.])
    image = nib.Nifti1Image(image.get_fdata(), affine)
    mask = nib.Nifti1Image(mask.get_fdata(), affine)
    def forbidden(*a, **kw):
        raise AssertionError('MNI volume was resampled')
    monkeypatch.setattr('neuromri.external_data.resample_to_output', forbidden)
    result = axial_slices(image, mask, already_mni=True)
    assert len(result) == 5
    assert all(info['world_z_mm'] == 1.5*info['slice_index'] for _, info in result)
    affine[0, 1] = .01
    with pytest.raises(ValueError, match='MNI'):
        axial_slices(nib.Nifti1Image(image.get_fdata(), affine),
                     nib.Nifti1Image(mask.get_fdata(), affine), already_mni=True)


def test_invalid_masks_and_nonfinite_inputs_fail_closed():
    image, mask = phantom()
    with pytest.raises(ValueError, match='Empty'):
        axial_slices(image, nib.Nifti1Image(np.zeros(image.shape), np.eye(4)))
    with pytest.raises(ValueError, match='geometry'):
        axial_slices(image, nib.Nifti1Image(mask.get_fdata(), np.diag([2, 1, 1, 1])))
    data = image.get_fdata()
    data[20, 20, 20] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        axial_slices(nib.Nifti1Image(data, np.eye(4)), mask)


def archive(path, files):
    with tarfile.open(path, 'w') as tar:
        for name, contents in files:
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            tar.addfile(info, io.BytesIO(contents))


def test_extract_only_anatomy_and_preserve_raw(tmp_path):
    path = tmp_path/'cohort.tar'
    archive(path, [('sub-a/anat/sub-a_T1w.nii.gz', b'volume'),
                   ('participants.tsv', b'metadata'),
                   ('sub-a/func/sub-a_bold.nii.gz', b'functional')])
    out = tmp_path/'raw'
    assert safe_extract(path, out) == 2
    assert not (out/'sub-a/func/sub-a_bold.nii.gz').exists()
    assert safe_extract(path, out) == 2
    (out/'participants.tsv').write_text('changed')
    with pytest.raises(RuntimeError, match='overwrite'):
        safe_extract(path, out)


@pytest.mark.parametrize('name', ['../outside.txt', '/outside.txt', 'a\\outside.txt'])
def test_archive_traversal_rejected(tmp_path, name):
    path = tmp_path/'bad.tar'
    archive(path, [(name, b'bad')])
    with pytest.raises(ValueError, match='Unsafe'):
        safe_extract(path, tmp_path/'out')


def test_archive_symlinks_rejected(tmp_path):
    path = tmp_path/'bad.tar'
    with tarfile.open(path, 'w') as tar:
        info = tarfile.TarInfo('link')
        info.type, info.linkname = tarfile.SYMTYPE, '../../outside'
        tar.addfile(info)
    with pytest.raises(ValueError, match='link'):
        safe_extract(path, tmp_path/'out')


def test_released_demographic_keys_and_clinical_status(tmp_path):
    path = tmp_path/'neurocon_patients.tsv'
    path.write_bytes(b'code\tstatus\tage\tsex\rcontrol001\tNORMAL\t60\tF\rpatient002\tPD\t65\tM\r')
    metadata = pd_demographics(tmp_path)
    assert metadata['sub-control001']['age'] == 60
    assert metadata['sub-patient002']['sex'] == 'M'
    path.write_text('code\tstatus\ncontrol001\tPD\n')
    with pytest.raises(ValueError, match='disagrees'):
        pd_demographics(tmp_path)


def test_inventory_can_extract_outside_immutable_raw(tmp_path, monkeypatch):
    raw = tmp_path/'raw_data/NEUROCON/raw'
    raw.mkdir(parents=True)
    files = []
    image, _ = phantom()
    for index, subject in enumerate(['sub-control01', 'sub-patient01']):
        path = tmp_path/f'{subject}_T1w.nii.gz'
        nib.save(nib.Nifti1Image(image.get_fdata()+index, image.affine), path)
        files.append((f'neurocon/{subject}/anat/{path.name}', path.read_bytes()))
    files.append(('neurocon_patients.tsv', b'code\tage\tsex\ncontrol01\t60\tF\npatient01\t65\tM\n'))
    archive(raw/'test.tar', files)
    registry = tmp_path/'access.json'
    registry.write_text(json.dumps(dict(cohorts=dict(NEUROCON=dict(
        files={'test.tar': 'https://example.invalid/test.tar'}, site='Test',
        license='CC BY-NC-SA 4.0', expected_PD=1, expected_controls=1)))))
    monkeypatch.setattr('neuromri.external_data.ACCESS', registry)
    extracted = tmp_path/'run/volumes'
    result = inventory(tmp_path/'raw_data', tmp_path/'run/manifests', extracted)
    assert len(result) == 2 and result.age.tolist() == [60, 65]
    assert all((extracted/file).is_file() for file in result.file)
    assert list(raw.iterdir()) == [raw/'test.tar']


def test_zenodo_inventory_selects_full_t1_joins_labels_and_excludes_bipolar(tmp_path, monkeypatch):
    raw = tmp_path/'raw/ZENODO3935636/raw'
    raw.mkdir(parents=True)
    image, _ = phantom()
    with zipfile.ZipFile(raw/'t1.zip', 'w') as z:
        z.writestr('/', b'')
        for i, name in enumerate(['crl_104', 'ea_217', 'tb_301']):
            volume = nib.Nifti1Image(image.get_fdata()+i, np.diag([1.5, 1.5, 1.5, 1.]))
            z.writestr(f'T1_preprocessed_SPM_1.5mm/T1_brain/wmrf{name}.nii', volume.to_bytes())
            z.writestr(f'T1_preprocessed_DARTEL_1.0mm/T1_GM_sm4_1mm_dartel/swrc1f{name}.nii', b'not full T1')
            z.writestr(f'T1_original/{name}.nii.gz', b'not registered')
    with zipfile.ZipFile(raw/'clinical.zip', 'w') as z:
        z.writestr('clinical_data_id_age_gender.csv', 'Subject,id,Age,gender\ncrl,104,79,2\nea,217,84,2\ntb,301,86,1\n')
    registry = tmp_path/'access.json'
    registry.write_text(json.dumps(dict(cohorts=dict(ZENODO3935636=dict(
        files={'t1.zip': 'https://example.invalid/t1', 'clinical.zip': 'https://example.invalid/clinical'},
        site='Alava', license='CC BY 4.0', expected_AD=1, expected_controls=1)))))
    monkeypatch.setattr('neuromri.external_data.ACCESS', registry)
    extracted = tmp_path/'volumes'
    frame = inventory(tmp_path/'raw', tmp_path/'manifests', extracted)
    assert set(frame.diagnosis) == {'AD', 'control'}
    assert frame.original_id.tolist() == ['crl_104', 'ea_217']
    assert frame.age.tolist() == [79, 84]
    assert frame.released_gender_code.eq(2).all() and frame.sex.eq('').all()
    assert len(list(extracted.rglob('*.nii'))) == 3
    assert not list(extracted.rglob('*.nii.gz'))


def test_zip_absolute_files_and_links_rejected(tmp_path):
    path = tmp_path/'bad.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('/outside.csv', b'bad')
    with pytest.raises(ValueError, match='Unsafe'):
        safe_extract_zip(path, tmp_path/'out')
