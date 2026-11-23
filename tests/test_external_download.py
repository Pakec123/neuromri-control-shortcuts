"""Downloader never exposes unverified transfers as completed raw archives."""
import csv
import hashlib
import io
from pathlib import Path

import pytest

from neuromri.external_download import download


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


def manifest(tmp_path, data=b'complete archive'):
    path = tmp_path/'downloads.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['file', 'bytes', 'sha256', 'url'])
        writer.writeheader()
        writer.writerow(dict(file='COHORT/raw/a.zip', bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(), url='https://publisher.invalid/a.zip'))
    return path


@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('range_supported', [False, True])
def test_verified_download_resume_and_ignored_range(tmp_path, monkeypatch, resume, range_supported):
    data = b'complete archive'
    dest = tmp_path/'raw/COHORT/raw/a.zip'
    dest.parent.mkdir(parents=True)
    if resume:
        dest.with_suffix('.zip.part').write_bytes(data[:4])
    calls = []
    def get(request, timeout):
        calls.append(request)
        offset = 4 if resume and range_supported else 0
        return Response(data[offset:], 206 if offset else 200,
                        {'Content-Range': f'bytes {offset}-{len(data)-1}/{len(data)}'})
    monkeypatch.setattr('neuromri.external_download.urlopen', get)
    download(tmp_path/'raw', manifest(tmp_path), retries=1)
    assert dest.read_bytes() == data
    assert not dest.with_suffix('.zip.part').exists()
    download(tmp_path/'raw', tmp_path/'downloads.csv', retries=1)
    assert len(calls) == 1


def test_hash_mismatch_and_existing_corruption_fail_closed(tmp_path, monkeypatch):
    registry = manifest(tmp_path)
    monkeypatch.setattr('neuromri.external_download.urlopen', lambda *a, **k: Response(b'bad'))
    with pytest.raises(ValueError, match='mismatch'):
        download(tmp_path/'raw', registry, retries=1)
    dest = tmp_path/'raw/COHORT/raw/a.zip'
    assert not dest.exists()
    dest.write_bytes(b'bad existing')
    with pytest.raises(ValueError, match='Existing'):
        download(tmp_path/'raw', registry, retries=1)


def test_manifest_path_traversal_rejected(tmp_path):
    registry = manifest(tmp_path)
    registry.write_text(registry.read_text().replace('COHORT/raw/a.zip', '../outside.zip'))
    with pytest.raises(ValueError, match='Unsafe'):
        download(tmp_path/'raw', registry, retries=1)
