"""Resumable, hash-verified ORIGINAL public downloads; standard library only."""
import argparse
import csv
import hashlib
from pathlib import Path, PurePosixPath
import time
from urllib.request import Request, urlopen


def verified(path, row):
    if not path.is_file() or path.stat().st_size != int(row['bytes']):
        return False
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest() == row['sha256']


def download(root, manifest='manifests/external/downloads.csv', retries=5):
    root = Path(root).resolve()
    for row in csv.DictReader(Path(manifest).open(encoding='utf-8')):
        relative = PurePosixPath(row['file'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in row['file'] or ':' in row['file']:
            raise ValueError('Unsafe download destination')
        dest = root.joinpath(*relative.parts)
        if dest.exists():
            if not verified(dest, row):
                raise ValueError(f'Existing raw file byte count/SHA256 mismatch: {dest}')
            print('Verified', dest, flush=True)
            continue
        dest.resolve().relative_to(root)  # never write through an escaping symlink
        if not row['url'].startswith('https://'):
            raise ValueError('Public download must use HTTPS')
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name+'.part')
        print('Fetching original URL', row['url'], '->', dest, flush=True)
        for attempt in range(retries):
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset and verified(part, row):
                    break
                headers = {'User-Agent': 'NeuroMRI-research/1.0 (Python urllib)'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                with urlopen(Request(row['url'], headers=headers), timeout=120) as response:
                    if response.status == 206:
                        if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                            raise ValueError('Incorrect resumed Content-Range')
                    elif response.status == 200:
                        offset = 0  # server ignored Range: restart, never append
                    else:
                        raise ValueError(f'Unexpected HTTP status: {response.status}')
                    with part.open('ab' if offset else 'wb') as stream:
                        for chunk in iter(lambda: response.read(4*1024*1024), b''):
                            stream.write(chunk)
                if not verified(part, row):
                    raise ValueError(f'Download byte count/SHA256 mismatch: {part}')
                break
            except (OSError, ValueError):
                if attempt == retries-1:
                    raise
                time.sleep(min(5*(attempt+1), 30))
        if not verified(part, row):
            raise ValueError(f'Unverified partial download: {part}')
        part.rename(dest)
        print('Downloaded and verified', dest, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--manifest', default='manifests/external/downloads.csv')
    a = p.parse_args()
    download(a.root, a.manifest)
