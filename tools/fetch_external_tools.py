"""Fetch exact public SynthStrip artifacts on a login/development host."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuromri.common import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='weights/files/external')
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    lock = json.loads(Path('manifests/external/tools.json').read_text())['synthstrip']
    for file, url_key, sha_key in [('mri_synthstrip', 'script_url', 'script_sha256'),
                                  ('synthstrip.1.pt', 'weights_url', 'weights_sha256')]:
        path = root/file
        if not path.exists():
            part = path.with_name(file+'.part')
            with urllib.request.urlopen(lock[url_key], timeout=120) as src, part.open('wb') as dest:
                while block := src.read(1024*1024):
                    dest.write(block)
            if digest(part) != lock[sha_key]:
                raise ValueError(f'Download hash mismatch: {file}')
            part.rename(path)
        if digest(path) != lock[sha_key]:
            raise ValueError(f'Existing tool hash mismatch: {path}')
        print(file, lock[sha_key])


if __name__ == '__main__':
    main()
