"""Verify or restore the exact model files listed in snapshots.json."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.parse import quote
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def verify(path, expected):
    if path.stat().st_size != expected['bytes']:
        raise ValueError(f'Wrong file size: {path}')
    with path.open('rb') as source:
        if file_digest(source, 'sha256').hexdigest() != expected['sha256']:
            raise ValueError(f'Wrong SHA-256: {path}')


def restore(path, url, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix='.part', delete=False) as temporary:
        candidate = Path(temporary.name)
        try:
            with urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, temporary, length=2**20)
            temporary.close()
            verify(candidate, expected)
            candidate.rename(path)
        finally:
            candidate.unlink(missing_ok=True)


def main():
    snapshots = json.loads((HERE / 'snapshots.json').read_bytes())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=(*snapshots, 'all'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--endpoint', default='https://huggingface.co')
    args = parser.parse_args()
    if not args.endpoint.startswith('https://'):
        parser.error('endpoint must use HTTPS')
    roles = snapshots if args.role == 'all' else {args.role: snapshots[args.role]}
    for role, snapshot in roles.items():
        directory = PROJECT / snapshot['directory']
        for name, expected in snapshot['files'].items():
            path = directory / name
            if not path.exists() and args.download and not expected.get('bundled'):
                print(f'Downloading {role}/{name}', flush=True)
                prefix = f"{args.endpoint.rstrip('/')}/{snapshot['repo']}/resolve/{snapshot['revision']}"
                restore(path, f'{prefix}/{quote(name)}', expected)
            else:
                verify(path, expected)
        print(f"Verified {role}: {len(snapshot['files'])} files at {snapshot['revision']}")


if __name__ == '__main__':
    main()
