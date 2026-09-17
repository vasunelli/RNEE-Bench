"""Download and install the checksum-pinned scientific data without overwriting it."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_payload(root, spec):
    index_path = root / 'checksums.json'
    if not index_path.is_file() or digest(index_path) != spec['checksums_sha256']:
        raise ValueError('The scientific-data checksum index is missing or differs from the release.')
    index = json.loads(index_path.read_text(encoding='utf8'))
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(index) | {'checksums.json'} or len(actual) != spec['file_count']:
        raise ValueError('Scientific-data file inventory differs from the release.')
    for name, expected in index.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError('Scientific-data file differs from the release: ' + name)


def validate_archive(path, spec):
    if path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
        raise ValueError('Archive size or SHA-256 differs from data/release.json.')


def install(archive, target, spec):
    command = shutil.which('7z') or shutil.which('7zz') or shutil.which('7za')
    if not command:
        raise ValueError('Install 7-Zip or 7zz and make its command available on PATH.')
    parent = target.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.energy-extract-', dir=parent) as temporary:
        staging = Path(temporary).resolve()
        if not staging.is_relative_to(parent):
            raise ValueError('Unexpected extraction directory.')
        subprocess.run([command, 'x', '-y', '-bsp0', '-bso0', str(archive), '-o' + str(staging)], check=True)
        payload = (staging / spec['archive_root']).resolve()
        if not payload.is_relative_to(staging) or target.exists() or not target.is_relative_to(parent):
            raise ValueError('The destination changed or extraction path is invalid.')
        validate_payload(payload, spec)
        payload.rename(target)


def main():
    spec = json.loads((REPOSITORY / 'data/release.json').read_text(encoding='utf8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded archive, without network access.')
    parser.add_argument('--data-dir', type=Path, default=REPOSITORY / spec['local_directory'])
    parser.add_argument('--verify-only', action='store_true', help='Check existing scientific data without downloading.')
    args = parser.parse_args()
    target = args.data_dir.expanduser().resolve()
    if target.exists():
        validate_payload(target, spec)
        print('Existing scientific data match the published release:', target)
        return 0
    if args.verify_only:
        parser.error('Scientific data are not installed at ' + str(target))
    if args.archive:
        archive = args.archive.expanduser().resolve()
        if not archive.is_file():
            parser.error('Archive does not exist: ' + str(archive))
        validate_archive(archive, spec)
        install(archive, target, spec)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.energy-download-', dir=target.parent) as temporary:
            staging = Path(temporary).resolve()
            if not staging.is_relative_to(target.parent):
                raise ValueError('Unexpected download directory.')
            archive = staging / spec['archive_name']
            print('Downloading', spec['bytes'], 'bytes from', spec['tag'], flush=True)
            request = urllib.request.Request(spec['url'], headers={'User-Agent': 'RNEE-Bench-data'})
            with urllib.request.urlopen(request, timeout=120) as response, archive.open('xb') as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            validate_archive(archive, spec)
            install(archive, target, spec)
    print('Scientific data installed and verified:', target)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit('Data installation failed: ' + str(error))
