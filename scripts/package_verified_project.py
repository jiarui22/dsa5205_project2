"""Package verified source and aggregate evidence, excluding private experiment data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT_DIRS = {'.git', '.venv', 'data', 'runs', 'backups'}
CACHE_DIRS = {'__pycache__', '.pytest_cache'}
FORBIDDEN_SUFFIXES = {'.pkl', '.pickle', '.parquet', '.npy', '.npz', '.dta', '.joblib', '.zip', '.pem', '.key'}


def shareable_path(name):
    path = Path(name)
    return bool(path.parts) and path.parts[0] not in PRIVATE_ROOT_DIRS and not any(
        part in CACHE_DIRS for part in path.parts
    ) and path.suffix.lower() not in FORBIDDEN_SUFFIXES and not path.name.startswith('.env')


def package(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    notebook = json.loads((ROOT/'notebooks/Equity_Strategy_Research.ipynb').read_text())
    assert notebook['metadata']['clean_run_verification']['passed'] is True
    validation = json.loads((ROOT/'VALIDATION.json').read_text())
    assert validation['passed'] is True
    pages = len(PdfReader(ROOT/'reports/final/Group_34_Report.pdf').pages)
    assert 15 <= pages <= 25, pages
    listed = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            cwd=ROOT, check=True, capture_output=True).stdout.decode().split('\0')
    files = []
    for name in sorted(set(filter(None, listed))):
        path = Path(name)
        if not shareable_path(name):
            continue
        source = ROOT/path
        if not source.is_file():
            continue
        if source.is_symlink() or source.stat().st_size > 20_000_000:
            raise ValueError(f'Unexpected publication file: {name}')
        files.append({'path':name, 'bytes':source.stat().st_size,
                      'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'files':files, 'raw_data_included':False, 'fitted_models_included':False,
                'report_pages':pages, 'validation':validation}
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in files:
            archive.write(ROOT/entry['path'], 'dsa5205-project2-verified/'+entry['path'])
        archive.writestr('dsa5205-project2-verified/SHARE_MANIFEST.json', json.dumps(manifest, indent=2))
    print(json.dumps({'archive':str(destination),'files':len(files), 'report_pages':pages,
                      'bytes':destination.stat().st_size,
                      'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    package(parser.parse_args().output)
