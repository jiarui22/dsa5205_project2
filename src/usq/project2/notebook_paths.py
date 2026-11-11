"""Resolve an explicitly chosen, trusted research snapshot before reading pickle files."""
from pathlib import Path
import os
import json
from usq.database.common import digest


REQUIRED_FILES = (
    'checksums.sha256', 'metadata/snapshot.json',
    'metadata/preprocessing_complete.json',
    'metadata/activation.json',
    *(f'panels/{name}.pkl' for name in ('open', 'high', 'low', 'close', 'volume', 'universe')),
)


def resolve_data_directory(root, configured=None, *, environ=None):
    """Use an explicit path, TEAM34_DATA_DIR, or the project research snapshot."""
    env = os.environ if environ is None else environ
    raw = configured if configured is not None else env.get('TEAM34_DATA_DIR')
    root = Path(root).resolve()
    if raw:
        path = Path(raw).expanduser()
    else:
        pointer = root/'data/processed/latest_snapshot.json'
        if not pointer.is_file():
            raise FileNotFoundError('Run Equity_Data_Preparation.ipynb from top to bottom first: no completed upstream snapshot is published.')
        info = json.loads(pointer.read_text())
        path = (root/'data'/info['snapshot']).resolve()
        if not path.is_relative_to(root/'data/processed'):
            raise ValueError('Published snapshot must be inside project data/processed/')
        producer = root / info['producer']
        if not (producer/'complete.json').is_file() or not json.loads((producer/'complete.json').read_text()).get('complete'):
            raise ValueError('The upstream data run is missing or incomplete; rerun data preparation.')
        if digest(path/'checksums.sha256') != info['checksums_sha256']:
            raise ValueError('Published snapshot changed after data preparation')
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f'Research snapshot not found or incomplete: {path}\n'
            f'Missing: {", ".join(missing)}\n'
            'This Git repository intentionally excludes the database. Set DATA_DIRECTORY '
            'in the first notebook code cell to the trusted snapshot folder that directly '
            'contains panels/, metadata/, and checksums.sha256; alternatively set '
            'TEAM34_DATA_DIR before starting Jupyter. Do not point to panels/ itself '
            'or to its parent download directory. No input files have been modified.'
        )
    if not json.loads((path/'metadata/preprocessing_complete.json').read_text()).get('pipeline_complete'):
        raise ValueError('This snapshot was not completed by the data preparation notebook')
    if not json.loads((path/'metadata/activation.json').read_text()).get('producer_complete'):
        raise ValueError('Finish the upstream data notebook before starting strategy research')
    return path
