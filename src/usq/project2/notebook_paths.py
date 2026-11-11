"""Resolve an explicitly chosen, trusted research snapshot before reading pickle files."""
from pathlib import Path
import os


REQUIRED_FILES = (
    'checksums.sha256', 'metadata/snapshot.json',
    *(f'panels/{name}.pkl' for name in ('open', 'high', 'low', 'close', 'volume', 'universe')),
)


def resolve_data_directory(root, configured=None, *, environ=None):
    """Use an explicit path, TEAM34_DATA_DIR, or <repo>/data; never guess another source."""
    env = os.environ if environ is None else environ
    raw = configured if configured is not None else env.get('TEAM34_DATA_DIR')
    root = Path(root).resolve()
    path = Path(raw).expanduser() if raw else root / 'data'
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
    return path
