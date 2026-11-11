from pathlib import Path
import pytest
from usq.project2.notebook_paths import REQUIRED_FILES, resolve_data_directory


def snapshot(path):
    for name in REQUIRED_FILES:
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    return path.resolve()


def test_default_snapshot(tmp_path):
    data = snapshot(tmp_path / 'data')
    assert resolve_data_directory(tmp_path, environ={}) == data


def test_environment_and_explicit_precedence(tmp_path):
    a, b = snapshot(tmp_path / 'a'), snapshot(tmp_path / 'b')
    assert resolve_data_directory(tmp_path, environ={'TEAM34_DATA_DIR': str(a)}) == a
    assert resolve_data_directory(tmp_path, b, environ={'TEAM34_DATA_DIR': str(a)}) == b


def test_relative_directory_is_relative_to_repo(tmp_path):
    data = snapshot(tmp_path / 'inputs' / 'snapshot')
    assert resolve_data_directory(tmp_path, 'inputs/snapshot', environ={}) == data


def test_missing_data_fails_before_pickle_loading(tmp_path):
    with pytest.raises(FileNotFoundError, match='TEAM34_DATA_DIR') as error:
        resolve_data_directory(tmp_path, environ={})
    assert 'panels/open.pkl' in str(error.value)
    assert not (tmp_path / 'data').exists()


def test_rejects_parent_or_panels_directory(tmp_path):
    data = snapshot(tmp_path / 'snapshot')
    for wrong in (tmp_path, data / 'panels'):
        with pytest.raises(FileNotFoundError):
            resolve_data_directory(tmp_path, wrong, environ={})
