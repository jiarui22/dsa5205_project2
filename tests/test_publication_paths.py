import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('verified_packager', ROOT/'scripts/package_verified_project.py')
packager=importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def test_data_code_is_included_but_database_is_not():
    assert packager.shareable_path('src/usq/data/labels.py')
    assert packager.shareable_path('src/usq/data/panels.py')
    assert not packager.shareable_path('data/panels/open.pkl')
    assert not packager.shareable_path('runs/example/metadata.json')


def test_only_aggregate_evidence_is_shareable():
    assert packager.shareable_path('reports/final/notebook_evidence/portfolio_summary.csv')
    assert not packager.shareable_path('predictions/model.joblib')
    assert not packager.shareable_path('.env')
    assert not packager.shareable_path('src/usq/__pycache__/module.pyc')
