import importlib.util
import json
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('notebook_report_builder', ROOT/'scripts/build_notebook_report.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_report_rejects_notebook_without_clean_run(tmp_path):
    notebook = tmp_path/'notebook.ipynb'
    notebook.write_text(json.dumps({'metadata': {}, 'cells': []}))
    with pytest.raises(ValueError, match='fresh-kernel'):
        builder.build(tmp_path/'missing', notebook, tmp_path/'report')
    assert not (tmp_path/'report').exists()


def test_report_rejects_out_of_order_outputs(tmp_path):
    notebook = tmp_path/'notebook.ipynb'
    notebook.write_text(json.dumps({
        'metadata': {'clean_run_verification': {'passed': True}},
        'cells': [{'cell_type': 'code', 'execution_count': n, 'outputs': []} for n in [1, 3, 2]],
    }))
    with pytest.raises(ValueError, match='Nonsequential'):
        builder.build(tmp_path/'missing', notebook, tmp_path/'report')


def test_report_rejects_error_outputs(tmp_path):
    notebook = tmp_path/'notebook.ipynb'
    notebook.write_text(json.dumps({
        'metadata': {'clean_run_verification': {'passed': True}},
        'cells': [{'cell_type': 'code', 'execution_count': 1,
                   'outputs': [{'output_type': 'error', 'ename': 'NameError'}]}],
    }))
    with pytest.raises(ValueError, match='error outputs'):
        builder.build(tmp_path/'missing', notebook, tmp_path/'report')


def test_report_helpers_preserve_units():
    assert builder.pct(.249181633943254) == '24.92'
    assert builder.number(1.4074027) == '1.407'
    assert builder.escape('fundamental_cap') == r'fundamental\_cap'
