"""Run every notebook cell once in a fresh kernel, saving progress and failures."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/notebook_validation'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    from usq.project2.notebook_paths import resolve_data_directory
    data = resolve_data_directory(root, args.data)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    executed = output / 'Equity_Strategy_Research.executed.ipynb'
    if executed.exists():
        raise FileExistsError('Use a new validation output directory.')
    os.environ['TEAM34_DATA_DIR'] = str(data)
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'mpl'))
    os.environ.setdefault('JUPYTER_RUNTIME_DIR', str(output / 'jupyter_runtime'))
    kernels = output / 'kernels'
    spec = kernels / 'team34-validation'
    spec.mkdir(parents=True, exist_ok=True)
    (spec / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Team 34 clean validation', 'language': 'python',
    }))
    manager = KernelManager(kernel_name='team34-validation',
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    notebook = nbformat.read(root / 'notebooks/Equity_Strategy_Research.ipynb', as_version=4)
    for cell in notebook.cells:
        cell.metadata.pop('execution', None)
        if cell.cell_type == 'code':
            cell.outputs, cell.execution_count = [], None
    started = time.monotonic()
    status = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'completed_cells': [], 'passed': False}

    def save():
        nbformat.write(notebook, executed)
        (output / 'execution_status.json').write_text(json.dumps(status, indent=2))

    def begin(cell, cell_index, **kwargs):
        status['active_cell'] = cell_index
        (output / 'execution_status.json').write_text(json.dumps(status, indent=2))
        print(json.dumps({'event': 'cell_start', 'cell': cell_index,
                          'source': cell.source.splitlines()[0] if cell.source else '',
                          'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)

    def complete(cell, cell_index, **kwargs):
        status['completed_cells'].append(cell_index)
        status['elapsed_seconds'] = round(time.monotonic() - started, 1)
        save()
        print(json.dumps({'event': 'cell_complete', 'cell': cell_index,
                          'execution_count': cell.execution_count,
                          'elapsed_seconds': status['elapsed_seconds']}), flush=True)

    client = NotebookClient(notebook, km=manager, timeout=None,
                            resources={'metadata': {'path': str(root)}},
                            allow_errors=False, on_cell_start=begin, on_cell_executed=complete)
    save()
    try:
        client.execute(cleanup_kc=True)
        code = [cell for cell in notebook.cells if cell.cell_type == 'code']
        counts = [cell.execution_count for cell in code]
        assert counts == list(range(1, len(code) + 1)), counts
        assert not any(out.output_type == 'error' for cell in code for out in cell.outputs)
        status.update(passed=True, code_cells=len(code), execution_counts=counts,
                      elapsed_seconds=round(time.monotonic() - started, 1))
        notebook.metadata['clean_run_verification'] = {
            'passed': True, 'code_cells': len(code), 'execution_counts': counts,
            'completed_utc': datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps({'event': 'run_all_passed', **status}), flush=True)
    except Exception as error:
        status.update(error=f'{type(error).__name__}: {error}',
                      elapsed_seconds=round(time.monotonic() - started, 1))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
