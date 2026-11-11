"""Reject historical targets and cross-snapshot caches in the v2 experiment."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from usq.data.labels import NEXT_CLOSE_LABEL_POLICY, read_label_end
from usq.data.panels import PanelStore
from usq.database.common import digest


def validate_target(database, reference):
    root = Path(database)
    path = root / 'panels/label_metadata.json'
    if not path.exists():
        raise ValueError('Missing v2 label metadata; derive a next-close snapshot first')
    policy = json.loads(path.read_text())
    for key, expected in NEXT_CLOSE_LABEL_POLICY.items():
        if key not in policy or policy[key] != expected:
            raise ValueError(f'Incompatible label protocol ({key}); derive a next-close v2 snapshot')
    ends = read_label_end(PanelStore(root / 'panels'), reference)
    expected = pd.Series(reference.index, index=reference.index, name='label_end').shift(-11)
    if not ends.fillna('').equals(expected.fillna('')):
        raise ValueError('label_end must be the exact T+11 exchange session, with eleven censored tail rows')
    return ends


def bind_inputs(database, output, reference, *, create=False):
    """Bind all stages to actual label/universe bytes, not only field names."""
    root, out = Path(database), Path(output)
    ends = validate_target(root, reference)
    files = ['panels/label_metadata.json', 'panels/label_end.csv',
             'panels/ylabel.pkl', 'panels/universe.pkl']
    for optional in ('metadata/snapshot.json', 'checksums.sha256'):
        if (root / optional).exists():
            files.append(optional)
    signature = {name: digest(root / name) for name in files}
    path = out / 'input_signature.json'
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise ValueError('Database snapshot differs from this experiment; use a new output directory')
    elif create:
        path.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    else:
        raise ValueError('Missing v2 input signature; rerun the factor screen in a new output directory')
    return ends
