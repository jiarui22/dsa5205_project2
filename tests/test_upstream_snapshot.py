import json
from pathlib import Path
import pytest
from usq.project2.notebook_paths import resolve_data_directory, REQUIRED_FILES
from usq.database.common import digest
from usq.database.strategy_snapshot import activate_snapshot


def test_missing_pointer_never_falls_back_to_reference_snapshot(tmp_path):
    (tmp_path/'data/processed/research_snapshot').mkdir(parents=True)
    with pytest.raises(FileNotFoundError,match='Equity_Data_Preparation'):
        resolve_data_directory(tmp_path,environ={})


def test_only_completed_unchanged_upstream_snapshot_can_be_selected(tmp_path):
    data=tmp_path/'data';snapshot=data/'processed/new_run/research_snapshot'
    for name in REQUIRED_FILES:
        p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    (snapshot/'metadata/activation.json').unlink()
    (snapshot/'metadata/preprocessing_complete.json').write_text(json.dumps({'pipeline_complete':True}))
    producer=tmp_path/'runs/new_run';producer.mkdir(parents=True)
    (producer/'complete.json').write_text(json.dumps({'complete':False}))
    with pytest.raises(ValueError,match='completed data notebook'):
        activate_snapshot(data,snapshot,producer)
    assert not (data/'processed/latest_snapshot.json').exists()
    (producer/'complete.json').write_text(json.dumps({'complete':True}))
    activate_snapshot(data,snapshot,producer)
    assert resolve_data_directory(tmp_path,environ={})==snapshot
    (snapshot/'checksums.sha256').write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        resolve_data_directory(tmp_path,environ={})


def test_incomplete_producer_cannot_be_reused_after_publication(tmp_path):
    data=tmp_path/'data';snapshot=data/'processed/new_run/research_snapshot'
    (snapshot/'metadata').mkdir(parents=True);(snapshot/'checksums.sha256').write_text('hashes')
    producer=tmp_path/'runs/new_run';producer.mkdir(parents=True)
    (producer/'complete.json').write_text(json.dumps({'complete':True}))
    activate_snapshot(data,snapshot,producer)
    (producer/'complete.json').unlink()
    with pytest.raises(ValueError,match='missing or incomplete'):
        resolve_data_directory(tmp_path,environ={})
