import json
import pytest
from usq.project2.strategy_study import create_experiment_config
from usq.project2.data_preparation import prepare_research_data


def test_runtime_configuration_has_no_historical_experiment_dependencies(tmp_path):
    p=create_experiment_config(tmp_path/'input',tmp_path/'work')
    cfg=json.loads(p.read_text())
    assert not set(cfg)&{'baseline','exploration','deep','late_fusion'}
    assert len(cfg['signals']['fundamental'])==6
    assert cfg['database']==str((tmp_path/'input').resolve())


def test_existing_prepared_data_are_rejected(tmp_path):
    source=tmp_path/'input';source.mkdir();(source/'checksums.sha256').write_text('')
    work=tmp_path/'work';(work/'database').mkdir(parents=True)
    with pytest.raises(RuntimeError,match='Clean-run policy'):
        prepare_research_data(source,work)


def test_standalone_initialization_never_requires_archived_paths(tmp_path, monkeypatch):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from usq.project2 import final_study

    dates = pd.Index(['20220103', '20220104'])
    frame = pd.DataFrame([[1.0], [1.1]], index=dates, columns=['A'])
    config = create_experiment_config(tmp_path/'input', tmp_path/'work')
    monkeypatch.setattr(final_study, 'ResearchMarket',
                        lambda _: SimpleNamespace(cap=np.ones(frame.shape)))
    monkeypatch.setattr(final_study.pd, 'read_pickle',
                        lambda p: frame * 0 + 1 if Path(p).name == 'universe.pkl' else frame.copy())
    monkeypatch.setattr(final_study, 'digest', lambda _: 'test-digest')
    monkeypatch.setattr(final_study, 'price_inputs', lambda *_: {})
    monkeypatch.setattr(final_study, 'wq', lambda *_: frame.copy())
    monkeypatch.setattr(final_study.FinalStudy, 'rebuild_fundamentals',
                        lambda self: self.parts.update({'factor': frame.copy()}))
    study = final_study.FinalStudy(config, tmp_path/'output', standalone=True)
    assert study.base is None
    assert study.common.to_numpy().all()
    pd.testing.assert_frame_equal(study.score, frame)


def test_model_worker_records_latest_string_label_date(tmp_path):
    import numpy as np
    import pandas as pd
    from usq.project2.model_selection import worker
    design = tmp_path/'design'; design.mkdir()
    output = tmp_path/'fit'; output.mkdir()
    dates = ['20180102','20180103','20180104','20190102']
    ends = ['20180118','20180119','20180122','20190118']
    (design/'axes.json').write_text(json.dumps({'index':dates,'columns':['A','B'],'label_end':ends}))
    np.save(design/'features.npy',np.random.default_rng(5205).normal(size=(4,2,12)).astype('float32'))
    np.save(design/'target.npy',np.full((4,2),.01,dtype='float32'))
    np.save(design/'allowed.npy',np.ones((4,2),dtype=bool))
    job = tmp_path/'job.json'
    job.write_text(json.dumps({'design':str(design),'folder':str(output),'kind':'ridge',
        'params':{'alpha_per_sample':.1},'half':'2019H1','start':'20190101','end':'20190630','signature':{}}))
    worker(job)
    meta=json.loads((output/'meta.json').read_text())
    assert meta['train_last_label_end']=='20180122'
    assert meta['train_last_label_end']<meta['forecast_start']
    assert (output/'complete.json').exists()
    assert pd.read_pickle(output/'prediction.pkl').index.tolist()==['20190102']
