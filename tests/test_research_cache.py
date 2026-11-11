import json
import numpy as np
from usq.project2.research_cache import bundle_complete,finish_bundle,atomic_json


def test_interrupted_memmap_is_not_a_completed_cache(tmp_path):
    signature={'shape':[20,5,8]};required=['features.npy','coverage.csv']
    cache=np.lib.format.open_memmap(tmp_path/'features.npy',mode='w+',dtype='float32',shape=(20,5,8));cache[:3]=1;cache.flush();del cache
    assert np.load(tmp_path/'features.npy').shape==(20,5,8)  # valid header, partial data
    assert not bundle_complete(tmp_path,'complete.json',signature,required)
    (tmp_path/'coverage.csv').write_text('feature,coverage\na,1\n')
    assert not bundle_complete(tmp_path,'complete.json',signature,required)
    cache=np.lib.format.open_memmap(tmp_path/'features.npy',mode='w+',dtype='float32',shape=(20,5,8));cache[:]=2;cache.flush();del cache
    finish_bundle(tmp_path,'complete.json',signature,required)
    assert bundle_complete(tmp_path,'complete.json',signature,required)
    cache=np.load(tmp_path/'features.npy',mmap_mode='r+');cache[-1]=99;cache.flush();del cache
    assert not bundle_complete(tmp_path,'complete.json',signature,required)


def test_portfolio_marker_rejects_missing_corrupt_or_changed_inputs(tmp_path):
    signature={'rule':{'buffer':150,'rebalance':10}};required=['daily.csv','summary.json']
    (tmp_path/'daily.csv').write_text('date,return\n20200102,0\n');atomic_json(tmp_path/'summary.json',{'annual_return':0})
    finish_bundle(tmp_path,'complete.json',signature,required);assert bundle_complete(tmp_path,'complete.json',signature,required)
    assert not bundle_complete(tmp_path,'complete.json',{'rule':{'buffer':None,'rebalance':10}},required)
    assert not bundle_complete(tmp_path,'complete.json',signature,required+['weights.pkl'])
    (tmp_path/'daily.csv').unlink();assert not bundle_complete(tmp_path,'complete.json',signature,required)
    (tmp_path/'complete.json').write_text('{');assert not bundle_complete(tmp_path,'complete.json',signature,required)


def test_bundle_marker_is_published_only_after_all_outputs_exist(tmp_path):
    (tmp_path/'first').write_text('a')
    try:finish_bundle(tmp_path,'complete.json',{},['first','second'])
    except FileNotFoundError:pass
    else:raise AssertionError('Missing output must fail completion')
    assert not (tmp_path/'complete.json').exists()
