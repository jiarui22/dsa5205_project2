"""Migration and leakage regressions for the next-close ten-interval target."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from usq.data.labels import NEXT_CLOSE_LABEL_POLICY, close_forward_returns
from usq.database.close_labels import build_close_labels
from usq.database.common import digest, dump, save_panel
from usq.database.complete import write_checksums
from usq.project2.models import Design
from usq.project2.protocol import bind_inputs, validate_target
from usq.project2.screen import POLICY


def snapshot(root, legacy=False):
    (root/'panels').mkdir(parents=True)
    (root/'metadata').mkdir()
    dates = pd.Index(pd.bdate_range('2021-12-01', periods=40).strftime('%Y%m%d'), name='date')
    sids = pd.Index(['A', 'B'], name='sid')
    price = pd.DataFrame(np.column_stack([100+np.arange(40)**2, 200+np.arange(40)]), index=dates, columns=sids)
    panels = {'close_adj': price, 'adjustment_segment': price*0+1, 'universe': price*0+1}
    pd.DataFrame({'date': dates}).to_csv(root/'panels/calendar.csv', index=False)
    pd.DataFrame({'sid': sids, 'column_order': [0, 1]}).to_parquet(root/'metadata/security_master.parquet')
    fields = {n: save_panel(p, root/'panels'/f'{n}.pkl', dates, sids) for n, p in panels.items()}
    manifest = {'stock_fields': fields, 'fundamental_fields': [], 'research_accepted': False}
    if legacy:
        y, ends = close_forward_returns(price, segment=panels['adjustment_segment'], delay=0)
        for n in ['ylabel', 'label_10', 'forward_10d_raw']:
            fields[n] = save_panel(y, root/'panels'/f'{n}.pkl', dates, sids)
        ends.to_csv(root/'panels/label_end.csv', index_label='date')
        manifest['label_policy'] = {'formula': 'close_adj[t+10]/close_adj[t]-1'}
    dump(manifest, root/'metadata/snapshot.json')
    dump({'research_accepted': False}, root/'metadata/acceptance.json')
    pd.DataFrame([{'field': n, 'source': 'synthetic', 'unit': 'test', 'notes': ''} for n in fields]).to_csv(root/'metadata/data_dictionary.csv', index=False)
    write_checksums(root)
    return price, panels['universe']


@pytest.mark.parametrize('legacy', [False, True])
def test_snapshot_migration_preserves_source_and_prices(tmp_path, legacy):
    source, out = tmp_path/'source', tmp_path/'v2'
    price, u = snapshot(source, legacy)
    before = {p.relative_to(source): digest(p) for p in source.rglob('*') if p.is_file()}
    audit = build_close_labels(source, out)
    assert before == {p.relative_to(source): digest(p) for p in source.rglob('*') if p.is_file()}
    assert digest(source/'panels/close_adj.pkl') == digest(out/'panels/close_adj.pkl')
    y = pd.read_pickle(out/'panels/ylabel.pkl')
    assert y.iloc[0, 0] == pytest.approx(price.iloc[11, 0]/price.iloc[1, 0]-1, rel=1e-6)
    assert y.tail(11).isna().all().all()
    assert audit['last_eleven_rows_all_missing']
    validate_target(out, u)
    dictionary = pd.read_csv(out/'metadata/data_dictionary.csv')
    assert not dictionary.field.duplicated().any()
    with pytest.raises(ValueError, match='new output'):
        build_close_labels(source, out)


def test_old_metadata_wrong_exit_and_cross_snapshot_reuse_are_rejected(tmp_path):
    source, out = tmp_path/'source', tmp_path/'v2'
    _, u = snapshot(source)
    build_close_labels(source, out)
    run = tmp_path/'run'; run.mkdir()
    bind_inputs(out, run, u, create=True)
    meta = out/'panels/label_metadata.json'
    original = meta.read_text()
    old = json.loads(original); old['entry_offset'] = 0
    dump(old, meta)
    with pytest.raises(ValueError, match='Incompatible label'):
        validate_target(out, u)
    meta.write_text(original)
    ends = out/'panels/label_end.csv'; original_end = ends.read_bytes()
    pd.Series(u.index, index=u.index, name='label_end').shift(-10).to_csv(ends, index_label='date')
    with pytest.raises(ValueError, match=r'T\+11'):
        validate_target(out, u)
    ends.write_bytes(original_end)
    y = pd.read_pickle(out/'panels/ylabel.pkl'); y.iloc[0, 0] += .01
    y.to_pickle(out/'panels/ylabel.pkl')
    with pytest.raises(ValueError, match='snapshot differs'):
        bind_inputs(out, run, u)


def test_actual_training_purge_uses_eleven_session_exit(tmp_path):
    source, db, out = tmp_path/'source', tmp_path/'v2', tmp_path/'run'
    _, u = snapshot(source)
    build_close_labels(source, db)
    out.mkdir(); (out/'factors').mkdir()
    policy_path = out/'research_protocol.json'; dump(POLICY, policy_path)
    selection = {'wq': ['WQ019'], 'fundamental': ['F01'], 'directions': {'WQ019': 1, 'F01': 1},
                 'protocol_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest()}
    dump(selection, out/'factor_selection.json')
    for n in ('WQ019', 'F01'):
        pd.DataFrame([[1., 2.]]*len(u), index=u.index, columns=u.columns).to_pickle(out/'factors'/f'{n}.pkl')
    bind_inputs(db, out, u, create=True)
    design = Design(db, out)
    _, _, record = design.training('20220101', 'A')
    assert record['train_last_signal'] == '20211216'
    assert record['train_last_label_end'] == '20211231'
    # Updating directions cannot silently reuse a signed feature cache.
    selection['directions']['WQ019'] = -1
    dump(selection, out/'factor_selection.json')
    with pytest.raises(ValueError, match='Feature cache differs'):
        Design(db, out)


def test_next_close_target_matches_ten_real_held_intervals():
    from usq.project2.ledger import Market
    dates = pd.Index(pd.bdate_range('2023-01-02', periods=15).strftime('%Y%m%d'))
    price = pd.DataFrame({'A': np.r_[1., 100+np.arange(14)**2], 'B': np.full(15, 100.)}, index=dates)
    m = Market.__new__(Market)
    m.index, m.columns = price.index, price.columns
    m.close = price.to_numpy()
    m.universe = m.tradable = np.ones(price.shape, dtype=bool)
    m.ret = price.pct_change(fill_method=None).fillna(0).to_numpy()
    m.terminal = np.zeros(price.shape, dtype=bool)
    m.source_return = np.zeros(price.shape)
    m.spy = np.zeros(len(price))
    scores = pd.DataFrame([[2., 1.]]*len(price), index=dates, columns=price.columns)
    daily, _, _, _ = m.backtest(scores, dates[0], dates[11], top_n=1, cost_bps=0, rebalance_every=100)
    target, _ = close_forward_returns(price)
    assert daily.iloc[1].gross_return == 0  # huge pre-fill move must be excluded
    assert daily.net_nav.iloc[-1]-1 == pytest.approx(target.iloc[0, 0])
