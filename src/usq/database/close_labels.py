"""Derive a new immutable snapshot with unneutralized ten-session close targets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .reader import ResearchDatabase
from .common import digest, dump, save_panel
from .complete import write_checksums
from usq.data.labels import NEXT_CLOSE_LABEL_POLICY, close_forward_returns, daily_close_returns


def build_close_labels(source, output):
    source, output = Path(source), Path(output)
    if output.exists() or source.resolve() in output.resolve().parents:
        raise ValueError('Use a new output directory outside the source snapshot')
    db = ResearchDatabase(source)
    price = db.load('close_adj', verify_hash=True)
    segment = db.load('adjustment_segment', verify_hash=True)
    universe = db.load('universe', verify_hash=True).eq(1)
    for name, entry in db.meta['stock_fields'].items():
        if digest(source/'panels'/(name+'.pkl')) != entry['sha256']:
            raise ValueError('Source snapshot changed: '+name)
    derived = ['returns', 'forward_10d_raw', 'ylabel', 'label_10', 'label_available']
    if any(name in db.meta['stock_fields'] for name in derived[1:]):
        old_policy = db.meta.get('label_policy', {})
        if old_policy.get('formula') not in {
            'close_adj[t+10]/close_adj[t]-1', NEXT_CLOSE_LABEL_POLICY['formula']
        }:
            raise ValueError('Unknown source target convention; use a documented base or close10 snapshot')
    raw, end = close_forward_returns(price, 10, segment)
    daily = daily_close_returns(price, segment)
    if end.tail(11).notna().any() or raw.tail(11).notna().any().any():
        raise AssertionError('Unmatured end-of-snapshot labels must be missing')
    shutil.copytree(source, output, ignore=shutil.ignore_patterns('checksums.sha256', 'completion_verification.json'))
    old_note = output/'补齐说明.md'
    if old_note.exists():
        old_note.rename(output/'基础数据补齐说明.md')
    manifest = json.loads((output/'metadata/snapshot.json').read_text())
    fields = manifest['stock_fields']
    policy = {
        **NEXT_CLOSE_LABEL_POLICY,
        'aliases': ['forward_10d_raw', 'ylabel', 'label_10'],
        'standardized': False, 'neutralized': False, 'vwap_required': False,
        'returns_field': 'returns[t]=close_adj[t]/close_adj[t-1]-1; backward-looking input, not ylabel',
        'signal_time': 'after close at t; entry=t+1 close; label_end=t+11 exchange sessions',
        'execution_note': 'Next-close-aligned prediction target. First held return is t+1 close to t+2 close; a forecast horizon is not a mandatory holding period.',
        'segment_gate': 'All t..t+11 sessions in the same positive known adjustment_segment, including signal-to-entry identity; positive finite entry/exit prices; no price filling.',
        'universe_gate': 'Membership applied at signal date by consumers; future membership/survival not required for labels.',
        'maturity': 'Keep exact label_end; remove any training row whose label_end is not before the next prediction interval. Last eleven signal sessions are censored.',
        'missing_terminal_event_limit': 'Missing exit close (including delisting without executable quote) stays NaN, not zero. Original terminal-event ledger remains needed for reliable portfolio accounting.'}
    common = 'Next-close-aligned target close_adj[t+11]/close_adj[t+1]-1; ten holding intervals, eleven-session signal-to-exit offset. Unneutralized and unstandardized. No VWAP input. See label_policy.'
    for field in ['forward_10d_raw', 'ylabel', 'label_10']:
        fields[field] = save_panel(raw, output/'panels'/(field+'.pkl'), db.dates, db.sids)
        fields[field].update({'source': 'close_adj + adjustment_segment', 'unit': 'decimal ten-session return', 'notes': common})
    fields['returns'] = save_panel(daily, output/'panels/returns.pkl', db.dates, db.sids)
    fields['returns'].update({'source': 'close_adj + adjustment_segment', 'unit': 'decimal one-session return',
        'notes': 'Backward-looking daily adjusted-close return; distinct from preserved vendor daily_return, which can include terminal event returns.'})
    available = raw.notna().astype('uint8')
    fields['label_available'] = save_panel(available, output/'panels/label_available.pkl', db.dates, db.sids, 'mask')
    fields['label_available'].update({'source': 'finite forward_10d_raw', 'unit': '0/1', 'notes': 'Not a stock universe or tradability flag.'})
    end.to_csv(output/'panels/label_end.csv', index_label='date')
    dump(policy, output/'panels/label_metadata.json')
    manifest.update({'status': 'next_close10_v2_labeled_research_prototype', 'research_accepted': False,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'label_policy': policy, 'parent_snapshot_sha256': digest(source/'metadata/snapshot.json'),
        'parent_checksums_sha256': digest(source/'checksums.sha256') if (source/'checksums.sha256').exists() else None,
        'label_end_sha256': digest(output/'panels/label_end.csv'),
        'label_code_sha256': digest(Path(__file__)),
        'label_operator_code_sha256': digest(Path(__file__).parents[1]/'data/labels.py')})
    dump(manifest, output/'metadata/snapshot.json')
    dictionary = pd.read_csv(output/'metadata/data_dictionary.csv')
    added = [{'field': name, 'source': fields[name]['source'], 'unit': fields[name]['unit'], 'notes': fields[name]['notes']}
             for name in ['returns', 'forward_10d_raw', 'ylabel', 'label_10', 'label_available']]
    dictionary = dictionary.loc[~dictionary.field.isin(derived)]
    pd.concat([dictionary, pd.DataFrame(added)], ignore_index=True).to_csv(output/'metadata/data_dictionary.csv', index=False)
    research = (price.index >= '20150101') & (price.index <= '20251231')
    denominator = universe.loc[research].to_numpy().sum()
    good = raw.notna()
    coverage = float((good & universe).loc[research].to_numpy().sum()/denominator)
    audit = {'passed': True, 'research_accepted': False, 'shape': list(raw.shape),
        'label_coverage_on_supplied_2015_2025_universe': coverage,
        'label_missing_pct_on_supplied_2015_2025_universe': 100*(1-coverage),
        'last_signal_date_with_calendar_matured_target': str(end.dropna().index[-1]),
        'last_eleven_rows_all_missing': True, 'aliases_identical': True,
        'label_version': NEXT_CLOSE_LABEL_POLICY['label_version'],
        'source_snapshot_unchanged': True, 'vwap_used_for_label': False,
        'legacy_backtest_engine_migrated': True,
        'warning': 'Ordinary workflow/backtest/Barra paths use close_adj with next-close execution. Terminal events and missing held industry still require explicit source resolution; no formal full-period strategy has been validated.'}
    # Verify the emitted artifacts through the public reader, full axes before slices.
    child = ResearchDatabase(output)
    for name in ['returns', 'forward_10d_raw', 'ylabel', 'label_10', 'label_available']:
        frame = child.load(name, verify_hash=True)
        if name in ['forward_10d_raw', 'ylabel', 'label_10']:
            if not np.array_equal(frame.to_numpy(), raw.to_numpy(dtype='float32'), equal_nan=True):
                raise AssertionError('Target alias round-trip mismatch: '+name)
        del frame
    acceptance = json.loads((output/'metadata/acceptance.json').read_text())
    acceptance['label_policy'] = policy
    acceptance['label_coverage_on_supplied_universe'] = coverage
    acceptance['vwap_is_global_label_requirement'] = False
    acceptance['legacy_backtest_engine_migrated'] = True
    dump(acceptance, output/'metadata/acceptance.json')
    dump(audit, output/'metadata/label_audit.json')
    for name, entry in db.meta['stock_fields'].items():
        if digest(source/'panels'/(name+'.pkl')) != entry['sha256']:
            raise AssertionError('Original snapshot modified')
    note = f'''# 次日收盘入场的十日收益标签版（v2）

本目录是在已补齐数据基础上生成的独立派生快照，原补齐库、旧标签快照和下载文件均未修改；旧标签及旧训练结果不沿用为 v2 结果。

股票字段：{len(fields)} 个；完整面板 {len(db.dates):,} 日 × {len(db.sids):,} 个稳定 SID。日期 {db.dates[0]}—{db.dates[-1]}。

## 统一标签

`ylabel[T] = close_adj[T+11] / close_adj[T+1] - 1`

T+1 收盘入场，T+11 收盘退出，恰好十个持仓收益区间；均为交易日，不是自然日。该值等于完整日收益链的复合收益，不是十个简单日收益的算术和。`forward_10d_raw`、`ylabel`、`label_10` 完全相同；不做市值/行业中性化或截面标准化。`returns` 是向后看的单日复权收盘收益，原供应商 `daily_return` 原样保留。

每段必须保持同一个正且已知的复权分段，端点价格为正且有限，不跨未知证券/收益链接。未来股票池资格不参与标签计算，缺失价格不填零。`label_available` 为明确 0/1；`label_end.csv` 记录准确结束日期。训练切分必须按标签结束日期清除未到期样本。

2015—2025 原股票池内标签覆盖率约 **{coverage*100:.3f}%**，包括期末截尾造成的缺失。快照最后十一个信号交易日无完整入场后十日价格，标签全部缺失；最后一个日历上可到期的信号日为 **{audit['last_signal_date_with_calendar_matured_target']}**。

## 标签与交易不能混淆

标签是预测目标，不是收盘后还能按同一收盘价成交的收益。信号在 T 收盘后形成，实际回测 T+1 收盘执行、T+2 记录第一段持仓收益，按真实执行/记账时点计算持仓收益及费用；SPY 基准时点必须匹配。十日预测期限不等于强制每笔持仓十天，也不擅自改变已约定的调仓与平滑规则。

## VWAP 与研究范围

VWAP 不再是标签、统一研究样本或基础行情的强制输入。优先实现不依赖 VWAP 的 WorldQuant 原公式；依赖 VWAP 的公式允许输出 NaN，并记录滚动窗口后的实际覆盖率。不可把 close、OHLC 均价或不匹配的金额/成交量冒充真实 VWAP。

允许研究较低覆盖率因子，但需在 2015—2021 年冻结覆盖规则，同时报告逐年有效样本、行业/规模分布、同样本对照、RankIC 稳定性与扣费后结果。不得只凭少量样本上的高 RankIC 入选，也不得查看 2023—2025 表现后改变覆盖阈值。主对照模型仍保持相同股票池；不因某个因子缺失就整体删除该股票的历史。

## 未解决事项

每日股票数仍为 1,833—2,292，并未达到每日 4,000+；股票池规则、完整原始退市事件账本、最新财务修订偏差和证券级市值口径限制仍然存在。原始 VWAP 缺失不再是统一标签的阻塞，但仍影响依赖它的具体因子覆盖。

**常规 workflow/backtest/Barra 路径已迁移至 close_adj，明确 T+1 收盘执行、调仓日费用与前收盘风险暴露；退市无收盘价及持仓行业缺失仍需单独核实。** 这不等于整库已通过正式研究验收。本构建不筛选因子、训练模型或运行正式回测。旧 `close_sum` 与 `vwap_compound` 模式仅作为明确命名的兼容实现，不用于本快照。
'''
    (output/'标签与使用说明.md').write_text(note, encoding='utf-8')
    write_checksums(output)
    print(json.dumps({'output': str(output), 'stock_fields': len(fields), **audit}, ensure_ascii=False), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build_close_labels(args.source, args.output)


if __name__ == '__main__':
    main()
