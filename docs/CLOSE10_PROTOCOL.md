# Project2：十日收盘收益标签与可缺失 VWAP

## 主标签

```text
returns[T] = close_adj[T] / close_adj[T-1] - 1
ylabel[T]  = close_adj[T+10] / close_adj[T] - 1
```

T+10 指交易日，不是自然日。十日日收益链的复合等于端点比值，不是十个简单日收益的算术和。`forward_10d_raw`、`ylabel`、`label_10` 为完全相同的未中性化、未标准化标签。向后看的 `returns` 可供量价因子使用，未来标签不可混入特征。

用同一正且已知的 `adjustment_segment` 覆盖整个 T—T+10 区间，拒绝跨未知证券身份或收益连接。只要分段连续性已有证据，中间缺报价不必使端点收益失效；端点缺失、非正或无穷、分段缺失或变更均使标签缺失，不填零。没有分段资料时，通用函数保守要求整个窗口有有效价格。

保存 `label_end.csv`，训练时必须要求结束日期早于对应预测期起点。最后十个交易日尚无到期标签，保留 NaN，不能从混合来源的 2026 尾部悄悄补入。

## 生成独立快照

使用与已补齐数据库一致的可信 pickle 读取环境，在仓库根目录：

```sh
PYTHONPATH=src python -m usq.database.close_labels \
  --source '/path/to/team17_completed_2014_2025' \
  --output '/path/to/new/team17_close10_2014_2025'
```

输出目录必须尚不存在；原库保留不变。新增五个同轴股票字段：`returns`、`forward_10d_raw`、`ylabel`、`label_10`、`label_available`。更新数据字典、快照、哈希和标签覆盖检查。已知源字段先核验哈希，目标列不取交集、不缩股票池。

新默认 `prepare(mode='close_compound')` 也可在通用 PanelStore 中生成同口径标签，只依赖 `close_adj`（或 `close*adj_factor`），可选 `adjustment_segment`；不需要 volume、amount、VWAP、行业或市值。真实项目建议使用上述独立快照入口，保留版本沿革。旧 `vwap_compound` 与 `close_sum` 仍可显式调用，但不属于当前主协议。

## 预测与执行分离

T 收盘后才形成的信号，不能假设按刚观察到的 T 收盘价成交。十日标签只是模型预测目标，回测仍在最早 T+1 执行、按真实持仓和执行／记账价格计算收益及费用。调仓间隔与十日预测期限不是一回事，不能把原有调仓规则默默改成十日锁仓。

`workflow`、回测和 Barra 常规收益适配现已迁移到 close_adj，明确 T+1 收盘执行、实际调仓日收费、前收盘风险暴露；人工 demo 也使用新口径。退市等终止事件仍不能机械转成普通收盘收益，原始 VWAP 因子也不改公式，详见 [迁移说明](CLOSE_EXECUTION_MIGRATION.md)。旧 VWAP 风险缓存须重建。

## VWAP 缺失政策

- 不以全市场 VWAP 完整度作为统一标签或整个研究的验收门槛；优先实现不依赖 VWAP 的原始公式。
- 依赖 VWAP 的公式保留真实缺失，不用 close、OHLC 均价或错误配对的金额／成交量替代后继续冒充原公式。
- 覆盖率统计针对完整算子及滚动窗口后的因子结果，不以原始 VWAP 的覆盖率代替最终因子覆盖。
- IC/RankIC 使用有效因子、有效标签与信号日股票池的配对样本；披露逐日样本数、逐年覆盖率、规模／行业构成。有效股票太少时不输出该日相关性。
- 低覆盖率因子可以研究，但不能仅凭少数样本的高 RankIC 入选；做同样本基线、分期稳定性、共同样本稳健性及扣费后表现检查。覆盖阈值在 2015—2021 开发期冻结，不能查看 2023—2025 收益后更改。
- 主模型 A/B 保持相同股票池和样本资格政策，不因一个因子缺失就整体删除股票。Ridge/Lasso 的填补器及缺失标记只能在当期训练数据上拟合；XGBoost 可保留原生 NaN，零值仍是有效值。

股票池每日 4,000+ 未达标、生成规则未取得、退市事件原始账本缺失，以及最新财务下载的修订偏差仍是独立问题，不能因为标签不需要 VWAP 就一并视为解决。

## 小型测试

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_close_labels.py -v
```

本流程只更新标签与研究协议，没有进行正式因子筛选、模型训练或回测。
