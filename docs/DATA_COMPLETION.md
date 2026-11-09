# Team17 本地数据补齐

本流程只读原始输入，在**新的目录**建立整合快照；不会改动下载文件，也不会自动推送任何文件到 GitHub。只用于团队可信 pickle 文件。运行需要匹配 `requirements-completion.txt` 的环境；已验证 Python 3.13.5。

## 输入

1. Team17 下载目录，含 `us_stock/panels` 与 `us_stock/metadata`。
2. 新提供的 `universe_daily.parquet`、`stocks.parquet`。
3. 旧本地数据库目录，含 `panels` 与 `metadata`。
4. 旧构建的 `_stage` 目录，含 `financial_features.parquet` 和 `quarterly_selected.json`。它的源财务 ZIP 哈希必须与旧数据库一致。
5. 已有 SPY 原始缓存 `SPY.gz` 及其同目录 `SPY.json` 来源及哈希说明。

每位组员应通过合法渠道保留相同版本的源文件，核对哈希后构建。学校授权的 Compustat/CRSP 文件、整套数据库及 pickle 不应上传公开仓库。补齐依赖已有财务构建中间文件；单靠两个新 Parquet 无法重建财务数据。

## 构建

在仓库根目录运行，替换下列示例路径；输出目录必须尚不存在。

```sh
PYTHONPATH=src python -m usq.database.complete \
  --team '/path/to/dsa5205_team17_data' \
  --parent '/path/to/old_database' \
  --stage '/path/to/old_build/_stage' \
  --universe '/path/to/universe_daily.parquet' \
  --stocks '/path/to/stocks.parquet' \
  --spy-cache '/path/to/yahoo/SPY.gz' \
  --output '/path/to/new/team17_completed_2014_2025'
```

本机本次输出为 `team17_completed_2014_2025`，2014-01-02—2025-12-31，3,018 个交易日、4,366 个稳定 SID。所有股票字段同轴，数值一般 float32，行业为字符串分类，股票池为 uint8 0/1。指数与 SPY 的列集合单独检查。

股票池使用新文件的集合及顺序，明确排除旧行情的额外 1,347 个 SID，不是悄悄取交集。2014 年价格保留作预热，但原始入池标记全为 0。2026 年 1 月混合来源尾部不纳入本整合库，原始文件仍保留；持有期未结束的年末标签必须截尾。

## 恢复与转换

- 根据已核对的 Compustat stage 恢复股数、现金、短债、单季度经营现金流、季度和 TTM 资本开支、单季度净利润及收入；财务美元金额从百万美元转换为美元，股数转换为实际股数。
- 公布时点采用有效 RDQ，否则有效 FDATEQ 代理，于下一交易日开始使用，向前延续，财报期末距当前超过 550 个自然日失效。不作向后填充。
- 7 个已有财务字段必须与 stage 展开结果逐单元一致。恢复字段若有旧宽表，也必须逐单元一致。
- 财务事件表保留各字段实际可用日期。季度事件表每个财务字段带独立的 `<field>_available_ts`；TTM、差分现金流可能晚于本季度自己的 RDQ，不能只用本行 `available_ts` 连接所有值。截止期后才可用的数值已屏蔽。
- 恢复财务版本标记、来源哈希、历史行业记录、公司季度 AJEXQ。AJEXQ 不是可直接使用的逐日总回报复权因子。
- 原始 open/high/low 从已提供复权价除以同日正复权因子恢复；不是重新取得的独立报价。原始 close 直接保留。
- 原有 `market_cap_split_synced` 原样保留，同时提供统一名称 `market_cap`，美元计价、CRSP **证券级**市值，不冒充多类别股份合并后的公司级总市值。
- 以历史有效日期投影 PERMNO，静态股票清单 PERMNO 不用于整个历史。投影不替代独立 CCM 原始连接档案。
- 对旧样本的真实 amount 与配对成交量，要求历史代码匹配、所有原始 OHLC 同时满足 0.1% 和 0.01001 美元误差上限、量价为正、隐含 VWAP 在当日区间 ±0.5% 内。通过的稀疏观察保存为 `amount`、`amount_source_volume`、`vwap_local_pair`；不使用 CRSP volume 反推另一个来源的金额，也不改变原有 vwap/vwap_adj。
- SPY 缓存校验哈希后提供独立的复权收盘、复权开盘和原始收盘表。复权收盘为供应商总回报代理，不是自建精确分红账本。恢复旧缓存价格指数表，不能把价格指数收益冒充总回报。
- 重建独立交易日历、字段字典、输入哈希、当前面板哈希和表格行数清单，并读回检查全部输出。

## 读取

安装项目或设置 `PYTHONPATH=src` 后：

```python
from usq.database import ResearchDatabase

db = ResearchDatabase('/path/to/team17_completed_2014_2025')
prices = db.stocks(['open', 'close', 'volume', 'market_cap'], '2023-01-01', '2025-12-31')
financials = db.fundamentals(
    ['cash', 'earnings_q', 'revenue_q', 'capital_expenditures_ttm'],
    '2023-01-01', '2025-12-31',
)
universe = db.load('universe', '2023-01-01', '2025-12-31')
industry = db.industry('2023-01-01', '2025-12-31')
spy_close = db.indices('total_return_close', '2023-01-01', '2025-12-31')
spy_open = db.indices('total_return_open', '2023-01-01', '2025-12-31')
price_indices = db.indices('price_close', '2023-01-01', '2025-12-31')
history = db.financial_history('2023-01-01', '2025-12-31')
```

`fundamentals()` 返回字段名到 DataFrame 的字典，财务宽表已按公布时点向前延续，不要再做 backfill。基本读取时每个字段检查全轴后才切片；`db.load(..., verify_hash=True)` 可额外校验对应股票字段哈希。

`universe_core_complete` 是原股票池与六项当前核心字段齐全的交集，仅为诊断补充，不悄悄替换原池。原股票池仍有少量核心字段缺失，交易可用标记也须独立处理。

## 验证

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_completion.py -v
```

小型测试不需要授权数据，覆盖禁止隐式丢轴、代数恢复、日期有效 PERMNO、历史冲突隔离、公布日前不填充、陈旧期限、TTM 单独可用时点、配对金额核验和基准哈希/身份检查。

## 仍未解决

- 每天只有 1,833—2,292 只，不满足每日 4,000+；股票池生成代码尚缺，不能承诺无未来筛选或完整证券类型筛查。
- 全市场真实 VWAP/金额仍缺。稀疏本地样本不是整个市场金额库，不能对缺失单元强填。
- 独立 CRSP 输入清单、完整 CCM 连接来源及原始退市事件账本未取得。`delisting_observations.parquet` 仅为已有标记的抽取，不是原始退市事件表。
- 财务是最新下载版本加公布日期代理，仍有修订偏差，不是严格 PIT。
- 公司级多类别股份市值、财务行业适用性及回测具体退出规则仍须在研究程序中明确。
- 保留的复权因子可能采用事后归一化锚点；量价因子需在信号日对窗口重锚，或使用消去公共尺度的运算，不能直接跨股票排序绝对复权价并宣称无未来信息。继承的 crsp_policy 仅记载上游流程，当前快照不含旧 10 日 VWAP 标签，也不含 2026 年尾部。

因此 `research_accepted=false` 继续保留。结构检查通过不等于已经完成正式因子研究或回测；本流程不做因子筛选、模型训练或交易规则选择。
