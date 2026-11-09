# 美股量化研究系统

日频数据接口 → 十日复权收盘收益标签 → 因子研究与三层评估 → 累积历史半年重训 → 收盘执行回测与风格分析。常规链路已移除 VWAP 依赖，终止事件等例外见下文。

组员搭建与分工请阅读 [详细说明书](docs/HANDBOOK.md)。数据由数据组寻找和整理，本仓库不附带真实行情、财报或策略收益结果。

## Project2：已实现的研究入口

`src/usq/project2/` 是本次课程研究入口，与下方旧版递归挖掘及单模型 CLI 分开。它复现六个研报基本面因子、评估 36 个 WorldQuant 候选，按 2015-2021 因子筛选、2022 验证、2023-2025 半年扩展训练运行 Ridge/Lasso/XGBoost 的量价 A 与融合 B 对照。实际去重后保留四个 WQ 因子，不宣称实现全部 101 个。依赖已在 Python 3.13.5 验证；真实行业 pickle 需要该依赖环境。

从仓库根目录运行，`--database` 指向团队各自拥有的同一数据库快照。不要把商业数据或模型 pickle 上传公开仓库。

```bash
python3.13 -m venv .venv-project2
source .venv-project2/bin/activate
python -m pip install -r requirements-project2.txt
# macOS XGBoost 另需 brew install libomp
PYTHONPATH=src python -m usq.project2.run --database ../team17_close10_2014_2025 --output ../project2_research_fresh --all
PYTHONPATH=src python -m usq.project2.supplement --database ../team17_close10_2014_2025 --output ../project2_research_fresh
PYTHONPATH=src python -m usq.project2.risk --database ../team17_close10_2014_2025 --output ../project2_research_fresh --supplement
MPLCONFIGDIR=/tmp/usq-project2-mpl PYTHONPATH=src python -m usq.project2.paper --database ../team17_close10_2014_2025 --output ../project2_research_fresh --pdf-output ../project2_research_fresh/report.pdf
PYTHONPATH=src python -m pytest tests -o addopts='' -q
```

预期产物：`factor_selection.json`、`training_history.json`、六组预测与持仓、`backtests/summary.csv`、`barra_style/`、`supplement/`、离线结果页和英文 PDF。补充分析在主结果之后运行，不能当作新的盲测，也不能回流重选主因子、主模型或交易规则。

简化研究账本只存在于 `project2/ledger.py`：使用 close_adj、T+1 收盘执行、实际成交日收费；不能买入的目标留现金，不能卖出的旧仓保留，终止和估值例外逐笔记录。仓库严格回测引擎仍对未知持仓收益报错。财务按发布日期向后展开但不是严格历史 PIT。SPY 是独立绩效基准，Barra-style 不冒充商业 MSCI 风险模型。

## 本地数据库试验版

新增 Compustat 季度数据分块接入、财政季度/累计口径转换、公布日期向后延续、独立日历、统一字段宽表、原始行情缓存及读取接口。主研究期为 **2015–2025**，另有 2014 预热和 2026 年初标签尾部。运行步骤与来源限制见 [数据库搭建说明](docs/DATABASE_BUILD.md)。

当前为限制明确的试验版：全候选池基本面，行情先做代表性样本；历史证券身份和全量行情未验收。不能把“候选列数”当成“已完成行情的股票数”，也不能直接送入正式回测。个人配置、原始数据、pickle/Parquet 和下载缓存均被 Git 忽略。

Team17 本地补齐流程见 [DATA_COMPLETION.md](docs/DATA_COMPLETION.md)。Project2 主标签为 `close_adj[T+10]/close_adj[T]-1`，不要求 VWAP，见 [CLOSE10_PROTOCOL.md](docs/CLOSE10_PROTOCOL.md)。回测、分层、Barra 常规收益链已改为 close_adj，并明确 T+1 收盘执行；[迁移说明](docs/CLOSE_EXECUTION_MIGRATION.md)列出不可直接替换的终止事件、原因子公式和 SPY 风格基准限制。程序能运行不等于整库通过正式研究验收。

## 安装和验证

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
usq demo --output runs/demo --full
```

`demo --full` 使用人工数据，实际训练六个半年模型，每个固定 500 棵树，明确使用 CPU。生成自包含的回测和 Barra HTML。它不验证真实美股 Alpha，也不替代服务器 CUDA 验收。Mac 若缺 OpenMP，需安装 `libomp`；生产环境建议 Linux。

## 正式运行顺序

先准备 `data/panels/` 与独立交易日历 `calendar.csv`，按说明书统一稳定证券 ID、复权、PIT 基本面和退市处理。路径可在 `configs/default.yaml` 修改。所有命令从仓库根目录执行。

```bash
usq audit-data
usq prepare
usq mine
usq evaluate --level simple --stage mining
usq evaluate --level simple --stage validation
usq barra-build
usq evaluate --level complex --stage validation --barra
usq evaluate --level correlation --stage mining
usq select
usq train
usq backtest --barra
```

`select` 未通过既定规则时明确中止，不会为凑数量强行保留因子。首次正式运行前应冻结候选预算、阈值、交易成本和数据版本。

## 代码分类

| 目录 | 功能 |
|---|---|
| `src/usq/data/` | 字段宽表、PIT 展开、多来源比较、数据审计、VWAP 和标签 |
| `src/usq/evaluation/` | 简单、复杂、因子间三层独立调用 |
| `src/usq/mining/` | 原版递归生成器与算子、多进程批次、量纲/跨层过滤、已有因子扩展、断点日志及 IC 导出 |
| `src/usq/model/` | 单模型 XGBoost，2015 起扩展训练，2023–2025 每半年重训 |
| `src/usq/backtest/` | 复权收盘收益、次日收盘调仓、真实调仓日费用、漂移、期末退出及分层 |
| `src/usq/barra/` | 参考描述子/合成/正交化核心、美股适配、固定模型库、持仓归因与原报告扩展 |
| `src/usq/research/` | 候选全集的区块 Bootstrap 验证研究工具 |
| `src/usq/report.py` | 离线净值、回撤、分组、暴露及归因可视化 |
| `tests/` | 时间对齐、purge、CV、PIT、缺失值、归因恒等式等测试 |

## 口径与实现边界

| 基础指标 | 定义 |
|---|---|
| IC | 日因子值与标签的截面 Pearson 相关 |
| RankIC | 日因子排名与标签排名的截面相关 |
| ICIR | 日 IC 均值除以日 IC 标准差；RankICIR 同理 |
| CV | 每日 RankIC 为正时除以前两个日历年历史最小值，为负时除以历史最大值；不含当天，历史不足或零分母记为缺失 |

- 默认 `label.mode: close_compound` 为 `close_adj[T+10]/close_adj[T]-1`，主 `ylabel`/`label_10` 不中性化或标准化。`vwap_compound` 与算术相加的 `close_sum` 仅保留为明确命名的旧标签口径，当前人工 demo 也采用收盘主链路。预测标签不是收盘后能按同一收盘价成交的收益。
- 模型使用截图参数，固定 500 树，不早停、不四折、不 bagging。`missing: null` 转为原生接口的 `np.nan`，零值仍是有效值。
- 回测为 T 收盘后信号、T+1 收盘近似执行，费用在调仓日计入；支持低频漂移和默认期末清仓。仍是权重研究模拟，不含完整终止结算、真实撮合或容量冲击账本。缺失持仓收益和不可执行交易报错，不未来重选。
- Barra 沿用参考实现的描述子、标准化、合成、正交化、WLS 和收益归因核心，改为美股收盘收益与前收盘暴露适配；旧 VWAP 缓存需重建。SPY 仅作为绩效基准，同池市值风格参考不冒充 SPY 持仓归因；不是授权 MSCI USE4 产品。
- 101/191 提供完整编号的批量导入、缺失审计和统一评估接口，**未内置全部 292 条标准公式实现**。可将经核对的外部公式输出接入；`--require-complete` 会阻止缺号时声称全库完成。
- 默认 CUDA；若 GPU 不可用会拒绝静默回退。CPU 运行须在配置中明确指定。
- 爆搜默认使用 `FactorExpressionGenerator` 与配套 `Operators`，可配置复杂度、窗口、量纲类别、已有因子和并行进程。原版高层双表达式配对每层最多抽取 100 对；它不是所有树结构的完全穷举。旧模板仅用于历史日志读取，不再作为默认生成器。

计算口径见 [实现口径记录](docs/REFERENCE_AUDIT.md)，迁移边界见 [递归引擎核对](docs/ENGINE_PORT.md)，验证范围见 [验证记录](docs/VALIDATION.md)。
