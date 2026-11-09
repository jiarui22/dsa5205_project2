# DSA5205 Project 2 Team 34

美股基本面与量价融合选股研究的代码、英文报告和已完成结果。供 Team 34 继续完善报告、录制视频及复现实验。主结果已经冻结；不要看到 2023-2025 的收益后再重选因子或参数，并将其称为新的样本外结果。

## 先看报告和交接清单

- [英文报告初稿，25 页，13 张图](reports/Team34_Project2_Report_Draft.pdf)
- [接手说明和待办](docs/TEAM34_HANDOFF.md)
- [中文解读与视频建议](reports/Team34_研究解读与交付清单.html)
- [离线研究结果页](results/project2/研究结果.html)（下载后用浏览器打开）
- [主回测汇总](results/project2/backtests/summary.csv)
- [补充组合汇总](results/project2/supplement/portfolio_summary.csv)

PDF 是 2026-11-09 成稿时的快照，文中“未推送”的描述指成稿时状态；代码和报告现已放入本仓库。尚未提供组员姓名、学号和真实个人分工，正式提交前请补齐。

## 已经完成的研究

复现 6 个基本面因子；实现并评估 36 个 WorldQuant 候选，按开发期规则去重后保留 WQ019、WQ040、WQ026、WQ052。Ridge、Lasso、XGBoost 各运行纯量价 A 与融合 B 两组，训练历史从 2015 年累积，每半年从头拟合，不使用固定长度训练窗口。

| 时间 | 用途 |
| --- | --- |
| 2015-2018 | 因子初筛与方向 |
| 2019-2021 | 稳定性与去重 |
| 2022 | 小网格参数和共同交易规则验证 |
| 2023-2025 | 六个半年预测期，固定规则回测 |

收益标签为 `close_adj[T+10]/close_adj[T]-1`。交易采用 T 收盘后信号、T+1 收盘执行、T+2 计首段持仓收益；Top100 等权、每 5 个交易日调仓、不额外平滑，每成交 1 美元收取 10bps。

| 模型 | A 净年化 | B 净年化 |
| --- | ---: | ---: |
| Ridge | 9.68% | 8.89% |
| Lasso | 9.02% | 8.82% |
| XGBoost | 10.56% | 10.10% |

同期 SPY 净值基准年化约 22.86%。加入基本面降低了换手，但没有提高这套学习流程的净复合收益；成对区块置信区间跨零。事后补充的基本面等权评分组合年化约 19.88%，只作诊断，不替换主策略。

## 安装环境

使用 Python 3.13。已在 Python 3.13.5、pandas 3.1.1 上运行；较旧 pandas 可能不能读取团队行业 pickle。所有命令从仓库根目录运行。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# macOS 的 XGBoost 另需 OpenMP
brew install libomp
```

最后一条仅适用于 macOS。Linux 请使用对应平台的 OpenMP 运行库。

## 没有全量数据库也能先改报告

仓库包括报告使用的汇总 CSV、逐日组合收益、图表和协议，不包括个股行情、财报、预测宽表或持仓。可以直接修改 `src/usq/project2/paper.py`，然后仅从公开汇总结果重新生成报告：

```bash
MPLCONFIGDIR=/tmp/team34-mpl PYTHONPATH=src python -m usq.project2.paper \
  --output results/project2 \
  --pdf-output reports/rebuilt/Team34_Project2_Report_Draft.pdf
```

此入口不加载数据库、不重新选因子或训练模型。它会重建 `results/project2/paper_figures/` 中的图表及报告元数据；重建 PDF 输出到忽略目录，不覆盖已审核的英文报告。图表数据可在 `results/project2/` 中追溯。

## 有相同数据库后复现实验

每位组员自行取得团队授权共享的同一快照，设置 `DATA` 指向该目录。标准日面板、元数据及哈希保持原样；不要上传商业原始文件、个股宽表或凭据。

```bash
DATA=../research_database
OUT=../team34_research_fresh
PYTHONPATH=src python -m usq.project2.run --database "$DATA" --output "$OUT" --all
PYTHONPATH=src python -m usq.project2.supplement --database "$DATA" --output "$OUT"
PYTHONPATH=src python -m usq.project2.risk --database "$DATA" --output "$OUT" --supplement
MPLCONFIGDIR=/tmp/team34-mpl PYTHONPATH=src python -m usq.project2.paper \
  --database "$DATA" --output "$OUT" --pdf-output "$OUT/Team34_Report.pdf"
PYTHONPATH=src python -m pytest tests -o addopts='' -q
```

输入目录至少应有 `panels/`（含 `universe.pkl`、`close_adj.pkl`、`ylabel.pkl`、`label_end.csv` 等）、`metadata/fundamental_quarterly.parquet` 和 `checksums.sha256`。所需字段和单位见 [数据字典](data_schema/data_dictionary.csv)。数据库构建与补齐程序保存在 `src/usq/database/`，说明见 `docs/DATABASE_BUILD.md` 和 `docs/DATA_COMPLETION.md`；旧示例配置只下载样本行情，不能靠它重新取得本研究的完整商业数据。

## 目录

| 目录 | 内容 |
| --- | --- |
| `src/usq/project2/` | 本次因子、训练、回测、补充分析和报告入口 |
| `src/usq/barra/` | 原参考核心及美股适配 |
| `src/usq/database/` | 数据库构建、补齐和读取 |
| `reports/` | 审核过的英文 PDF 与中文说明 |
| `results/project2/` | 可公开的汇总结果、图表、协议及测试记录 |
| `data_schema/` | 字段字典和独立交易日历，无个股数据 |
| `tests/` | 公式、时间对齐、账本、数据库与归因测试 |
| `docs/` | 交接、来源记录和继承的底层文档 |

`docs/UPSTREAM_README.md` 及部分旧文档描述底层框架早期状态，不是当前研究结果。当前入口以本 README、`src/usq/project2/` 和已冻结结果为准。

## 研究口径

财务按公布代理日期排序并 ffill，但不是严格历史 PIT；本轮 WQ 候选不依赖 VWAP，统一标签和回测使用 close_adj。Barra-style 是研究模型，不是商业 MSCI 产品；特异贡献不直接等同 alpha。完整执行规则、来源和局限见英文报告。

代码源自 [Quant_U.S._stocks](https://github.com/Panyuxiang0830/Quant_U.S._stocks)，行情抓取模式参考 [global-stock-data](https://github.com/simonlin1212/global-stock-data)。公式及研报来源见报告参考文献和 [NOTICE](NOTICE)。
