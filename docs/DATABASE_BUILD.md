# 本地数据库试验版：2015–2025

主研究期为 2015-01-01 至 2025-12-31，2014 年仅作指标预热，2026 年 1 月行情仅作收益标签尾部。不是只有 2015–2023。

本版完成全候选池基本面和统一宽表结构，日行情先做有限证券样本验证。候选列数不等于已下载行情的股票数，更不等于每天都存在 4,000 只可交易股票。以实际 `metadata/acceptance.json` 为准，不把试验版标记为正式研究验收通过。

## 试验版限制

- ID 使用 `CNA:GVKEY:PRIUSA`，来自 Compustat 公司键与主美国证券标签。不是把公司 GVKEY 冒充证券 ID。原文件的 PRIUSA/ticker 为描述性信息，未重建完整的历史有效区间。
- 排除来源 SIC 6722/6726 及明确基金名称，保留银行和 REIT。历史普通股、ADR、权证、优先股、测试证券等类型仍缺独立历史主表核实。不能宣称这些身份在历史每一天已全部核验。
- 基本面为最新下载版本，无法还原历史所有修订。按有效 RDQ，缺失时按有效 FDATEQ 回退，取日期之后第一个交易日作为近似可用日。RDQ 是盈利公告日，不保证完整财报每个字段同日已公布；FDATEQ 也不等于 SEC 提交日。
- 季度和累计流量先转换为连续四季 TTM，再按可用时间 ffill。绝不反向填充。期末资产、负债、普通股权益和股数不求和。默认财报期末超过 550 天后作过期缺失。
- `market_cap` 按用户确认保存为估算总市值，美元计价：最近已公布的 CSHOQ 股数 × 原始主证券股价。已知拆股发生后，同步调整沿用的财报股数。CSHOQ 对 ADR 已按存托凭证口径处理，不重复乘 ADR 比率。多类别股价与股数口径仍需核实；退市证券缺拆股资料时估值存在额外限制。
- 原始 OHLCV、成交额来自新浪日线；`amount=0` 按来源缺失处理，不当成真实零成交额。只有实际 amount 和同源 volume 都有效时才计算 VWAP，绝不使用 OHLC 均价伪造。
- Yahoo OHLC 本身为拆股调整基准，不能直接假装原始股价。这里仅用其 adjusted_close 与新浪原始 close 的比值生成 `adj_factor`，仅在两者都存在时生成；包括拆股与现金分红调整近似。退市证券缺调整历史时保留缺失，不擅自填 1。
- 只有当日和前一个交易日的调整价都存在时才计算 `daily_return`；不跨缺失日计算，不将缺失收益填 0。
- 行业使用旧年度来源的历史 `SICH`，聚合为 `SIC2:xx` 字符串，按年度来源日期向后延续。不把当前 GICS 倒灌历史。新公司缺历史行业则保持缺失。
- `universe` 是明确 0/1 的试验候选资格掩码：已知六项财务核心字段和来源上市/退市界限。历史证券类型/映射未经独立核实，不能把它称为最终可交易股票池。也不依据未来行情能否下载而删历史股票。
- 不承诺退市终值、并购换股、现金派息或缺失证券的全部历史行情。正式研究前必须补齐并验收。

## 每位组员准备同一版本的数据

1. 下载同一份 `Comp_Quarterly6126.csv.zip`。本次 ZIP SHA256 为 `27ea52eb9c1d7a5e2a96c18af2e4d7fad1bbfc8322a035cba5f2a867aab5225f`。
2. 如需同样的历史行业，另准备原始 North America 年度 `data.dta`，在个人配置填 `annual_source_dta`。本机可复用之前只读提取的 SQLite 临时表，但该临时路径不应写进团队公共配置。
3. 行情接口会修订历史。所有人各自“重新下载”不保证同一数据库。要完全复现，须使用同一份有权限使用的原始行情缓存，并逐项比对 `raw_market_manifest.json`。没有同样的行情快照，不能声称字节级相同。
4. 原始 Compustat 与第三方行情不提交到公开 Git。确认学校授权允许各组员使用；行情缓存仅按来源许可在有权使用者之间流转。

原始 ZIP 和 DTA 只读，构建程序不会修改或覆盖它们。没有凭据写入配置或仓库。

## 安装和运行

本次验证使用 Python 3.13。先创建独立虚拟环境，激活后执行：

```bash
python -m pip install -r requirements-database.txt
python -m pip install --no-deps -e .
```

复制 `configs/database.example.json` 为被 Git 忽略的 `configs/database.local.json`，只改本机输入路径和输出路径。输出目录必须是新快照，已有 `snapshot.json` 时拒绝覆盖。

```bash
python -m usq.database stage --config configs/database.local.json
python -m usq.database fetch --config configs/database.local.json
python -m usq.database build --config configs/database.local.json
python -m usq.database audit --config configs/database.local.json
python -m pytest tests/test_database.py tests/test_data.py
```

`stage` 分块读取 ZIP 内 CSV 的必要字段，筛选范围后缓存 Parquet；检查原始文件哈希后才能复用。冲突财政期、无可用日期记录存放于 staging，不混入下游股票面板。

`fetch` 默认只拉配置列出的代表性样本。要续拉完整候选集，复制一份配置并设置 `market_symbols: null`、`market_fetch_limit: null`。先确认历史 ID 及源授权，保持不超过 4 个并发。HTTP 403/429 会停止批次，不更换 IP、不绕过限制；原始缓存校验哈希且可断点续用。

全量行情取得后，使用同一冻结候选表和原始缓存、但新的 `output` 路径重新构建。不要直接改写已交付的旧快照。当前原始缓存固定在输出根目录的 `raw/market` 下，新的输出目录需放置同一来源缓存或指向合法的复制版本。

## 目录

```text
data/us_2015_2025/
  raw/market/{sina,yahoo}/      原始 gzip 响应及来源元数据
  panels/
    calendar.csv              独立 XNYS 日历，YYYYMMDD
    shares_out.pkl ...        每个字段单独 DataFrame
    industry.pkl              SIC2 分类字符串
    universe.pkl              uint8，明确 0/1
    indices_close.pkl         指数专用列，不与股票列对齐
  metadata/
    frozen_pool.json
    security_master.parquet
    fundamental_vintages.parquet
    industry_history.parquet
    snapshot.json
    acceptance.json
    raw_market_manifest.json
    market_fetch_report.json
    data_dictionary.csv
```

股票宽表所有行列、顺序完全一致：行 YYYYMMDD，列为冻结 SID。数值字段 float32，分类字符串使用 category 保存，universe 为 uint8。代码验证完整轴，不悄悄取交集。pickle 只读取团队可信产物。

`fundamental_vintages.parquet` 的 vintage 表示本次源文件下载版本，不表示已取得真正的历史 PIT archive。它保留每条转换后财务事件的 sid、period_end、available_date、field、value、source、source_vintage 和源行号。

## 读取接口

```python
from usq.database import ResearchDatabase

db = ResearchDatabase('data/us_2015_2025')  # 明知是试验版时允许读取
sid = 'CNA:001690:01'                       # Apple 源主证券 ID

stock = db.stocks(['close', 'volume', 'market_cap'],
                  start='2024-01-01', end='2025-12-31', sids=[sid])
financial = db.fundamentals(['book_equity', 'earnings_ttm', 'revenue_ttm'],
                            start='2024-01-01', end='2025-12-31', sids=[sid])
industry = db.industry(start='2024-01-01', end='2025-12-31', sids=[sid])
indices = db.indices(start='2024-01-01', end='2025-12-31', symbols=['^GSPC', '^IXIC'])
history = db.financial_history(start='2024-01-01', end='2025-12-31', sids=[sid])

# 正式研究必须启用门禁；当前试验版会抛错，不允许无提示通过。
production = ResearchDatabase('data/us_2015_2025', require_accepted=True)
```

未知 SID、错误字段、轴不一致都会报错，不猜 ticker、不静默删列。指数使用独立列集合。接口逐字段读取，不同时加载所有数十个大矩阵。

## 来源与例外依据

- 行情端点设计参考 [global-stock-data](https://github.com/simonlin1212/global-stock-data)，固定参考提交 `b7f22029660769baa29a4403cc61c285f35f8905`。该项目主要提供文档内代码，并非一个可直接安装的全市场数据库；这里另写缓存、转换、质量检查与本地读取层。
- 独立日历：`exchange_calendars==4.13.2` 的 XNYS，参考 [NYSE 日历](https://www.nyse.com/markets/hours-calendars)。代码额外测试 2018 国葬休市、Juneteenth 实施前后和 2025 国葬休市，不从某一股票的报价推导交易日。
- META/FB 换代码边界仅在有明确依据的公司上处理：[Nasdaq 公告](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2022-125)，2022-06-09 生效。FB 之后被 ETF 复用，所以只接旧 FB 至 2022-06-08 的片段。
- 财务变量定义核对用户提供的季度字段 PDF 和 Compustat 厂商变量手册：NIQ 为单季、NIY/CAPXY/OANCFY 为年初累计；CSHOQ 对美股外资公司为 ADR/ADS 数量口径。不能重复作 ADR 换算。

Git 仓库仅保存代码、配置示例、测试和使用说明；原始数据、数据库快照、行情缓存、个人配置及凭据不上传。每位成员在本地准备数据源，并按本文步骤构建数据库。
