"""Offline, data-backed first-experiment report (not the final course PDF)."""
from __future__ import annotations
import argparse
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .screen import write_json


LOGIC = {
    "WQ019": "7日反转，由过去250日收益排名调节强度；偏好长期相对强、近期回调的股票。",
    "WQ040": "高点价格波动排名 × 负的高点价格—成交量相关性；衡量价量关系而非单纯成交量大小。",
    "WQ026": "成交量与高点价格各自的5日时间序列排名，再计算相关并取近期最大值的相反数。",
    "WQ052": "近期低点下移幅度 × 长期收益排名 × 短期成交量排名；长期强势股的活跃回撤信号。",
    "F01_EP_reconstructed": "东方证券，第8、23页：signed-log(TTM净利润) 对 signed-log(市值) 的含截距截面OLS残差。",
    "F02_BP_reconstructed": "东方证券，第8、29页：signed-log(净资产) 对 signed-log(市值) 的截面OLS残差。",
    "F03_SPQ_reconstructed": "东方证券，第8、25页：signed-log(单季收入) 对 signed-log(市值) 的截面OLS残差。",
    "F04_ROEQ_reconstructed": "东方证券，第8、37页：signed-log(单季净利润) 对 signed-log(净资产) 的截面OLS残差。",
    "F05_FCF_yield": "东北证券，第5页：(TTM经营现金流−TTM资本开支)/(市值+总负债−现金)；Compustat迁移版本。",
    "F06_earnings_surprise": "招商证券，第12–13、27页：(单季净利润−简单预期利润)/公告前一交易日市值；公告后ffill。",
}


def run(database, output):
    out, db = Path(output), Path(database)
    if not (out/"model_run_complete.json").exists():
        raise ValueError("Do not report an incomplete model run")
    selected = json.loads((out/"factor_selection.json").read_text())
    screen = pd.read_csv(out/"evaluation/factor_screen.csv")
    summary = pd.read_csv(out/"backtests/summary.csv").set_index("strategy")
    histories = json.loads((out/"training_history.json").read_text())
    assert all(row["train_last_label_end"] < row["forecast_start"] for row in histories)
    names = summary.index.tolist()
    assert len(names) == 6
    daily = {name: pd.read_csv(out/"backtests"/f"{name}_daily.csv", dtype={"date": str}).set_index("date") for name in names}
    reference = daily[names[0]]
    for name in names:
        assert daily[name].index.equals(reference.index)
        np.testing.assert_allclose(daily[name].benchmark_return, reference.benchmark_return)
        np.testing.assert_allclose(daily[name].net_return, daily[name].gross_return-daily[name].cost, atol=1e-12)
    choices = json.loads((out/"frozen_model_choices.json").read_text())
    risk = pd.read_csv(out/"barra_style/mean_style_exposures.csv").set_index("strategy")
    attribution = pd.read_csv(out/"barra_style/attribution_summary.csv")
    assert attribution.max_daily_reconciliation_error.max() < 1e-9
    tests = []
    for half in (f"{y}H{h}" for y in (2023, 2024, 2025) for h in (1, 2)):
        part = [r for r in histories if r["half"] == half]
        assert len(part) == 6
        assert len({r["samples"] for r in part}) == 1
        tests.append({"half": half, "samples": part[0]["samples"], "last_training_signal": part[0]["train_last_signal"], "last_label_end": part[0]["train_last_label_end"]})
    pd.DataFrame(tests).to_csv(out/"training_windows.csv", index=False)
    features = selected["wq"]+selected["fundamental"]
    import joblib
    coefficients = []
    for path in sorted((out/"models").glob("*.joblib")):
        model = joblib.load(path)
        used = features[:len(selected["wq"])] if "_A_" in path.stem else features
        values = model.feature_importances_ if hasattr(model, "feature_importances_") else model.coef_
        for i, feature in enumerate(used):
            coefficients.append({"model_half": path.stem, "factor": feature,
                                 "value_coefficient_or_importance": float(values[2*i]),
                                 "missing_coefficient_or_importance": float(values[2*i+1])})
    pd.DataFrame(coefficients).to_csv(out/"model_feature_effects.csv", index=False)
    versions = {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost", "joblib", "threadpoolctl", "plotly")}
    code = Path(__file__).parent
    verification = {"complete_models": names, "factor_count": len(features), "wq_tested": 36,
                    "full_calendar_sid_axes": [3018, 4366], "oos_days": len(reference),
                    "all_training_labels_mature_before_forecast": True,
                    "A_B_identical_training_sample_counts": True, "common_spy_returns": True,
                    "cost_reconciliation": True, "max_barra_reconciliation_error": float(attribution.max_daily_reconciliation_error.max()),
                    "software": versions,
                    "database_snapshot_checksums_sha256": hashlib.sha256((db/"checksums.sha256").read_bytes()).hexdigest(),
                    "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code.glob("*.py"))}}
    write_json(verification, out/"research_verification.json")
    parts = []
    included_js = False
    def chart(fig, title, height=450):
        nonlocal included_js
        fig.update_layout(template="plotly_white", height=height, title=title, margin=dict(l=55, r=30, t=65, b=40), legend=dict(orientation="h", y=-.15))
        parts.append(fig.to_html(full_html=False, include_plotlyjs=True if not included_js else False, config={"displaylogo": False, "responsive": True}))
        included_js = True
    def heading(title, text=""):
        parts.append(f"<h2>{html.escape(title)}</h2><p>{text}</p>")
    def table(frame, formats=None):
        parts.append('<div class="table-wrap">'+frame.to_html(index=True, float_format=lambda x: f"{x:.4f}", formatters=formats, border=0)+"</div>")
    def pct(x):
        return f"{x:.2%}"
    heading("1. 先看结论", "这是首轮实际实验，不是研究计划。A=4个量价因子；B=同一套量价因子+6个基本面因子。所有模型都使用相同资格样本和相同交易规则；没有根据2023–2025收益再换因子或参数。")
    deltas = []
    for model in ("ridge", "lasso", "xgboost"):
        a, b = summary.loc[f"{model}_A"], summary.loc[f"{model}_B"]
        deltas.append({"模型": model, "加入基本面的年化收益变化(pp)": (b.annual_return-a.annual_return)*100,
                       "Sharpe变化": b.sharpe-a.sharpe, "最大回撤变化(pp，正为改善)": (b.max_drawdown-a.max_drawdown)*100,
                       "日均单边换手变化(pp)": (b.mean_daily_turnover-a.mean_daily_turnover)*100})
    table(pd.DataFrame(deltas).set_index("模型"))
    parts.append("<p>本轮三个模型加入基本面后，日均换手均下降，但净年化收益均未提高；因此当前结果不支持‘加入这六个基本面一定增强量价模型’。这是本股票池、公式迁移、标签期限和交易规则下的结果，不能泛化成‘基本面因子没有价值’。后续正式报告应保留这一不支持原假设的发现，不按测试收益再挑赢家。</p>")
    heading("2. 2023–2025扣费后主结果", "费用为每成交1美元收取10bps；Top100等权、不加杠杆。Sharpe采用252日年化、无风险利率0。相对净值年化=策略净值/SPY净值的年化收益，不等于两个年化收益简单相减。")
    view = summary[["annual_return", "annual_volatility", "sharpe", "max_drawdown", "mean_daily_turnover", "annual_return_minus_spy", "relative_nav_annual_return"]].copy()
    view.columns = ["年化收益", "年化波动", "Sharpe", "最大回撤", "日均单边换手", "年化收益减SPY", "相对净值年化"]
    table(view, {c: pct for c in view.columns if c != "Sharpe"})
    parts.append(f"<p>相同投资区间的 SPY 年化收益：{summary.iloc[0].spy_annual_return:.2%}。单边换手=成交额/(2×调仓前净值)，买和卖均收费。</p>")
    dates = pd.to_datetime(reference.index, format="%Y%m%d")
    fig = go.Figure()
    for name in names:
        fig.add_scatter(x=dates, y=daily[name].net_nav, name=name)
    fig.add_scatter(x=dates, y=reference.benchmark_nav, name="SPY", line=dict(color="black", width=3))
    chart(fig, "扣费后净值（初始1）")
    fig = go.Figure()
    for name in names:
        fig.add_scatter(x=dates, y=daily[name].relative_nav, name=name)
    fig.add_hline(y=1, line_dash="dash")
    chart(fig, "相对SPY净值（不是市场中性组合）")
    fig = go.Figure()
    for name in names:
        fig.add_scatter(x=dates, y=daily[name].drawdown, name=name)
    fig.update_yaxes(tickformat=".0%")
    chart(fig, "策略回撤")
    heading("3. 因子筛选与逻辑", "实现36个不依赖VWAP或美元成交额ADV的WorldQuant原始编号。2015–2018只定方向，2019–2021要求定向RankIC≥0.01、覆盖率≥50%、至少2/3年度为正；平均绝对日截面排名相关性≥0.8的候选去重。5个通过门槛、4个去重后入选，没有为凑十个而降低门槛。HAC t值使用10阶Newey–West型方差修正；未做36候选的多重检验校正，不能据此声称显著发现。")
    selected_names = selected["wq"]+selected["fundamental"]
    fs = screen.set_index("factor").loc[selected_names, ["early_rankic", "direction", "stability_rankic_signed", "stability_hac_t_signed", "stability_coverage"]]
    table(fs)
    for name in selected_names:
        parts.append(f"<p><strong>{name}</strong>：{LOGIC[name]}</p>")
    parts.append("<p>四个回归重构因子金额先固定为百万美元，再做 sign(x)·ln(1+|x|) 变换和含截距OLS。超预期预测值=去年同季利润+此前八季同比变化额均值；要求历史组成项在本次公告前已知。因子方向只由2015–2018决定，基本面因子即使筛选较弱也按预定六个整体进入B，检验增量而非事后只保留赢家。</p>")
    annual = pd.read_csv(out/"evaluation/factor_annual.csv", dtype={"year": str})
    heat = annual.loc[annual.factor.isin(selected_names)].pivot(index="factor", columns="year", values="signed_rankic").reindex(selected_names)
    chart(go.Figure(go.Heatmap(z=heat.to_numpy(), x=heat.columns, y=heat.index, colorscale="RdBu", zmid=0)), "开发期年度定向RankIC")
    quintiles = pd.read_csv(out/"evaluation/factor_forward_quintiles.csv")
    q = quintiles.loc[quintiles.phase.eq("2019-2021")].pivot(index="factor", columns="group", values="mean_forward_10d_return")
    table(q.reindex(selected_names), {c: pct for c in q.columns})
    parts.append("<p>上表仅为配对样本的未来10日收益五分层描述：标签重叠，不计费用，不能把它当成可交易净值。G5为开发期确定方向后的高分组。四个传统比值的对应结果也保存在因子评价表中。</p>")
    heading("4. 时间划分与半年累积训练", "2015–2021选因子；2022只做小网格参数与交易规则验证；2023–2025进行六次半年预测。每次从2015起使用全部累积历史，不是固定长度window。训练仅使用label_end严格早于预测期起点的样本。A/B训练样本数量一致；基本面缺失不会删掉A中已有样本。")
    table(pd.DataFrame(tests).set_index("half"))
    parts.append("<p>输入按当日资格截面转为[-0.5,0.5]排名，缺失排名填0并另加缺失标记。Ridge/Lasso/XGBoost分别在A的2022日RankIC上选参数，再原封不动用于B。常数预测的IC无定义；定义IC的有效验证日须≥80%，不允许仅凭少数日的高IC胜出。没有随机时间拆分，也没有测试期早停。</p>")
    table(pd.read_csv(out/"validation/hyperparameters.csv").set_index(["model", "candidate"]))
    parts.append("<pre>"+html.escape(json.dumps(choices, ensure_ascii=False, indent=2))+"</pre>")
    heading("5. 换手、费用与平滑", "下面先展示2022交易规则验证。全模型统一采用Ridge-A验证净Sharpe最高的规则；2023–2025不再重选。平滑不必然更好：它同时改变换手、信号时效与毛收益。费用敏感性只改变收费，不重新训练或重选股票。")
    rules = pd.read_csv(out/"validation/trading_rules.csv")
    table(rules[["smooth", "rebalance_every", "annual_return", "sharpe", "mean_daily_turnover"]].set_index(["smooth", "rebalance_every"]))
    fig = go.Figure()
    for name in names:
        c = pd.read_csv(out/"backtests"/f"{name}_costs.csv")
        fig.add_scatter(x=c.cost_bps, y=c.annual_return, mode="lines+markers", name=name)
    fig.update_xaxes(title="每成交美元费用(bps)")
    fig.update_yaxes(tickformat=".0%")
    chart(fig, "样本外费用敏感性：净年化收益")
    heading("6. 样本外五分层与年度表现", "这是按相同延迟执行及10bps费用计算的模型评分五分层，不是直接把10日ylabel累加。每组全额投资、等权；分组之间是不同组合，不应把两组净值相减后当作已实现多空策略。")
    fig = make_subplots(rows=3, cols=2, subplot_titles=names)
    for k, name in enumerate(names):
        g = pd.read_csv(out/"backtests"/f"{name}_groups.csv", dtype={"date": str}).set_index("date")
        for col in g:
            fig.add_scatter(x=dates, y=(1+g[col]).cumprod(), name=col, legendgroup=col, showlegend=k == 0, row=k//2+1, col=k%2+1)
    chart(fig, "五分层净值：G5为模型高分组", height=950)
    annual = pd.read_csv(out/"backtests/annual.csv")
    table(annual[["strategy", "year", "total_return", "sharpe", "max_drawdown", "spy_total_return"]].set_index(["strategy", "year"]))
    heading("7. Barra-style风格与收益解释", "复用仓库的联合WLS框架：市场+SIC2行业+9个可用风格；用前一收盘日暴露解释当日收盘收益。此处不是商业MSCI Barra全套产品。Growth未构建，不冒称十风格齐全。SPY仅为绩效基准，不把同池市值组合冒充SPY，也不声称已取得SPY持仓的主动风格暴露。")
    table(risk)
    parts.append("<p>三个B组合都比对应A组合更偏小市值与价值，动量暴露转为负值。结合下方归因，B的风格贡献也更负：这与‘信息增加了，但组合风格改变且净收益未改善’一致。它是描述性解释，不是对风格因果关系的证明，也不能将未解释收益直接称为alpha。</p>")
    chart(go.Figure(go.Heatmap(z=risk.to_numpy(), x=risk.columns, y=risk.index, colorscale="RdBu", zmid=0)), "组合平均绝对风格暴露（标准化单位）")
    table(attribution.set_index("strategy"))
    parts.append("<p>上表使用Carino几何链接，各项之和等于策略累计收益−SPY累计收益；这不是相对净值的累计收益。SPY整体收益单独扣除，风格贡献是策略自身的绝对风格贡献。Specific_or_unexplained包含未解释收益和少量事件/估值处理差异，不能直接叫无风险alpha或因果证明。</p>")
    heading("8. 执行与数据口径（简要）", "财务按公布代理日期排序并ffill，但采用最新下载版本，<strong>不是严格历史PIT</strong>。标签与交易、Barra统一使用close_adj；VWAP缺失不阻断研究，本轮36个候选均不需要VWAP，因此没有把改公式后的结果冒充原版。T收盘后信号最早T+1收盘成交，T+2才获得首段持仓收益。")
    parts.append("<p>课堂执行假设：无法成交的新买单留现金，不能卖出的旧仓保留并按最近可用收盘估值；终止事件优先用已有事件收益结算，缺少结算则按最后估值转现金，逐笔日志另存。研报发表于2024/2026，因此本研究是公式迁移的回溯检验，而不是宣称这些创意在2015年已被预先提出的实盘盲测。</p>")
    heading("9. 可复现入口与产物", "本页为首轮结果页，不替代课程要求的15–25页最终PDF和视频。数据库原文件未修改；研究代码已放入本地仓库，未自动推送。")
    parts.append("<pre>python -m pip install -r requirements-project2.txt\n# Mac另需 brew install libomp\nPYTHONPATH=src python -m usq.project2.run --database ../team17_close10_2014_2025 --output ../project2_research_fresh\nPYTHONPATH=src python -m usq.project2.diagnostics --database ../team17_close10_2014_2025 --output ../project2_research_fresh\nPYTHONPATH=src python -m usq.project2.risk --database ../team17_close10_2014_2025 --output ../project2_research_fresh\nPYTHONPATH=src python -m usq.project2.report --database ../team17_close10_2014_2025 --output ../project2_research_fresh</pre>")
    links = {"原始因子筛选表": "evaluation/factor_screen.csv", "半年训练窗口": "training_windows.csv", "模型样本外IC": "evaluation/model_test_ic.csv", "主结果": "backtests/summary.csv", "年度结果": "backtests/annual.csv", "Barra归因": "barra_style/attribution_summary.csv", "校验与环境": "research_verification.json"}
    parts.append("<p>"+" · ".join(f'<a href="{path}">{label}</a>' for label, path in links.items())+"</p>")
    parts.append('<p>WorldQuant公式来源：<a href="https://arxiv.org/pdf/1601.00991">Kakushadze (2015), 101 Formulaic Alphas</a>。基本面：东方证券《基本面因子的重构》（2024）；东北证券《自由现金流策略的质量与趋势增强》（2026）；招商证券《除了分析师预期，我们还有什么预期？》（2026）。</p>')
    css = "body{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;color:#182536;background:#f4f7fb;max-width:1260px;margin:auto;padding:40px 28px;line-height:1.8}header{background:#17324f;color:white;padding:34px;border-radius:14px;margin-bottom:28px}h1{margin:0;font-size:30px}h2{margin-top:44px;color:#17324f}p{max-width:1100px}table{border-collapse:collapse;white-space:nowrap;background:white;width:100%;font-size:13px}td,th{padding:9px 12px;border-bottom:1px solid #e6eaf0;text-align:right}th{background:#eaf0f7}td:first-child,th:first-child{text-align:left}.table-wrap{overflow-x:auto;border:1px solid #dce3ed;border-radius:8px;margin:20px 0}pre{background:#e9eef5;padding:20px;overflow:auto;line-height:1.6}a{color:#125d97}.plotly-graph-div{background:white;margin:20px 0;border-radius:10px}@media print{body{background:white;padding:0}.table-wrap{overflow:visible}h2{break-after:avoid}.plotly-graph-div{break-inside:avoid}}"
    document = '<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Project2 首轮研究结果</title><style>'+css+'</style></head><body><header><h1>基本面是否增强量价选股？</h1><p>Project 2 · 首轮实证结果 · 2015起累积训练 / 2022验证 / 2023–2025回测</p></header>'+"\n".join(parts)+"</body></html>"
    path = out/"研究结果.html"
    path.write_text(document, encoding="utf-8")
    print(json.dumps({"report": str(path), "strategies": names, "verification": verification}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.database, a.output)
