"""Team 34 English research report; consume frozen results, never refit models."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from pypdf import PdfReader

from .screen import write_json


SHORT = {"F01_EP_reconstructed": "F1: EP residual", "F02_BP_reconstructed": "F2: BP residual",
         "F03_SPQ_reconstructed": "F3: quarterly SP residual", "F04_ROEQ_reconstructed": "F4: quarterly ROE residual",
         "F05_FCF_yield": "F5: FCF yield", "F06_earnings_surprise": "F6: earnings surprise",
         "equal_weight_scores_A": "Equal-score A", "equal_weight_scores_B": "Equal-score B",
         "equal_weight_scores_C": "Fundamental-only equal-score C", "equal_weight_eligible_pool": "Eligible-pool equal weight"}
COLORS = ["#185785", "#00877e", "#c26f24", "#8b55a0", "#6d7882", "#d75769"]
NAVY, TEAL = colors.HexColor("#17334d"), colors.HexColor("#00877e")


def label(s):
    return SHORT.get(s, s.replace("xgboost", "XGBoost").replace("ridge", "Ridge").replace("lasso", "Lasso").replace("_", " "))


def percent(x):
    return f"{x:.2%}"


def run(database, output, pdf_output):
    out, db, pdf = Path(output), Path(database) if database else None, Path(pdf_output)
    figures = out/"paper_figures"
    figures.mkdir(exist_ok=True)
    pdf.parent.mkdir(exist_ok=True, parents=True)
    read = lambda p: pd.read_csv(out/p)
    selection = json.loads((out/"factor_selection.json").read_text())
    chosen = json.loads((out/"frozen_model_choices.json").read_text())
    main = read("backtests/summary.csv").set_index("strategy")
    names = main.index.tolist()
    daily = {n: pd.read_csv(out/"backtests"/f"{n}_daily.csv", dtype={"date": str}).set_index("date") for n in names}
    dates = pd.to_datetime(daily[names[0]].index, format="%Y%m%d")
    screen = read("evaluation/factor_screen.csv").set_index("factor")
    horizons = read("supplement/factor_horizons.csv")
    controls = read("supplement/reconstruction_vs_ratio.csv")
    uncertainty = read("supplement/paired_uncertainty.csv")
    supplemental = read("supplement/portfolio_summary.csv").set_index("strategy")
    risks = read("barra_style/mean_style_exposures.csv").set_index("strategy")
    attr = read("barra_style/attribution_summary.csv").set_index("strategy")
    windows = read("training_windows.csv")
    half = read("supplement/halfyear_performance.csv")
    model_ic = read("evaluation/model_test_ic.csv").set_index("strategy")
    rules = read("validation/trading_rules.csv")
    hyper = read("validation/hyperparameters.csv")
    smooth = read("supplement/smoothing_descriptive.csv")
    fundamentals = selection["fundamental"]
    selected = selection["wq"]+fundamentals
    if db is not None:
        universe = pd.read_pickle(db/"panels/universe.pkl")
        counts = universe.loc["20150101":"20251231"].sum(axis=1)
        panel_shape = {"calendar_sessions": len(universe), "security_columns": universe.shape[1]}
    else:
        panel_shape = json.loads((out/"report_inputs/panel_shape.json").read_text())
        counts = pd.read_csv(out/"report_inputs/universe_counts.csv", dtype={"date": str}).set_index("date").eligible_count
    assert len(main) == 6 and len(windows) == 6
    for name in names:
        np.testing.assert_allclose(daily[name].benchmark_return, daily[names[0]].benchmark_return)
        np.testing.assert_allclose(daily[name].net_return, daily[name].gross_return-daily[name].cost, atol=1e-11)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.labelcolor": "#263747",
                         "axes.prop_cycle": plt.cycler(color=COLORS), "savefig.facecolor": "white"})
    def save(fig, name):
        fig.tight_layout(pad=1.3)
        fig.savefig(figures/f"{name}.png", dpi=190, bbox_inches="tight")
        plt.close(fig)
    def timeaxis(ax):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="y", alpha=.18)
    def lines(ax, column, benchmark=False):
        for i, n in enumerate(names):
            ax.plot(dates, daily[n][column], color=COLORS[i//2], linestyle="--" if n.endswith("B") else "-", lw=1.3, label=label(n))
        if benchmark:
            ax.plot(dates, daily[names[0]].benchmark_nav, color="#222222", lw=2, label="SPY")
        timeaxis(ax)
        ax.legend(ncol=3, fontsize=8.7, loc="upper left")
    fig, ax = plt.subplots(figsize=(8.2, 3.1))
    ax.plot(pd.to_datetime(counts.index), counts, color=COLORS[0], lw=1)
    ax.set(ylabel="Daily eligible securities", title="A dynamic pool, not 4,366 stocks every day")
    timeaxis(ax); save(fig, "sample")
    wq = screen.loc[screen.family.eq("WQ")].sort_values("stability_rankic_signed")
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.barh(wq.index, wq.stability_rankic_signed, color=[COLORS[1] if n in selection["wq"] else "#b9c7d2" for n in wq.index])
    ax.axvline(.01, color=COLORS[2], linestyle="--", lw=1)
    ax.tick_params(axis="y", labelsize=9)
    ax.set(xlabel="Directional mean daily RankIC, 2019-2021", title="36 evaluated candidates; four retained after redundancy filtering")
    ax.grid(axis="x", alpha=.2); save(fig, "screen")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    for ax, group, title in zip(axes, [selection["wq"], fundamentals], ["Price-volume factors", "Fundamental factors"]):
        for name in group:
            z = horizons.loc[horizons.factor.eq(name) & horizons.phase.eq("2019-2021")].sort_values("horizon")
            ax.plot(z.horizon, z.rankic, marker="o", markersize=3, label=label(name).replace(" residual", ""))
        ax.axhline(0, color="grey", lw=.7)
        ax.set(title=title, xlabel="Forward sessions", xticks=[1, 5, 10, 20])
        ax.grid(alpha=.16); ax.legend(fontsize=7.3, loc="best")
    axes[0].set_ylabel("Directional mean daily RankIC"); save(fig, "horizons")
    fig, ax = plt.subplots(figsize=(8.2, 3.8)); lines(ax, "net_nav", True)
    ax.set(ylabel="Wealth, initial 1", title="Main experiment: net performance at 10 bps per traded dollar")
    save(fig, "nav")
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True)
    lines(axes[0], "relative_nav"); axes[0].axhline(1, color="grey", lw=.8)
    axes[0].set(ylabel="Strategy / SPY", title="Relative wealth and strategy drawdowns")
    lines(axes[1], "drawdown"); axes[1].yaxis.set_major_formatter(PercentFormatter(1)); axes[1].set_ylabel("Drawdown")
    axes[1].get_legend().remove(); save(fig, "relative_drawdown")
    half["strategy"] = half.model+"_"+half.family
    hp = half.pivot(index="strategy", columns="half", values="total_return").reindex(names)
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    zmax = max(abs(hp.to_numpy()).max(), .01)
    im = ax.imshow(hp.to_numpy()*100, cmap="RdBu", vmin=-zmax*100, vmax=zmax*100, aspect="auto")
    ax.set_xticks(range(6), hp.columns); ax.set_yticks(range(6), [label(n) for n in names])
    for (i, j), value in np.ndenumerate(hp.to_numpy()):
        ax.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=9, color="white" if abs(value) > zmax*.6 else "#142536")
    fig.colorbar(im, ax=ax, label="Half-year net return (%)", shrink=.8)
    ax.set_title("Calendar half-year outcomes, without restarting the portfolio"); save(fig, "halves")
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for i, n in enumerate(names):
        c = pd.read_csv(out/"backtests"/f"{n}_costs.csv")
        ax.plot(c.cost_bps, c.annual_return, marker="o", ms=3, color=COLORS[i//2], linestyle="--" if n.endswith("B") else "-", label=label(n))
    ax.axvline(10, color="grey", lw=.7); ax.axhline(0, color="grey", lw=.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set(xlabel="Cost per traded dollar (bps)", ylabel="Net annualized return", title="No retraining or rule selection across cost assumptions")
    ax.legend(fontsize=8.6, ncol=3); ax.grid(alpha=.16); save(fig, "cost")
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.0), sharex=True)
    for family, offset, color in (("A", -.17, COLORS[0]), ("B", .17, COLORS[1])):
        s = smooth.loc[smooth.family.eq(family)].sort_values(["smooth", "rebalance_every"])
        axes[0].bar(np.arange(8)+offset, s.annual_return, width=.32, color=color, label=f"Ridge {family}")
        axes[1].bar(np.arange(8)+offset, s.mean_daily_turnover, width=.32, color=color)
    axes[0].axhline(0, color="grey", lw=.7); axes[0].legend(ncol=2)
    axes[0].set(ylabel="Net annualized return", title="Post-main-run smoothing diagnostics: no test-period re-selection")
    axes[1].set(ylabel="Mean daily one-way turnover", xticks=range(8), xticklabels=[f"SMA {s}\nRebal {r}" for s in (1, 3, 5, 10) for r in (1, 5)])
    for ax in axes: ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis="y", alpha=.16)
    save(fig, "smooth")
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 6.2), sharex=True)
    for ax, n in zip(axes.flat, names):
        g = pd.read_csv(out/"backtests"/f"{n}_groups.csv", index_col=0)
        for k, col in enumerate(g):
            ax.plot(dates, (1+g[col]).cumprod(), lw=1, label=col, color=COLORS[k])
        ax.set_title(label(n), fontsize=10); timeaxis(ax)
    axes[0, 0].legend(ncol=5, fontsize=9, loc="upper left")
    save(fig, "groups")
    def heat(frame, name, title, height=3.5):
        fig, ax = plt.subplots(figsize=(8.2, height))
        limit = max(float(np.nanmax(np.abs(frame))), .01)
        im = ax.imshow(frame, aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(frame.columns)), frame.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(frame)), [label(n) for n in frame.index], fontsize=8.7)
        fig.colorbar(im, ax=ax, shrink=.8, label="Standardized exposure")
        ax.set_title(title, fontsize=11); save(fig, name)
    heat(risks, "styles", "Portfolio styles: absolute exposures, not SPY look-through active exposures", 3.9)
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    means = uncertainty.cagr_difference.to_numpy()*100
    low = uncertainty.cagr_diff_ci_low.to_numpy()*100; high = uncertainty.cagr_diff_ci_high.to_numpy()*100
    ax.errorbar(means, range(3), xerr=np.array([means-low, high-means]), fmt="o", color=COLORS[0], capsize=4)
    ax.axvline(0, color="grey", linestyle="--"); ax.set_yticks(range(3), ["Ridge", "Lasso", "XGBoost"])
    ax.set(xlabel="Fusion B minus price-volume A CAGR (percentage points)", title="Paired block-bootstrap 95% intervals all include zero")
    ax.grid(axis="x", alpha=.2); save(fig, "uncertainty")
    base_names = ["equal_weight_scores_A", "equal_weight_scores_B", "equal_weight_scores_C", "equal_weight_eligible_pool"]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for n in base_names:
        z = pd.read_csv(out/"supplement/portfolios"/f"{n}_daily.csv")
        ax.plot(dates, z.net_nav, label=label(n), lw=1.4)
    ax.plot(dates, daily[names[0]].benchmark_nav, color="black", label="SPY", lw=1.6)
    ax.legend(fontsize=8.4); timeaxis(ax); ax.set(ylabel="Net wealth, initial 1", title="Supplementary simple baselines, not replacement main strategies")
    save(fig, "baselines")
    month = read("supplement/monthly_target_style_exposures.csv")
    month = month.loc[month.phase.eq("2015-2021")]
    mf = month.pivot_table(index="factor", columns="style", values="target_exposure", aggfunc="mean").reindex(selected)
    heat(mf, "single_styles", "2015-2021 month-end Top100 target exposures (development diagnostics)", 4.0)
    print(json.dumps({"stage": "paper_figures_complete", "figures": len(list(figures.glob('*.png')))}), flush=True)

    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10.2, leading=14.8, textColor=NAVY, spaceAfter=9)
    small = ParagraphStyle("Small", parent=body, fontSize=8.6, leading=11.6, textColor=colors.HexColor("#4c5b69"), spaceAfter=7)
    title_style = ParagraphStyle("Title", parent=body, fontName="Helvetica-Bold", fontSize=24, leading=29, spaceAfter=16)
    heading = ParagraphStyle("Heading", parent=body, fontName="Helvetica-Bold", fontSize=17, leading=22, spaceAfter=13)
    sub = ParagraphStyle("Sub", parent=body, fontName="Helvetica-Bold", fontSize=11.3, leading=15, textColor=TEAL, spaceAfter=8)
    code_style = ParagraphStyle("Code", parent=small, fontName="Courier", fontSize=8.2, leading=11, backColor=colors.HexColor("#eef3f7"), borderPadding=8, spaceAfter=13)
    table_style = ParagraphStyle("Cell", parent=body, fontSize=8.5, leading=11, spaceAfter=0)
    header_style = ParagraphStyle("HeaderCell", parent=table_style, fontName="Helvetica-Bold", textColor=colors.white)
    story, sections = [], []
    width = A4[0]-90
    def p(text, style=body):
        story.append(Paragraph(text, style))
    def plain(text, style=body):
        p(escape(str(text)).replace("\n", "<br/>"), style)
    def page(section, title):
        if sections: story.append(PageBreak())
        sections.append(title)
        plain(section.upper(), sub); plain(title, title_style if len(sections) == 1 else heading)
    def table(headers, rows, weights=None):
        weights = weights or [1]*len(headers)
        widths = [width*w/sum(weights) for w in weights]
        cells = [[Paragraph(escape(str(x)), header_style) for x in headers]]
        cells += [[Paragraph(escape(str(x)), table_style) for x in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                               ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#edf3f7"), colors.white]),
                               ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#c9d7df"))]))
        story.extend([t, Spacer(1, 11)])
    def image(name, height, caption):
        im = Image(str(figures/f"{name}.png"))
        ratio = min(width/im.imageWidth, height/im.imageHeight)
        im.drawWidth, im.drawHeight = im.imageWidth*ratio, im.imageHeight*ratio
        im.hAlign = "CENTER"; story.append(im); plain(caption, small)
    def source(path):
        plain("Reproducible source: "+path, small)

    page("DSA5205 / Project 2 / Team 34", "Do Fundamental Signals Improve Price-Volume Stock Selection?")
    p("A reproducible U.S. equity study with expanding-history learning, transaction costs, and Barra-style interpretation", sub)
    p("<b>Team 34</b><br/>English research report - 9 November 2026<br/>Working draft: member names, student IDs and individual contributions have not yet been supplied.", small)
    plain("Abstract", sub)
    p("We test whether six financial-statement signals improve stock selection beyond four price-volume factors retained from 36 implemented WorldQuant candidates. Factor development uses 2015-2021, model and trading-rule validation uses 2022, and the main evaluation covers 2023-2025. Ridge, Lasso and XGBoost are refitted every half-year using all eligible history from 2015, with incomplete forward-return labels purged at each boundary. Within each model family, the price-volume and fusion experiments share sample eligibility, hyperparameters and execution assumptions.")
    p("At a cost of 10 basis points per traded dollar, the price-volume models earn annualized net returns of 9.02%-10.56%, versus 8.82%-10.10% after adding fundamentals. The same-period SPY benchmark earns 22.86%. Fundamentals reduce turnover, but do not improve realized net compound returns; paired time-block confidence intervals do not establish a reliable improvement. Barra-style diagnostics associate fusion with smaller-company, value and negative-momentum tilts.")
    p("Post-main-run diagnostics offer an important qualification: a simple equal-score fundamental portfolio earns 19.88% annually. The evidence therefore concerns this learning-and-portfolio pipeline, not a universal failure of fundamental information. Financial inputs are release-aligned latest-download statements, not a strict historical PIT archive. This is a retrospective classroom study, not a live investment recommendation.")
    plain("Keywords: factor replication; RankIC; expanding training; transaction costs; style attribution", small)

    page("01 / Motivation", "Economic hypotheses and experimental logic")
    p("Price-volume signals can capture short-term reversal, changing trading activity and the interaction between past performance and recent selling pressure. Financial-statement information instead describes profitability, valuation, cash generation and earnings expectations. These signals update on different clocks. Combining them could improve ranking, reduce unnecessary trading or diversify economic exposures, but it can also add redundant information and estimation noise.")
    table(["Question", "Operational test"], [["Does fundamental information add predictive value?", "Compare A (four WQ signals) with B (A plus six financial signals) within each model family."], ["Does prediction translate into implementable performance?", "Use delayed-close Top100 portfolios, drifted weights and costs; do not trade the forward label directly."], ["Where does performance come from?", "Compare SPY, sample-period stability, turnover and joint market/industry/style attribution."], ["Does complexity beat a transparent rule?", "Report supplementary equal-score and eligible-pool equal-weight portfolios without promoting a test-period winner."]], [1, 1.75])
    p("The primary hypothesis is directional: adding fundamentals improves net selection performance under a fixed investment process. We also examine mechanisms: whether turnover falls, whether top-ranked portfolios retain a monotonic relationship with scores, and whether changes in return correspond to recognizable style tilts.")
    p("The three learning families represent different assumptions. Ridge spreads weight across correlated weak signals; Lasso favors a sparse explanation; shallow boosted trees permit limited nonlinearities and interactions. None is assumed to succeed. A negative finding remains informative when labels, validation, benchmarks and costs are consistently specified.")
    plain("Analysis hierarchy", sub)
    p("The six learned A/B strategies are the main experiment. Simple baselines, signal-horizon comparisons, monthly factor exposures and uncertainty intervals are diagnostic extensions. The latter were completed after main outcomes became visible and are not a new untouched holdout. They are not used to change the retained factors, selected hyperparameters or main trading rule.")

    page("02 / Data and sample", "Assets, fields and the research universe")
    p("We study a dynamic pool of U.S.-listed company securities in USD. The dataset combines team-supplied daily price/volume panels and CRSP-linked security descriptors with Compustat North America quarterly financial statements. SPY adjusted-close data were cached from Yahoo through the global-stock-data ingestion pattern. The independent calendar is the XNYS session calendar. ETFs and investment funds are not stock-selection targets; SPY is an external investable benchmark.")
    table(["Item", "Observed scope"], [["Panel calendar", f"{panel_shape['calendar_sessions']:,} sessions, 2 January 2014 to 31 December 2025; 2014 is warm-up."], ["Stable-security column union", f"{panel_shape['security_columns']:,} IDs; daily eligibility changes."], ["Daily core universe, 2015-2025", f"{int(counts.min()):,}-{int(counts.max()):,}; median {int(counts.median()):,}."], ["Main investment period", "752 sessions in 2023-2025; additional factor history starts in 2015."], ["Financial units", "USD; monetary inputs divided by one million before signed-log regressions."], ["Market capitalization", "The database's USD market_cap field; its source construction is documented in the data dictionary."]], [1.1, 2.3])
    image("sample", 205, "Figure 1. Historical column count and daily investable count are different quantities.")
    p("All stock panels retain identical full calendar and security axes. The common model eligibility additionally requires at least one observed retained WQ signal; missing fundamental features do not remove observations from B that are present in A. Price inputs, classifications and source hashes remain outside the public code repository. Commercial source files should only be accessed under the team's applicable institutional permissions.", small)

    page("03 / Information timing", "The target is not the traded return")
    p("At signal date t, the supervised label is the raw ten-session adjusted-close return. It is neither cross-sectionally neutralized nor standardized. An interval is labeled only when its endpoint prices and adjustment-segment continuity satisfy the dataset's rules. Daily features use current and earlier observations; forward labels are evaluation targets, never predictors.")
    plain("y[i,t] = close_adj[i,t+10] / close_adj[i,t] - 1", code_style)
    table(["Time", "What happens", "Information restriction"], [["t, after close", "Observe features and generate a score.", "No information from t+1 or later."], ["t+1, close", "Execute the previously determined target; pay costs.", "Do not earn the return that ended before the fill."], ["t+2, close", "Record the first held close-to-close return.", "Weights are beginning-of-interval holdings."], ["Training cut-off", "Use a sample only if label_end precedes the forecast half-year.", "No partially realized ten-day targets."]], [.65, 1.4, 1.6])
    p("The prediction horizon and execution delay are distinct. The label starts at t, but a signal observed after that close cannot be filled at the same price. Consequently, a positive RankIC does not guarantee that the predictable component survives the t+1 close delay. All portfolio results are computed from the holdings ledger, not by summing labels.")
    plain("Financial availability", sub)
    p("Quarterly records are sorted by the usable publication proxy date and carried forward, never backfilled. A statement is generally usable from the next trading session after the release date; TTM quantities wait for the required constituents. The financial archive is a latest-download snapshot, not strict historical PIT: later restatements may be reflected in earlier reported values. This limitation is separate from the training-label purge and is not eliminated by forward filling.")
    p("Labels near each development boundary are also purged: 2015-2018 direction fitting does not consume 2019 outcomes, and the 2019-2021 screen does not consume 2022 outcomes. No 2026 returns enter the 2025 investment results.")

    page("04 / Fundamental replication", "Six signals from three source reports")
    p("Four factors adapt the residual-reconstruction framework of Orient Securities [R2]. Monetary values are fixed in USD millions before the signed-log transform. Each trading-day cross-section fits an OLS regression with an intercept over valid eligible observations and keeps its residual.")
    plain("g(x) = sign(x) * log(1 + abs(x) / 1,000,000)\nresidual[i,t] = g(A[i,t]) - a[t] - b[t] * g(B[i,t])", code_style)
    table(["ID / source", "Definition and economic intent"], [["F1 / [R2], p.23", "TTM net income on market cap: relative earnings strength beyond scale."], ["F2 / [R2], p.29", "Book equity on market cap: relative book valuation."], ["F3 / [R2], p.25", "Quarterly revenue on market cap: sales valuation."], ["F4 / [R2], p.37", "Quarterly net income on book equity: relative quarterly profitability."], ["F5 / [R3], p.5", "(Operating cash flow TTM - capex TTM) / (market cap + total liabilities - cash)."], ["F6 / [R4], pp.12-13,27", "(Actual quarterly earnings - expected earnings) / market cap on the session before release."]], [1, 2.8])
    p("For F5, enterprise value follows the report's convention rather than an assumed universal vendor definition. Only positive denominators are retained. SIC2 60-69 observations are inapplicable at this factor level; banks and REITs are not removed from the entire model pool. This is a U.S. Compustat adaptation of the signal, not a full reproduction of the original strategy's filters or weighting.")
    p("For F6, expected earnings equal last year's same-quarter earnings plus the average of the preceding eight quarterly year-over-year changes in earnings dollars. The necessary historical constituents must have been usable before the current release. Actual earnings enter only after release; the resulting surprise is then carried forward. The denominator is strictly before release, not simply before its next-session usable date.")
    p("Traditional EP, BP, quarterly SP and quarterly ROE ratios are retained as controls. All six fundamental signals remain in B even when individual development metrics are weak; the study does not retrospectively retain only their winners.", small)

    page("05 / Price-volume replication", "The four retained WorldQuant formulas")
    p("The implemented candidate set contains 36 numbered formulas from Kakushadze [R1], not the entire 101-factor collection. Candidates were chosen to avoid unavailable VWAP and full-market dollar-volume ADV inputs. The retained signals need prices, share volume and historical close returns only. Original formula numbers are preserved.")
    table(["Factor", "Operations / windows", "Economic interpretation"], [["WQ019", "Negative sign of 7-session price change, scaled by the cross-sectional rank of a 250-session return sum.", "Recent reversal, amplified by longer-run relative performance."], ["WQ040", "Negative 10-session high-price volatility rank times 10-session high-price/volume correlation.", "Price-volume co-movement conditioned on price variability."], ["WQ026", "Negative 3-session maximum of the 5-session correlation between 5-session volume and high-price time ranks.", "Avoid strong contemporaneous price-volume rank co-movement."], ["WQ052", "Difference between lagged and current 5-session low minima, times long-horizon return rank and 5-session volume rank.", "A recent low-price setback in relatively strong, actively traded shares."]], [.65, 1.65, 1.5])
    p("Operators use only trailing observations and complete rolling windows. Cross-sectional ranks use the contemporaneous eligible set and average ties. Time ranks are percentile ranks; standard deviations use sample scaling. Unknown adjustment-segment boundaries are not bridged. Factor OHLC inputs are forward-anchored within known segments so that future back-adjustment constants cancel; investment returns continue to use close_adj ratios.")
    p("The common label and backtest chain use close_adj instead of VWAP. Crucially, this is not described as replacing VWAP inside these four formulas: none requires it. A formula that did replace its VWAP input would need a separate variant name. ADV in the source paper denotes dollar volume, not share volume; no retained formula confuses those inputs.")
    p("The interpretations above are hypotheses for organizing evidence, not proof of behavioral causality. A favorable historical RankIC alone does not establish a durable or readily tradable anomaly.")
    source("src/usq/project2/factors.py; factor_catalog.json")

    page("06 / Factor selection", "Screening without filling a quota")
    p("Direction is fixed by the sign of mean daily RankIC in 2015-2018. The 2019-2021 stability screen requires directional mean RankIC at least 0.01, at least 50% eligible paired coverage, and positive annual RankIC in at least two of three years. Candidates are ordered by stability RankIC and greedily de-duplicated when their average absolute daily rank correlation reaches 0.8. Five pass the basic screen; WQ006 is redundant, leaving four rather than forcing ten.")
    image("screen", 364, "Figure 2. Green candidates survive screening and redundancy control; the dashed line is the 0.01 stability threshold.")
    table(["Retained factor", "2019-2021 RankIC", "Coverage", "HAC t, lag 10"], [[n, f"{screen.loc[n,'stability_rankic_signed']:.4f}", percent(screen.loc[n,'stability_coverage']), f"{screen.loc[n,'stability_hac_t_signed']:.2f}"] for n in selection["wq"]], [1.05, 1, .8, 1])
    p("Daily RankIC observations overlap because the target spans ten sessions. HAC statistics are descriptive dependence-adjusted diagnostics, not discovery proof. No multiple-testing correction across all 36 candidates is used, and post-selection inference is not claimed.", small)

    page("07 / Factor evidence", "Reconstruction is not uniformly better than a ratio")
    p("The four reconstruction-versus-ratio comparisons below use common valid stock-day pairs and each factor's development-fixed direction. This prevents apparent improvement caused only by the reconstructed version admitting a different set of negative- or missing-book-equity observations. The table uses 2019-2021; 2022 is separately recorded as validation-period description.")
    c = controls.loc[controls.phase.eq("2019-2021")]
    table(["Pair", "Residual RankIC", "Ratio RankIC", "Difference"], [[label(r.reconstructed).split(": ")[0], f"{r.reconstructed_rankic:.4f}", f"{r.ratio_rankic:.4f}", f"{r.difference:+.4f}"] for r in c.itertuples()], [.7, 1, 1, .85])
    image("horizons", 251, "Figure 3. Horizon diagnostics use only 2019-2021 and labels matured inside that phase. They do not change the fixed ten-session main target.")
    p("All four residual versions trail their ratio controls on mean paired RankIC in the stability phase. Quarterly ROE remains among the stronger financial signals, but this does not establish a general benefit from regression reconstruction. In 2022, its paired residual RankIC rises to 0.0635 versus 0.0593 for the ratio, illustrating time variation rather than a universal ranking.")
    p("F4 and F5 have stability RankIC around 0.0230 and 0.0203 on their own valid coverage; earnings surprise is close to zero. The fundamental factors are retained together as the pre-specified information block. The horizon curves are diagnostic: their strongest point is not used to retune the target after seeing investment-period outcomes.")
    source("supplement/reconstruction_vs_ratio.csv; supplement/factor_horizons.csv")

    page("08 / Learning framework", "Shared preprocessing, different inductive assumptions")
    p("Each known feature is transformed to its current eligible cross-sectional percentile minus 0.5. A missing transformed value is filled with zero and accompanied by a missingness indicator. This does not fit a scale on future dates: it is a contemporaneous cross-sectional operation. Raw ten-session targets are not winsorized, demeaned or standardized.")
    table(["Family", "Estimation and rationale", "Overfitting control"], [["Ridge", "Closed-form / Cholesky fit. Correlated financial and technical signals may carry small, distributed effects.", "L2 shrinkage; alpha equals selected alpha_per_sample times the training observation count."], ["Lasso", "Coordinate descent. A sparse combination can discard weak or redundant signals.", "L1 penalty, max_iter=500, tol=1e-5; no arbitrary use of constant scores as stock rankings."], ["XGBoost", "CPU histogram boosting. Shallow trees can model thresholds and modest interactions.", "120 trees, learning rate 0.05, subsample 0.8, reg_lambda=10, min_child_weight=100."]], [.75, 1.7, 1.7])
    p("A has four signal values plus four missing indicators. B has ten values plus ten indicators. A and B are trained on exactly the same eligible observations at each half-year boundary. They share random seed 5205, family-specific chosen hyperparameters, trade delay, Top100 allocation, cost rate and rebalancing schedule. B is not separately retuned to obtain more favorable outcomes.")
    p("The goal is not to establish that more features must help. The relevant question is whether they help after controlling the learning and investment process. An extra input can alter rank tails or missingness/industry exposures without improving the selected portfolio, even when its average prediction correlation is positive.")
    p("Reported feature-effect files separate value coefficients/importances from missingness effects. Linear coefficients and tree importances are different objects; their magnitudes are not compared as if they shared units. No claim of causal explanation is based on these weights.")
    source("src/usq/project2/models.py; model_feature_effects.csv")

    page("09 / Temporal design", "Expanding history from 2015, not a fixed rolling window")
    table(["Phase", "Permitted decisions"], [["2015-2018", "Initial factor direction; this early factor description is in-sample."], ["2019-2021", "Stability screening, coverage and redundancy; no 2022 outcomes used."], ["2022", "Small family hyperparameter grid and one common trading rule."], ["2023-2025", "Six forecast halves; mechanical re-estimation using accumulated realized history."]], [1, 2.9])
    p("For every forecast half-year, the training start stays in 2015. We do not discard older observations using a 3-year or 5-year sliding window. The last signal date is earlier than the nominal cut-off because its full ten-session target must already be observable. Training is from scratch, not continuation of a previous tree ensemble.")
    table(["Forecast half", "Paired A/B samples", "Last training signal", "Last label end"], [[r.half, f"{r.samples:,}", str(r.last_training_signal), str(r.last_label_end)] for r in windows.itertuples()], [1, 1.2, 1.15, 1.1])
    p("The 2023H1 model uses no target ending after 2022. The 2023H2 model may incorporate realized 2023H1 targets through the stated purge boundary. This is legitimate recursive learning under a fixed protocol, not discretionary reuse of the test set to choose new models. All prior training history remains in the fit.")
    p("The final ten sessions of the source panel have no ten-day forward target. They may still receive live-style model scores and appear in the investment ledger, but they are absent from target-based prediction evaluation. This distinction avoids inventing end-of-sample labels.")
    source("training_windows.csv; training_history.json")

    page("10 / Validation", "A small search, with a clear selection rule")
    p("Each family has two parameter candidates. Each candidate is estimated for both halves of 2022 using the same expanding-history and purge protocol: 12 validation fits in total. The metric is pooled mean daily RankIC on A, with labels matured by 2022 year-end. Parameters are then shared with B. This grid is deliberately small and is not revisited in 2023-2025.")
    table(["Family / candidate", "Setting", "2022 RankIC", "Defined-day share"], [[r.model.title()+f" / {r.candidate}", r.params, f"{r.rankic:.4f}", percent(r.defined_rankic_day_fraction)] for r in hyper.itertuples()], [1.05, 1.5, .8, .9])
    p("The stronger Lasso penalty produces only two score levels in one validation half and constant predictions in the other. A correlation is undefined for a constant cross-section; such a candidate cannot win merely by being evaluated on a small favorable subset. The defined-RankIC-day threshold is 80%. Constant predictions imply cash, not the first 100 security IDs.")
    p("The retained parameters are Ridge alpha_per_sample=0.01, Lasso alpha=0.0001 and XGBoost max_depth=2. All other boosted-tree controls are fixed. Trading-rule validation tests score-percentile SMA windows 1/3/5/10 and rebalancing every 1/5 sessions using Ridge A, choosing the highest net Sharpe and favoring lower turnover on ties.")
    table(["SMA window", "Rebalance sessions", "2022 net Sharpe", "Daily one-way turnover"], [[int(r.smooth), int(r.rebalance_every), f"{r.sharpe:.3f}", percent(r.mean_daily_turnover)] for r in rules.sort_values("sharpe", ascending=False).head(4).itertuples()], [.85, 1, 1, 1.2])
    p("The winning common rule is no additional smoothing and rebalancing every five sessions. Its validation Sharpe is still negative. Selecting the least-poor candidate is not evidence of a profitable validation strategy; we retain this outcome rather than widening the search.", small)

    page("11 / Portfolio implementation", "A self-financing delayed-close ledger")
    p("At a scheduled signal date, rank eligible scores and assign equal target weights to the 100 highest values, or all available names if fewer qualify. Portfolios are long-only and unlevered. Between rebalances, weights drift with marked asset values; they are not silently reset to equal weight every day. Ineligible or unfillable new purchases remain cash instead of being replaced by tomorrow's next-best name.")
    plain("gross_return[t] = sum_i beginning_weight[i,t] * marked_return[i,t]\nfee[t] = 0.001 * actual_traded_dollars[t] / beginning_NAV[t]\nnet_return[t] = gross_return[t] - fee[t]", code_style)
    p("Targets are fractions of post-cost wealth. A small fixed-point calculation solves the fees and final target dollars consistently, so the ledger does not overinvest by treating a pre-fee target as post-fee capital. Purchases and sales both incur the cost. Initial entry and feasible final liquidation are included.")
    table(["Convention", "Treatment"], [["Turnover", "One-way turnover = actual traded notional / (2 x pre-trade NAV)."], ["10 bps cost", "Assumed combined commission/spread/slippage proxy per dollar bought or sold, not a broker quote."], ["Unavailable sale", "Retain the old position and its mark until executable; do not use future availability to reselect."], ["Terminal / stale marks", "Use available terminal source returns; otherwise last-mark cash settlement for terminal events, or carry the last mark pending a quote. Exceptions are logged."], ["Cash and benchmark", "Uninvested cash has zero modeled yield; all strategies use the same SPY investment window."]], [1, 2.8])
    p("Adjusted closes are return proxies, not literal executable auction prices. The ledger is a classroom approximation, not a full order-book, capacity or corporate-action settlement simulator. No borrowing is needed for the main long-only experiments. These execution assumptions are kept identical across each A/B comparison. SPY is buy-and-hold using vendor adjusted closes, without modeled entry or exit fees.")
    source("src/usq/project2/ledger.py; backtests/*_events.csv")

    page("12 / Main out-of-sample results", "Fusion does not improve net compound returns")
    table(["Strategy", "CAGR", "Volatility", "Sharpe", "Max DD", "Turnover"], [[label(n), percent(r.annual_return), percent(r.annual_volatility), f"{r.sharpe:.3f}", percent(r.max_drawdown), percent(r.mean_daily_turnover)] for n, r in main.iterrows()], [1.1, .8, .85, .7, .8, .9])
    image("nav", 263, "Figure 4. Main net wealth, 2023-2025. Solid lines are A; dashed lines are B; black is SPY. No final model is chosen using this figure.")
    p("Annualization uses 252 sessions: CAGR = terminal wealth^(252/N) - 1; volatility = sample daily standard deviation x sqrt(252); Sharpe = annualized arithmetic mean / volatility, with a zero risk-free rate.", small)
    p("The SPY benchmark earns 22.86% annualized over the common investment window. All six learned strategies lag it. Ridge A versus B earns 9.68% versus 8.89%; Lasso earns 9.02% versus 8.82%; XGBoost earns 10.56% versus 10.10%. These comparisons concern CAGR after the same 10 bps cost and five-session trading rule.")
    p("Adding fundamentals lowers mean daily one-way turnover in all families, but the gross-return loss is not fully offset by cost savings. A model can use more information and still deliver an inferior tail-ranked portfolio. The result does not justify calling fundamentals universally ineffective, nor does a positive absolute return constitute success against the stated investable alternative.")
    source("backtests/summary.csv; backtests/*_daily.csv")

    page("13 / Relative performance and losses", "Positive absolute returns can still disappoint an investor")
    image("relative_drawdown", 382, "Figure 5. Relative wealth equals strategy net wealth divided by SPY wealth. Drawdown is measured against each strategy's own running peak, including initial wealth 1.")
    table(["Strategy", "CAGR minus SPY", "Relative-wealth CAGR", "Relative max DD"], [[label(n), percent(r.annual_return_minus_spy), percent(r.relative_nav_annual_return), percent(r.relative_nav_max_drawdown)] for n, r in main.iterrows()], [1.2, 1.05, 1.2, 1.1])
    p("CAGR minus SPY and the CAGR of relative wealth are not interchangeable. The former subtracts two annual growth rates; the latter compounds the ratio of wealth paths. Neither is a CAPM alpha. This study uses SPY excess performance and Barra-style attribution, not CAPM or Fama-French regression.", small)

    page("14 / Stability across time", "Three years and six mechanically forecast halves")
    image("halves", 254, "Figure 6. Half-year net returns use the continuous portfolio path. They are not six independently liquidated and restarted strategies.")
    ann = read("backtests/annual.csv").pivot(index="strategy", columns="year", values="total_return").reindex(names)
    table(["Strategy"]+[str(x) for x in ann.columns], [[label(n)]+[percent(v) for v in row] for n, row in ann.iterrows()], [1.3, 1, 1, 1])
    p("Variation across half-years is substantial relative to small full-sample A/B differences. The models are mechanically refitted at each boundary, but factor membership, hyperparameters and the main trade schedule remain fixed. Realized earlier-half labels may enter subsequent fits only after they mature; no future-half target is consumed by the previous fit.")
    p("These tables are descriptive tests of stability, not invitations to retain only favorable dates. Excluding weak calendar years after seeing their results would change the original experiment. The study keeps the complete 2023-2025 path, including costs, instead of constructing a retrospectively favorable subperiod.")
    source("supplement/halfyear_performance.csv; backtests/annual.csv")

    page("15 / Prediction versus investment", "RankIC and low-cost returns do not tell the whole story")
    table(["Strategy", "Daily IC", "Daily RankIC", "Daily one-way turnover"], [[label(n), f"{model_ic.loc[n,'ic']:.4f}", f"{model_ic.loc[n,'rankic']:.4f}", percent(main.loc[n,'mean_daily_turnover'])] for n in names], [1.3, 1, 1, 1.3])
    image("cost", 257, "Figure 7. Costs are varied without refitting, changing signals or choosing a cheaper main scenario. The primary assumption remains 10 bps.")
    p("Fusion raises RankIC for the linear models in this sample without improving their selected portfolios' net CAGR. RankIC describes the entire eligible cross-section, whereas Top100 performance depends on a small rank tail and the delay between observation and execution. Prediction and allocation therefore need separate evaluation.")
    p("At zero assumed cost, XGBoost A earns about 20.54% annually, compared with 15.30% for B. At 10 bps they earn 10.56% and 10.10%. Fundamentals materially reduce cost sensitivity, but do not restore lost gross performance at the primary cost. At higher costs their ordering can reverse; that does not permit changing the main fee assumption after inspection.")
    source("evaluation/model_test_ic.csv; backtests/*_costs.csv")

    page("16 / Smoothing and turnover", "Smoothing trades information speed for lower activity")
    image("smooth", 357, "Figure 8. All original SMA/rebalance combinations are displayed for Ridge A/B. These investment-period outcomes are supplementary, not used to pick a new main rule.")
    table(["Ridge variant", "Net CAGR", "Gross CAGR", "Daily turnover"], [[f"{r.family}: SMA {int(r.smooth)}, every {int(r.rebalance_every)}", percent(r.annual_return), percent(r.gross_annual_return), percent(r.mean_daily_turnover)] for r in smooth.loc[smooth.rebalance_every.eq(5) & smooth.smooth.isin([1,10])].itertuples()], [1.7, 1, 1, 1])
    p("A longer moving average can reduce both gross opportunity and trading costs. For Ridge A, the ten-session-smoothed five-session rule lowers net CAGR relative to the frozen unsmoothed five-session rule. For B, some smoothed variants look better in this diagnostic. We report both patterns rather than promoting a favorable post-test variant.")
    p("Rebalancing less frequently and smoothing scores are different interventions. The first changes how often trades occur; the second changes which stocks are preferred. A fair turnover discussion must report gross return, net return and activity together, not just the largest final net wealth.")

    page("17 / Score stratification", "Do higher forecasts produce better portfolio groups?")
    image("groups", 443, "Figure 9. Five equal-capital, equal-weight score-group portfolios for each main model. G5 is the high-score group. All use the same delayed-close and fee convention.")
    p("Group-level net wealth tests score ordering beyond the Top100 tail. Each group is a separate funded portfolio, not one sleeve of a single fully invested strategy. Subtracting two wealth levels is not an executable long-short ledger, and no frictionless borrowable G5-minus-G1 strategy is claimed.")
    p("All six G5 portfolios have the highest group CAGR (8.08%-8.80%), but intermediate groups are not monotonically ordered; G1 exceeds G2-G4. The signal therefore separates the upper group without a smooth return gradient across all ranks. Correlation, group monotonicity and Top100 performance are different economic tests. None of these figures is used to refit the final model.", small)
    source("backtests/*_groups.csv")

    page("18 / Barra-style interpretation", "Fusion changes the economic portfolio")
    image("styles", 283, "Figure 10. Mean portfolio exposures use previous-close risk characteristics. These are absolute strategy exposures, not strategy-minus-SPY holdings exposures.")
    table(["Strategy", "Size", "Momentum", "Book / Price", "Earnings yield"], [[label(n)]+[f"{risks.loc[n,c]:+.3f}" for c in ["Size","Momentum","BookToPrice","EarningsYield"]] for n in names], [1.3, 1, 1, 1, 1])
    p("The repository's joint WLS estimates market, SIC2-industry and nine available style returns. Exposure at the prior close explains the current close-to-close stock return. Growth is not available; this is a research adaptation, not a commercial MSCI Barra implementation. Unknown industries are an explicit research-view category; original inputs are unchanged.")
    p("Across all three families, B has more negative Size, higher BookToPrice and negative Momentum exposure. Relative to A, it is a smaller-company, more value-oriented and less momentum-aligned portfolio. This association provides a coherent interpretation for changed returns, but is not evidence that these styles causally explain all A/B differences. SPY is a separate performance comparator, not an assumed same-pool style portfolio.")
    source("barra_style/mean_style_exposures.csv; src/usq/project2/risk.py")

    page("19 / Attribution and uncertainty", "Separate explanation from statistical evidence")
    table(["Strategy", "Style contribution", "Unexplained", "Cost contribution"], [[label(n), percent(attr.loc[n,"Style"]), percent(attr.loc[n,"Specific_or_unexplained"]), percent(attr.loc[n,"Costs"])] for n in names], [1.2, 1.2, 1, 1.2])
    p("Contributions above are Carino-linked cumulative contributions in return units, not annual returns. Full files also contain market, industry and explicit SPY-benchmark subtraction. Their sum equals strategy cumulative return minus SPY cumulative return; it does not equal the return on relative wealth. Daily reconciliation error is below 3 x 10^-17. The unexplained component includes omitted effects and classroom marking/settlement differences, so it is not labeled pure alpha.")
    image("uncertainty", 213, "Figure 11. 2,000 paired circular moving-block resamples, 20-session blocks, seed 5205. A and B always share the resampled dates.")
    table(["Model", "B - A CAGR", "95% interval, percentage points"], [[r.model.title(), f"{r.cagr_difference*100:+.2f} pp", f"[{r.cagr_diff_ci_low*100:+.2f}, {r.cagr_diff_ci_high*100:+.2f}]"] for r in uncertainty.itertuples()], [1, 1.2, 2])
    p("All intervals contain zero. They do not establish an improvement or prove equivalence. Only three investment years are observed, dependence may extend beyond the chosen block, and intervals are not adjusted for multiple comparisons. Arithmetic mean-return differences and CAGR differences are also distinct: higher volatility can reduce compound growth even when the average daily difference is positive.", small)

    page("20 / Supplementary baselines", "Simple fundamentals are not a failed information block")
    table(["Supplementary portfolio", "Net CAGR", "Sharpe", "Max DD", "Turnover"], [[label(n), percent(supplemental.loc[n,"annual_return"]), f"{supplemental.loc[n,'sharpe']:.3f}", percent(supplemental.loc[n,"max_drawdown"]), percent(supplemental.loc[n,"mean_daily_turnover"])] for n in base_names], [1.9, .8, .7, .8, .85])
    image("baselines", 245, "Figure 12. Equal-score C combines six fundamental percentile scores. It is not a newly trained C model or a newly selected main strategy.")
    p("The fundamental-only equal-score portfolio earns 19.88% annualized, versus 5.79% for equal-score A, 10.28% for equal-score B and 12.15% for the same eligible-pool equal-weight comparator. SPY still earns 22.86%. All score baselines retain the main eligible sample policy and costs; missing centered scores contribute zero rather than changing A/B sample eligibility.")
    p("This result narrows the main conclusion. The learned fusion pipeline does not improve realized net CAGR, but transparent fundamental aggregation can remain competitive and low-turnover. Possible explanations include estimation weights optimized for average ten-day squared-error prediction rather than portfolio utility, rank-tail concentration and style shifts. These are interpretations to investigate, not causal findings. The baseline was added after viewing main results and is not promoted to replace the main experiment.")

    page("21 / Single-factor mechanisms", "Which financial signal and which style?")
    image("single_styles", 284, "Figure 13. Development-period monthly Top100 target exposures, not realized holding attribution. Available snapshots are averaged; missing exposure weights are separately recorded.")
    table(["Single fundamental signal", "2023-2025 net CAGR", "Sharpe", "Max DD"], [[label(n), percent(supplemental.loc[n,"annual_return"]), f"{supplemental.loc[n,'sharpe']:.3f}", percent(supplemental.loc[n,"max_drawdown"])] for n in fundamentals], [1.9, 1.25, .8, 1])
    p("The FCF-yield portfolio earns 18.51% annually and the quarterly-ROE residual earns 15.26%. Other financial signals remain positive in absolute terms, but quarterly SP has a much larger maximum drawdown. These are fixed-direction descriptive portfolios; early direction fitting is in-sample for 2015-2018. No favorable single-factor test result is fed back into the main learned feature set.")
    p("Monthly 2015-2022 snapshots describe factor-style relationships, with development and 2022 validation stored separately. They do not claim complete daily realized attribution in the early warm-up period. Separate 2023-2025 single-factor attribution uses the same fitted risk model as the learned portfolios, with saved holdings and daily return reconciliation.", small)
    source("supplement/monthly_target_style_exposures.csv; supplement/barra_style/; supplement/portfolio_summary.csv")

    page("22 / Conclusion", "What worked, what did not, and what we learned")
    plain("Main answer", sub)
    p("Under the fixed labels, sample eligibility, model grid and five-session Top100 investment rule, adding the six fundamental factors does not improve realized 2023-2025 net compound returns in Ridge, Lasso or XGBoost. All three pairs show lower turnover after adding fundamentals. Cost savings are real in the simulation, but they are insufficient to offset changes in gross opportunity and portfolio composition at the primary fee assumption.")
    plain("Why this is a useful result", sub)
    p("The research separates signal replication, average prediction quality and actual portfolio construction. Reconstruction is not uniformly better than a ratio; a higher RankIC does not guarantee a stronger Top100 portfolio; reduced turnover does not necessarily improve net returns; and a positive strategy return can still trail the investable market benchmark. These distinctions are more useful than retroactively choosing a single attractive curve.")
    plain("Interpretation rather than overclaiming", sub)
    p("Risk diagnostics associate fusion with smaller-company, value and negative-momentum tilts. The stronger simple fundamental baseline suggests that information aggregation matters, but it is a supplementary finding rather than an independently confirmed production strategy. Uncertainty intervals are wide. We therefore reject neither all fundamental information nor all nonlinear learning; we find no reliable fusion advantage for this particular pipeline.")
    plain("Scope and limitations", sub)
    p("Financial statements are release-aligned latest-download values, not strict PIT. Some source reports postdate the investment sample, making this a retrospective transfer of formulas rather than evidence that those ideas were available in 2015. The main chain uses close_adj and avoids VWAP-dependent formulas. A simplified event-aware close ledger and nine-style research model do not establish real auction fills, capacity or commercial risk coverage. The final three-year period is short, and supplementary diagnostics are not a fresh holdout.")
    p("Future work could pre-register a new calendar period or an independent universe, test a lower-turnover portfolio objective and compare aggregation methods without changing the present conclusions. Those experiments have not been run here. The submitted evidence should preserve this fixed main experiment and clearly label any subsequent work.")

    page("23 / Reproducibility", "From the same local snapshot to the reported artifacts")
    p("The code consumes a trusted team-owned database snapshot. Large source data, pickle panels, model objects, personal paths and credentials are not included in a public repository. The data-build code and manifests describe preparation; licensed inputs must be obtained separately. A clean machine requires the same snapshot, not only the repository. Set DATA below to its actual local location; research_database is an illustrative relative directory.")
    for command in ["python3.13 -m venv .venv-project2\nsource .venv-project2/bin/activate\npython -m pip install -r requirements-project2.txt", "# macOS XGBoost requires OpenMP\nbrew install libomp", "# From repository root; DATA and OUT are task-specific paths\nDATA=../team17_close10_2014_2025\nOUT=../project2_research_fresh", "PYTHONPATH=src python -m usq.project2.run \\\n  --database $DATA --output $OUT --all", "PYTHONPATH=src python -m usq.project2.supplement \\\n  --database $DATA --output $OUT\nPYTHONPATH=src python -m usq.project2.risk \\\n  --database $DATA --output $OUT --supplement", "MPLCONFIGDIR=/tmp/usq-project2-mpl PYTHONPATH=src \\\n  python -m usq.project2.paper --database $DATA \\\n  --output $OUT --pdf-output $OUT/Team34_Report.pdf", "PYTHONPATH=src python -m pytest tests -o addopts='' -q"]:
        command = command.replace("team17_close10_2014_2025", "research_database")
        p(escape(command).replace("\n", "<br/>"), code_style)
    p("Tested research environment: Python 3.13.5; NumPy 2.5.4; pandas 3.1.1; PyArrow 26.0.1; scikit-learn 1.9.1; XGBoost 3.1.3. Random seed: 5205; XGBoost CPU threads: 4. Package locks are in requirements-project2.txt. A clean-environment reproduction of the entire pipeline has not been independently completed by a second team member.", small)
    source("README.md; requirements-project2.txt; research_verification.json; supplement/protocol.json")

    page("24 / References and submission", "Sources, implementation map and Team 34 completion notes")
    refs = ["[R1] Kakushadze, Z. (2016). 101 Formulaic Alphas. Wilmott 2016(84), 72-80. arXiv:1601.00991. https://arxiv.org/abs/1601.00991. Formula appendix; only the explicitly listed subset is implemented.",
            "[R2] Orient Securities (2024-03-21). Reconstructing Fundamental Factors (Jibenmian Yinzi de Chonggou), factor stock-selection series 102. Team-supplied Chinese report, pp.8,23,25,29,37. English title is a descriptive translation.",
            "[R3] Northeast Securities (2026-07-22). Quality and Trend Enhancement of a Free-Cash-Flow Strategy. Team-supplied Chinese report, p.5. English title is a descriptive translation; only the FCF signal is adapted.",
            "[R4] China Merchants Securities (2026-05-18). Beyond Analyst Expectations: Exploring Earnings-Expectation Models. Team-supplied Chinese report, pp.12-13,27. English title is a descriptive translation; simple expectation version.",
            "[R5] DSA5205, Project 2 assignment and grading rubric (2026). Team-provided course specification: systematic trading design, validation, evaluation and reproducibility.",
            "[R6] Quant_U.S._stocks research code, local working tree based on https://github.com/Panyuxiang0830/Quant_U.S._stocks. Research additions: src/usq/project2/. The current local work has not been newly pushed for this report.",
            "[R7] global-stock-data: https://github.com/simonlin1212/global-stock-data. SPY Yahoo-chart ingestion pattern; cached source metadata records upstream commit b7f22029660769baa29a4403cc61c285f35f8905. Company panels and Compustat records are separately team-supplied."]
    for r in refs: plain(r, small)
    plain("Implementation and figure traceability", sub)
    table(["Research component", "Source / artifact"], [["Factor formulas and selection", "factors.py; screen.py; Figures 2-3"], ["Training and execution", "models.py; ledger.py; Figures 4-9"], ["Style interpretation", "risk.py; exposures.py; Figures 10,13"], ["Supplementary uncertainty / baselines", "supplement.py; Figures 11-12"], ["Report generation", "paper.py; CSV-backed figures and machine-readable checks"]], [1.25, 2.3])
    p("<b>Team 34 completion notes.</b> The team identifier is confirmed. Full member names, student IDs and a truthful individual contribution statement must be supplied by the team before submission; they have not been invented here. The 15-25 minute presentation video remains to be recorded. Confirm institutional data-access permissions, run the README on a second member's machine, and provide the final reviewed code link or source archive. No Canvas submission or new Git push has been performed.", small)
    p("Historical research only. This report does not place orders, provide a live signal service or guarantee future investment performance.", small)

    def furniture(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(TEAL); canvas.setLineWidth(1.4)
        canvas.line(45, A4[1]-30, A4[0]-45, A4[1]-30)
        canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#647483"))
        canvas.drawString(45, 23, "TEAM 34  |  DSA5205 PROJECT 2  |  RESEARCH DRAFT")
        canvas.drawRightString(A4[0]-45, 23, f"{doc.page}")
        canvas.restoreState()
    document = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=45, rightMargin=45,
                                 topMargin=45, bottomMargin=44, title="Team 34 - Fundamental and Price-Volume Stock Selection",
                                 author="Team 34", subject="DSA5205 Project 2 - retrospective U.S. equity research")
    document.build(story, onFirstPage=furniture, onLaterPages=furniture)
    reader = PdfReader(pdf)
    texts = [page.extract_text() or "" for page in reader.pages]
    if len(reader.pages) != 25:
        raise ValueError(f"Expected 25 designed pages, got {len(reader.pages)}; inspect overflow before delivery")
    for i, title in enumerate(sections):
        if " ".join(title.split()) not in " ".join(texts[i].split()):
            raise ValueError(f"Section overflow / wrong page: {i+1}: {title}")
    for forbidden in ("Team 17", "team17", "Team 34 TBD", "<codex", "TODO"):
        if any(forbidden in t for t in texts):
            raise ValueError(f"Unresolved or wrong-team content: {forbidden}")
    # An input-directory basename is not a team attribution; retain literal CLI paths.
    report_manifest = {"team": "Team 34", "pages": len(reader.pages), "figure_count": len(list(figures.glob('*.png'))),
                       "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                       "main_results_not_refit": True, "roster_provided": False,
                       "section_titles": sections, "page_text_characters": [len(t) for t in texts]}
    write_json(report_manifest, out/"paper_manifest.json")
    # Compact Chinese companion, separate from the English submission body.
    companion = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>Team 34 研究解读与交付清单</title>
<style>body{font-family:-apple-system,'PingFang SC',sans-serif;max-width:900px;margin:40px auto;padding:0 24px;line-height:1.9;color:#17334d}h1,h2{color:#00877e}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}</style>
<h1>Team 34：英文报告配套中文解读</h1><p>英文正文为25页工作初稿。组别统一为 Team 34；尚未收到姓名、学号和个人分工，不编造成员信息。</p>
<h2>研究主线</h2><p>2015-2021选因子，2022只选参数和交易规则，2023-2025每半年重训。训练历史始终从2015累积，先剔除当时尚未实现的10日标签。主实验为三个模型各一对：纯量价A，与加入六个基本面因子的B。</p>
<h2>应该怎么讲结论</h2><p>六组模型扣费后净年化约8.82%-10.56%，都落后于同期SPY的22.86%。基本面降低了换手，却没有提高主实验的净复合收益；收益差置信区间均跨零，不能证明稳定增强。不能把这个结论扩张成“基本面没用”。</p>
<p>补充等权基本面组合年化19.88%，说明聚合方法和权重可能很重要，但这是看到主结果后补做的诊断，不能事后冒充最初主策略。Barra-style显示融合模型更偏小盘、价值和负动量；特异贡献不是可直接宣称的alpha。</p>
<h2>20分钟视频建议分配</h2><table><tr><th>时间</th><th>讲什么</th></tr><tr><td>0-3分钟</td><td>经济逻辑、资产范围、SPY基准</td></tr><tr><td>3-7分钟</td><td>六个研报因子、36个WQ候选筛选四个、时间切分与标签时点</td></tr><tr><td>7-11分钟</td><td>演示数据接口、验证结果、半年累积训练与延迟收盘执行</td></tr><tr><td>11-16分钟</td><td>净值、回撤、分层、费用和平滑</td></tr><tr><td>16-20分钟</td><td>Barra-style、置信区间、简单基线与研究结论</td></tr></table>
<h2>正式提交前还剩什么</h2><ol><li>补全Team 34成员姓名、学号和真实分工；审核英文措辞。</li><li>由另一位同学按README在自己的电脑运行，确认同一快照可复现。</li><li>录制15-25分钟视频；检查教职人员可访问并下载。</li><li>确认商业数据的学校授权，不在公共Git仓库或提交代码中附大数据文件和凭据。</li><li>经小组确认后再推送代码或打包源码；由组员自行提交Canvas。</li></ol>
<p>本轮不再改因子、参数或主交易规则。没有自动推送Git，也没有代为提交。</p></html>"""
    (out/"Team34_研究解读与交付清单.html").write_text(companion, encoding="utf-8")
    print(json.dumps({"stage": "paper_complete", "pdf": str(pdf), **report_manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", help="Optional full snapshot; omit to rebuild from published aggregate report_inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pdf-output", required=True)
    args = parser.parse_args()
    run(args.database, args.output, args.pdf_output)
