# DSA5205 Project 2 — Equity Strategy Research

Team 34: Wan Gehong, Wang Tianyi, Wu Jiarui, Yan Tianao.

This project builds a US equity dataset from raw downloads and compares a six-factor direct score with Ridge, Lasso and XGBoost under common portfolio rules. Both notebooks contain the saved outputs of a complete sequential execution on 11 November 2026.

## Notebooks

Run these notebooks in order, with a fresh kernel for each:

1. [Equity Data Preparation](notebooks/Equity_Data_Preparation.ipynb): source inventory, market-quality checks, dated security mapping, financial and industry events, monthly universe construction, total-return prices, next-close labels and source-coverage audits.
2. [Equity Strategy Research](notebooks/Equity_Strategy_Research.ipynb): input verification, factor screening, feature engineering, chronological CV, validation selection, evaluation, robustness, return attribution and accounting checks.

Data preparation reads raw inputs without requiring an older research database. It publishes `data/processed/latest_snapshot.json` only after every stage succeeds. The strategy notebook verifies the completed dataset, its checksums, canonical price chain and labels. Every execution creates a new timestamped run and trains models afresh.

## Local setup

Use Python 3.13 from the repository root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m ipykernel install --user --name team34-project2 --display-name "Python (Team 34 Project 2)"
```

Select **Python (Team 34 Project 2)** and use **Restart Kernel → Run All** in each notebook. On macOS, XGBoost may require `brew install libomp`.

Raw datasets are supplied separately and remain local:

```text
data/
  raw/
    wrds/14_1_crsp_daily/        CRSP daily prices, names and delistings
    wrds/14_2_compustat/         downloaded Compustat extracts
    wrds/14_3_ccm_link/          dated CCM security links
    compustat/quarterly/Comp_Quarterly6126.csv.zip
    compustat/annual/20250102updated north fundamental_dta.zip
    benchmark/SPY.gz            archived raw Yahoo chart response
    benchmark/SPY.json          source provenance
    sina/<capture>/            supplementary raw vendor responses
  processed/
    <data-run>/research_snapshot/
    latest_snapshot.json
```

The quarterly archive supplies full financial history and release proxies; the North America annual archive supplies historical SIC. Sina is checked with fresh requests, but its turnover/VWAP is excluded from strategy inputs because coverage is incomplete. Liquidity screening uses CRSP unadjusted close × share volume.

Allow several GB of free space for the published dataset, training matrices and model outputs. `runs/` contains local execution artifacts; `results/` contains compact, shareable summaries. Data, backups, model files and per-security forecasts/holdings are excluded from Git. Saved notebook outputs are retained.

## Research design

| Stage | Protocol |
|---|---|
| Data | CRSP 2014–2025, with earlier Compustat history for accounting features. Dated CCM links; explicit quote, missing-data and delisting treatment. |
| Universe | Monthly top 80% by 60-session mean dollar-volume proxy; 252 observed closes; raw close ≥ $5; daily security/price gates. The completed dataset contains 4,366 ever-selected securities. |
| Prices and targets | Causal float64 total-return chains from source returns. Target: T+1 close to T+11 close. Unmatured tail labels remain missing. |
| WQ screening | Directions from 2015–2017; 36 candidates screened in 2018–2019 using RankIC, coverage, annual stability and correlation. Six factors pass. |
| Model tuning | Three feature sets: fundamentals; fundamentals plus WQ; all engineered features. Expanding annual CV folds in 2020–2022 with purged labels and fresh semiannual fits: 63 configurations, 189 scored folds. |
| Portfolio selection | In 2023, compare four scores × three weightings: equal, square-root cap and market cap. Select maximum net Sharpe among finite-Sharpe candidates with mean cash ≤ 5%. |
| Evaluation | Freeze specifications before 2024–2025 evaluation. ML forecasts refit semiannually on the preceding five calendar years using mature labels only. |

Trading uses next-close execution, Top100 selection, a rank-150 retention buffer, ten-session rebalancing, a 2% formation-weight cap and 10 bps per traded dollar. SPY and matched full-pool portfolios are controls. Costs, leave-one-factor-out checks and benchmark-relative return attribution use the six-factor market-cap portfolio; holdings-based risk attribution uses the validation-selected portfolio.

WQ price and screening calculations retain float64 precision before final model-feature storage. This avoids changes caused by differently normalized, prematurely rounded adjustment factors.

## Latest results and verification

The 2023 rule selects **six-factor direct score with market-cap weights** (`fundamental_cap`), with validation net Sharpe **1.3325**.

| 2024–2025 | Net CAGR | Annual volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Selected strategy | 24.92% | 16.81% | 1.4074 | −20.76% |
| SPY | 22.19% | 16.35% | 1.3071 | −18.76% |

The notebooks include OOS prediction R², period-level return/risk plots, trading-cost and factor-ablation checks, linked return attribution, and a US adaptation of the repository's Barra-style market/industry/style model. Estimated residual alpha is conditional on that model and is not evidence of a guaranteed tradable alpha.

Verification completed: **74 tests, 396 fresh model fits, 38 portfolio ledgers**, source-hash checks, universe truncation checks and an independent source-return check of 8,945,343 labels. A repeated XGBoost fit produced exactly identical predictions. See [reproduction audit](results/reproduction_audit.json) and [Chinese project status](项目最新进度.md).

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q
```

Financial statements are latest-download vintages; release dates are proxies, FIC/PRIUSA are static descriptors, and some terminal payouts remain unknown. The 2024–2025 results are a historical evaluation of previously inspected data, rather than a new blind holdout.

**Deliverable status (12 November 2026):** the 21-page English [report](reports/final/Group_34_Report.pdf) now matches the saved results at upstream commit `743b82c`, including the raw-data pipeline, precision policy, selected model specifications, portfolio metrics, capitalization-weighted sensitivity tests, current attribution path and holdings-based risk analysis. This report revision did not train models, replay portfolios or resample bootstrap intervals. Its [input inventory](reports/final/notebook_report_inventory.json) records file hashes and the [report check](REPORT_VALIDATION.json) records publication verification. The new presentation video and confirmation of individual contributions/student IDs remain pending.

To refresh report assets from the saved CSVs and Notebook outputs, without running either research notebook:

```bash
.venv/bin/python scripts/build_report_from_saved_results.py
cd reports/final
TEXINPUTS=./tex-support: pdflatex -interaction=nonstopmode -halt-on-error Group_34_Report.tex
TEXINPUTS=./tex-support: pdflatex -interaction=nonstopmode -halt-on-error Group_34_Report.tex
```

LaTeX is required only for PDF compilation. The editable prose is `reports/final/notebook_report_template.tex`; the asset script copies it to the generated report source. Confidence intervals are read from the saved strategy Notebook at its displayed precision. The saved wealth-path image is retained, and other charts are redrawn from stored aggregates. Raw data and fitted model files are not required for this report-only rebuild.
