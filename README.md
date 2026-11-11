# DSA5205 Project 2 — Equity Strategy Research

Team 34: Wan Gehong, Wang Tianyi, Wu Jiarui, Yan Tianao.

The main entry point is [Equity Strategy Research](notebooks/Equity_Strategy_Research.ipynb). It builds the research dataset from authorized inputs, screens signals, trains and validates models, and evaluates portfolios net of trading costs. See [项目最新进度.md](项目最新进度.md) for the changes since the previous team version and the remaining deliverables.

## Setup and execution

From the repository root, create a Python 3.13 environment and install the project:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m ipykernel install --user --name team34-project2 --display-name "Python (Team 34 Project 2)"
```

On macOS, XGBoost may require the OpenMP runtime (`brew install libomp`).

Obtain the team's authorized research snapshot and place its contents under `data/`. The notebook expects `data/panels/`, `data/metadata/` and `data/checksums.sha256`. Alternatively, set `DATA_DIRECTORY` in the first notebook code cell to the existing snapshot folder, or set `TEAM34_DATA_DIR` before launching Jupyter. The chosen folder must directly contain `panels/`, `metadata/`, and `checksums.sha256`; do not select `panels/` itself. No second copy or renaming of the database is necessary. The first cell now checks the path and explains missing files before reading any pickle.

For a terminal execution with data stored elsewhere:

```bash
export TEAM34_DATA_DIR="/your/local/path/team17_close10_2014_2025"
```

This is a supplied market and accounting panel snapshot with a cached SPY benchmark, and is excluded from Git. Only use trusted pickle files. The notebook verifies input hashes, audits coverage and financial availability, and writes a separate cleaned dataset with rebuilt labels. The GitHub ZIP alone does not include `open.pkl` or the other database fields.

Open the notebook with the **Python (Team 34 Project 2)** kernel and choose **Restart Kernel → Run All**. A terminal run from the repository root is also supported:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=team34-project2 \
  --ExecutePreprocessor.timeout=-1 \
  --output=Equity_Strategy_Research.executed \
  --output-dir=runs \
  notebooks/Equity_Strategy_Research.ipynb
```

For a fresh-kernel execution with cell progress and a saved failure record, without registering a user-level kernel, use:

```bash
.venv/bin/python scripts/execute_notebook.py \
  --data "/your/local/path/team17_close10_2014_2025" \
  --output runs/notebook_validation
```

Use a new `--output` directory for each attempt. The executed notebook and `execution_status.json` are saved there; the independently constructed database and model artifacts are placed in a new timestamped `runs/experiment_...` directory. The runner stops at the first error and records consecutive code-cell execution counts on success.

Allow time and disk space for data preparation, 396 model fits, 38 portfolio ledgers and the daily factor model. A fresh execution builds a new timestamped experiment under `runs/` from the supplied snapshot.

**Verified release (11 November 2026).** The saved notebook now comes from one successful fresh-kernel, top-to-bottom execution: 13 code cells numbered 1–13, 396 fresh learner fits without checkpoint reuse, and 38 reconciled portfolio ledgers. Execution took approximately 87 minutes on the validation machine. All 64 automated tests passed. Portfolio summary metrics, the WQ screen and frozen choices reproduce upstream commit `a4a31d6`. The updated 21-page English report is [Group_34_Report.pdf](reports/final/Group_34_Report.pdf); its figures and tables use this verified run. See [VALIDATION.json](VALIDATION.json) for verification records. The new video and confirmation of individual team contributions remain pending. The shared source archive excludes the database and fitted models.

## Research design

| Stage | Dates and rule |
|---|---|
| WQ direction and screening | Calibrate directions in 2015–2017. Screen 36 WQ candidates in 2018–2019 using RankIC, coverage, annual stability and correlation. |
| Hyperparameter selection | Use expanding-window CV with annual scoring folds in 2020–2022. Compare fundamental, fundamental plus WQ, and engineered features for Ridge, Lasso and XGBoost. |
| Portfolio selection | In 2023, compare four scores under three weightings: equal, square-root capitalization and capitalization. Select the highest net Sharpe among the 12 candidates with finite Sharpe and mean cash at most 5%. |
| Chronological evaluation | Evaluate the frozen choices in 2024–2025. ML models refit every half-year on the preceding five calendar years using only labels that matured before the forecast boundary. |

Signals formed at the close are executed at the next close. The supervised target is the adjusted return from T+1 close to T+11 close. Portfolio tests use a Top100 target, rank-150 retention buffer, ten-session rebalancing, 2% formation weight cap and a 10 bps charge per traded dollar. SPY and matched full-pool portfolios are benchmarks. Accounting vintages, universe provenance and unresolved terminal payouts constrain interpretation.

## Results and diagnostics

The 2023 validation selects **six-factor direct scoring with capitalization weights**. Its 2024–2025 net CAGR is **24.92%**, net Sharpe **1.407**, and maximum drawdown **−20.76%**. On the same dates, SPY records **22.19%**, **1.307**, and **−18.76%**, respectively.

- [Portfolio summary](results/portfolio_summary.csv) and [selection record](results/frozen_family.json) document the candidate grid and decision.
- [Period metrics](results/period_metrics.csv) and [prediction OOS R²](results/period_oos_r2.csv) support the four-panel comparison. The six-factor score is mapped to return units with pooled OLS using mature labels from the same past five-year windows. This diagnostic leaves the portfolio's direct ranking rule intact.
- The notebook includes the selection, weighting and universe decomposition, plus holdings-based Barra-style attribution for the selected portfolio. [Factor exposures and contributions](results/barra_factors.csv) and [residual alpha](results/barra_alpha.csv) use the available market, SIC industry and nine style factors. Growth is unavailable. Residual alpha is conditional on this research risk model.
- [Portfolio audit](results/portfolio_audit.json) records ledger reconciliation. Full daily artifacts are written under `runs/`.

Run regression tests from the repository root:

```bash
.venv/bin/python -m pytest tests -q
```

Source is under `src/`, tests under `tests/`, and compact result tables under `results/`. The local `data/`, `backups/` and `runs/` directories are excluded from the repository.

## Edit and rebuild the report

The PDF and editable LaTeX are under `reports/final/`. To edit text and compile the supplied tables and figures, no database or model training is needed:

```bash
cd reports/final
TEXINPUTS=./tex-support: pdflatex -interaction=nonstopmode -halt-on-error Group_34_Report.tex
TEXINPUTS=./tex-support: pdflatex -interaction=nonstopmode -halt-on-error Group_34_Report.tex
```

LaTeX is required. Run twice for the contents and cross-references. For a results-driven report rebuild after a new successful tracked Notebook execution, run `scripts/build_notebook_report.py` from the repository root with `--work` pointing to that run's `runs/experiment_...` directory and `--executed-notebook` pointing to its executed notebook. It requires the clean-run verification record, fresh learner fits, and the ledger audit. Its report prose template is `reports/final/notebook_report_template.tex`; direct edits to the generated `Group_34_Report.tex` are replaced by a future results-driven rebuild.

The supplied history was already inspected in earlier research. The revised 2024–2025 evaluation is chronological, not a newly untouched blind holdout. The latest code uses five-year forecast-training windows and fundamental/combined/engineered input comparisons; these differ from the earlier expanding-window, price-only A versus mixed B design.
