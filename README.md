# btcfloor

`btcfloor` is a reproducible Python research toolkit for Bitcoin floor-price
analysis. It combines fixed power-law floors, weekly expectile bands, cycle
timing assumptions, forward floor-overlap signals, and validation reports into
one maintainable command-line package.

The goal is not to predict price. The goal is to make floor-pressure research
auditable: formulas live in code, reports are generated locally, and generated
market data is kept out of git.

## Features

- Giovanni fixed BTC power-law floor:
  `1.0117e-17 * days_since_genesis^5.82 * 0.42`.
- Burger 2019 RANSAC-style support reference, fit only on the Sep 2019
  vintage window and displayed as a lower inlier-sigma band.
- LuxAlgo-inspired weekly expectile power-law approximation, including the
  requested `0.01%` bottom expectile.
- Forward floor-distance checks for 1, 3, 6, 9, and 12 month horizons.
- Historical forward-floor overlap episodes.
- 1428-day cycle timing table using 1064 low-to-peak days and 364
  peak-to-low days.
- Full-sample and walk-forward cycle-low validation.
- Stability checks that refit adaptive models after excluding recent cycles.
- Static diagnostic plots and a standalone interactive Plotly weekly chart.
- Interactive Plotly market/on-chain and ROI/deployment dashboards with
  regenerated agent commentary.
- Interactive metals relative-strength dashboard with live COMEX futures GSR
  monitoring, rotation levels, and legacy long-history analog context.
- Interactive BTC/gold rotation dashboard with daily and weekly BTC/XAU moving
  average confirmation.
- `uv`-managed package, CLI, tests, docs, CI workflow, and git hygiene.

## Quick Start

Install dependencies and run tests:

```powershell
uv venv
uv sync --all-groups
uv run pytest -q
```

Run the core floor analysis:

```powershell
uv run btcfloor analyze
```

For normal daily use, refresh the full local decision surface. This updates
BTC market data, floor reports, tactical images, Checkonchain cohort data,
metals/GSR data, BTC/gold rotation data, pipeline health, and interactive
dashboards:

```powershell
uv run scripts/update_daily.py
```

Then open the primary dashboards:

- `reports/interactive/btc_market_dashboard.html`
- `reports/interactive/btc_floor_weekly.html`
- `reports/interactive/btc_roi_dashboard.html`
- `reports/interactive/btc_gold_rotation_dashboard.html`
- `reports/interactive/metals_relative_dashboard.html`
- `reports/interactive/pipeline_health_dashboard.html`

For automated visual checks with the Codex Browser, serve the repo over local
HTTP instead of using `file://`, because the Browser URL policy can block
automation on local file tabs:

```powershell
uv run python -m http.server 8765 --bind 127.0.0.1 --directory C:\CodexProjects\btcfloor
```

Then open, for example,
`http://127.0.0.1:8765/reports/interactive/btc_gold_rotation_dashboard.html`.

Generate only the interactive weekly chart:

```powershell
uv run btcfloor chart
```

Generated outputs are written under `data/` and `reports/`. These directories
are intentionally ignored by git.

## Commands

```powershell
uv run btcfloor download
uv run btcfloor diagnose
uv run btcfloor analyze
uv run btcfloor analyze --force-download
uv run btcfloor chart
uv run scripts/update_daily.py
uv run scripts/build_interactive_dashboards.py
uv run scripts/build_metals_dashboard.py
uv run scripts/build_btc_gold_dashboard.py
uv run scripts/build_pipeline_health_dashboard.py
```

## Research Outputs

The full analysis creates local artifacts such as:

- `reports/model_snapshot.csv`
- `reports/risk_horizons.csv`
- `reports/risk_role_based.csv`
- `reports/forward_floor_overlap_episodes.csv`
- `reports/model_evidence_summary.csv`
- `reports/walk_forward_cycle_low_validation.csv`
- `reports/stability.csv`
- `reports/interactive/btc_floor_weekly.html`
- `reports/interactive/btc_market_dashboard.html`
- `reports/interactive/btc_roi_dashboard.html`
- `reports/interactive/btc_gold_rotation_dashboard.html`
- `reports/interactive/metals_relative_dashboard.html`
- `reports/interactive/pipeline_health_dashboard.html`

These files are reproducible outputs, not source files.

## Methodology

See [docs/methodology.md](docs/methodology.md) and [MODEL_NOTES.md](MODEL_NOTES.md)
for formulas, assumptions, and model limitations.

Short version:

- The Giovanni floor is fixed and does not refit to recent market data.
- The Burger 2019 RANSAC support uses the same pre-2022 vintage window for the
  current snapshot, so later lows cannot move that reference line.
- The expectile floor is a transparent Python asymmetric least-squares fit over
  weekly BTC closes.
- The forward-floor signal asks whether current price overlaps a future floor:
  `spot_today < floor(today + horizon)`.
- Cycle timing is an explicit research assumption, not a forecast guarantee.
- Dashboard commentary is generated from the latest report CSV/JSON outputs so
  each update has a current quantitative read.

## Data Sources

The long-history source is the Coin Metrics community BTC CSV archive:

```text
https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv
```

If Coin Metrics lags the current date, the pipeline fills only the missing
recent daily BTC/USD rows from CoinGecko's public `market_chart/range` endpoint.
Generated `data_quality.md` reports the final source row counts.

The Checkonchain cohort charts are parsed from their public static Plotly HTML
pages and written to `data/processed/checkonchain_cohort_metrics.csv`. The
current dashboard uses STH-MVRV, STH-MVRV Z-score, LTH-MVRV, LTH-SOPR, LTH
realised price, STH realised price, Cointime Price, and LTH realised-loss
impulse metrics. Classic Woo/Bitbo CVDD is added when an authorized Bitbo API
key is available through `BITBO_API_KEY`; otherwise the dashboards use
Looknode's public classic-formula CVDD endpoint as a third-party fallback and
label it accordingly. BGeometrics CVDD is not used for tactical floor charts
because its documented current-supply normalization puts it on a different
scale.

The metals relative-strength dashboard uses Yahoo Finance COMEX futures
(`GC=F`, `SI=F`) for live gold/silver and GSR decision monitoring. It writes
`data/processed/yahoo_gold_futures.csv`,
`data/processed/yahoo_silver_futures.csv`, and
`data/processed/metals_gsr_daily.csv`, then marks GSR decision levels for
gold-vs-silver rotation. LBMA gold/silver fixes are retained only for the
legacy long-history analog panels until a better long-history metals source is
wired in.

The BTC/gold rotation dashboard combines the processed BTC daily series with
Yahoo Finance COMEX gold futures (`GC=F`) and writes
`data/processed/btc_gold_ratio_daily.csv`,
`data/processed/btc_gold_ratio_weekly.csv`, and
`reports/btc_gold_rotation_summary.json`. The ratio is close-to-close on the
latest shared BTC/gold trading date; weekend gold prices are not fabricated.

Daily rows are UTC-dated. Around a local midnight, the latest processed daily
row may still be the prior local calendar date until a UTC-dated row exists.

The raw and processed data files are generated locally and ignored by git.

## Repository Hygiene

The repository intentionally tracks source code, tests, and documentation.
It intentionally ignores:

- downloaded market data,
- generated reports and charts,
- Python bytecode and caches,
- local virtual environments,
- local secrets or `.env` files.

## Development

```powershell
uv sync --all-groups
uv run pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Disclaimer

This is research software, not financial advice. BTC is volatile, and model
floors can fail. See [docs/disclaimer.md](docs/disclaimer.md).

## License

MIT License. See [LICENSE](LICENSE).
