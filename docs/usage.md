# Usage

`btcfloor` is a local research toolkit. Source code, tests, and docs are
tracked in git; downloaded market data and generated reports live under
`data/` and `reports/` and are intentionally ignored.

## Daily update

For normal use, run the full updater:

```powershell
uv run scripts/update_daily.py
```

This refreshes:

- BTC daily data and core floor reports,
- tactical SMA/channel and floor-convergence plots,
- Checkonchain cohort and LTH realised-loss charts,
- interactive BTC market and ROI dashboards,
- metals/GSR data and the metals relative-strength dashboard.
- data/pipeline health status.

Primary dashboard entry points:

```text
reports/interactive/btc_market_dashboard.html
reports/interactive/btc_roi_dashboard.html
reports/interactive/metals_relative_dashboard.html
reports/interactive/pipeline_health_dashboard.html
reports/interactive/btc_floor_weekly.html
```

## Focused commands

Run only the core floor analysis:

```powershell
uv run btcfloor analyze --force-download
```

Regenerate only the original weekly Plotly chart:

```powershell
uv run btcfloor chart
```

Regenerate only the dashboard HTML from existing report CSV/JSON files:

```powershell
uv run scripts/build_interactive_dashboards.py
uv run scripts/build_metals_dashboard.py
uv run scripts/build_pipeline_health_dashboard.py
```

Regenerate only Checkonchain cohort figures:

```powershell
uv run scripts/plot_checkonchain_cohorts.py
```

## Data freshness

Coin Metrics is the canonical long-history BTC source. If it lags, the
pipeline appends only missing recent BTC/USD daily rows from CoinGecko's public
`market_chart/range` endpoint. Daily rows are UTC-dated, so Europe/Stockholm
can be on a new calendar day before a new UTC daily close exists.

Checkonchain cohort charts are parsed from public static Plotly pages.
Classic CVDD uses Bitbo when `BITBO_API_KEY` is configured; otherwise the
dashboard labels the Looknode public endpoint as a third-party fallback.

Metals/GSR uses Yahoo Finance COMEX futures (`GC=F`, `SI=F`) for live decision
monitoring. LBMA fixes are kept only for long-history analog context.

## Price fixes

Manual BTC data fixes can be recorded in `config/price_fixes.csv` with columns:

```csv
date,action,price_usd,reason
```

Supported actions are `replace` and `drop`.
