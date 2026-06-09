# Usage

`btcfloor` is a research toolkit. Source code, tests, and docs are tracked in
git; downloaded market data and generated reports live under `data/` and
`reports/` and are intentionally ignored. The scheduled refresh path runs in
GitHub Actions; local commands are for manual reproduction and debugging.

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
- metals/GSR data and the metals relative-strength dashboard,
- BTC/gold rotation data and dashboard,
- data/pipeline health status.

The GitHub Actions workflow runs the same update every 4 hours, publishes the
rebuilt static site to GitHub Pages, and uploads the same tree as a private
static-site artifact fallback. Local refreshes are not required for the hosted
site to update.

Primary dashboard entry points:

```text
reports/interactive/btc_market_dashboard.html
reports/interactive/btc_floor_weekly.html
reports/interactive/btc_roi_dashboard.html
reports/interactive/btc_gold_rotation_dashboard.html
reports/interactive/metals_relative_dashboard.html
reports/interactive/pipeline_health_dashboard.html
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
uv run scripts/build_btc_gold_dashboard.py
uv run scripts/build_pipeline_health_dashboard.py
```

Package the generated reports and dashboards into `dist/site/`:

```powershell
uv run scripts/build_static_site.py
```

The packaged site includes `index.html`, `reports/interactive/`,
`reports/figures/`, and top-level generated report files. It intentionally
excludes `data/raw/` and `data/processed/`.

The GitHub Pages project URL is:

```text
https://arcamz.github.io/btcfloor/
```

The repository must have Pages configured with Source set to GitHub Actions.

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
monitoring. LBMA fixes are kept only for long-history analog context and are
read from tracked snapshots in `resources/legacy/` during scheduled refreshes.
This keeps the analog panels available even when the LBMA API is unavailable.
To refresh those snapshots manually, set `BTCFLOOR_REFRESH_LBMA=1` before
running `uv run scripts/build_metals_dashboard.py`, then review and commit the
updated snapshot CSVs. In the metals analog charts, dashed 2026 LBMA lines are
static context; solid 2026 `GC=F` and `SI=F` lines are the live source-of-truth
price paths.

BTC/gold rotation uses processed BTC daily closes and Yahoo Finance COMEX gold
futures (`GC=F`) on the latest shared trading date. The dashboard does not
fabricate weekend gold closes.

## Price fixes

Manual BTC data fixes can be recorded in `config/price_fixes.csv` with columns:

```csv
date,action,price_usd,reason
```

Supported actions are `replace` and `drop`.
