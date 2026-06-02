# Usage

`btcfloor` is a command-line research toolkit. Commands write generated data
and reports under `data/` and `reports/`; both directories are ignored by git.

## Download data

```powershell
uv run btcfloor download
```

The long-history source is the Coin Metrics community BTC CSV archive. If that
file lags the current date, missing recent daily BTC/USD rows are appended from
CoinGecko's public `market_chart/range` endpoint. The raw CSV is cached under
`data/raw/` and normalized daily prices are written under `data/processed/`.
Daily rows are UTC-dated, so local dates shortly after midnight can be ahead of
the latest processed daily row.

## Run the full analysis

```powershell
uv run btcfloor analyze
```

This creates:

- model snapshots,
- forward floor-distance tables,
- role-based risk tables,
- cycle-low validation,
- walk-forward validation,
- stability checks,
- static diagnostic figures,
- a standalone interactive Plotly chart.

## Generate only the chart

```powershell
uv run btcfloor chart
```

The chart is written to `reports/interactive/btc_floor_weekly.html`.

## Price fixes

Manual data fixes can be recorded in `config/price_fixes.csv` with columns:

```csv
date,action,price_usd,reason
```

Supported actions are `replace` and `drop`.
