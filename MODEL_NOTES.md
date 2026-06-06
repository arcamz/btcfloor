# Model Notes

## Giovanni Power-Law Floor

The fixed Giovanni floor implementation uses:

```text
trend = 1.0117e-17 * days_since_genesis^5.82
floor = trend * 0.42
```

`days_since_genesis` is measured from the Bitcoin genesis block date, `2009-01-03`.

## Burger 2019 RANSAC Support Reference

Harold Christopher Burger's September 2019 power-law corridor article gives a
pre-2022 support-line methodology: fit BTC price in log-log space, then derive
lower support and robust normal-mode references from the same historical
window.

This project implements a transparent approximation:

```text
source window = 2010-07-17 through 2019-09-03
y = log10(price_usd)
x = log10(days_since_genesis)
iteratively remove the largest absolute residual until 50% of rows remain
fit y = a + b*x on the retained rows
floor = fitted line - 1.5 * retained-residual sigma
```

The model is meant as a pre-2022 provenance support reference. It is not a
claim that Burger published this exact sigma-band formula.

Recent Bitcoin Observatory papers are relevant for interpretation, but they are
post-2022 sources. They argue that floor definitions vary and that the `0.42x`
floor can have daily breaches, so this project treats floor lines as evidence
bands rather than unbreakable guarantees.

## LuxAlgo-Style Bottom Expectile

The public LuxAlgo Bitcoin Expectile Model page describes the model as:

- Multiple expectile log-log regressions.
- Designed for the Bitcoin all-time-history index on the weekly chart, also recommended on 3-day.
- Logarithmic chart scale.
- Tau values shown to users as `tau * 100`.
- Iteratively reweighted least squares.
- A default fit start date around `2010-07-16`.
- Optional correction for the missing genesis-block history.

This project implements the bottom floor as a transparent Python approximation:

```text
y = log10(price_usd)
x = log10(days_since_genesis)
fit y = a + b*x with asymmetric least squares
bottom tau = 0.0001, displayed as 0.01%
```

The current data source is Coin Metrics community BTC daily data. Its first valid positive `PriceUSD` row is `2010-07-18`, so the fit effectively starts two days after LuxAlgo's stated default date unless manual price fixes are configured.

The project does not currently copy or vendor TradingView Pine Script source. The implementation follows the public model description and keeps generated sensitivity outputs in `reports/expectile_sensitivity.csv`.

## Current Validation Surfaces

- `reports/model_snapshot.csv`: current floor estimates.
- `reports/risk_horizons.csv`: per-model forward floor distances.
- `reports/risk_ensemble.csv`: min/median/max floor ensemble and time-adjusted bottom pressure.
- `reports/expectile_sensitivity.csv`: bottom expectile sensitivity around the requested 0.01% floor.
- `reports/cycle_low_validation.csv`: historical low-window behavior.
- `reports/stability.csv`: refits excluding recent cycles.
- `reports/interactive/btc_floor_weekly.html`: weekly candle chart with floor overlays and 200-day SMA.
- `reports/interactive/btc_market_dashboard.html`: Plotly market/on-chain dashboard with zoomable floor, SMA, STH, LTH, and realised-loss views.
- `reports/interactive/btc_roi_dashboard.html`: Plotly ROI and deployment dashboard with 1.0x/1.5x exposure switches, downside/recovery geometry, and decision matrix.
- `reports/interactive/btc_gold_rotation_dashboard.html`: Plotly BTC/XAU rotation dashboard with daily/weekly moving-average confirmation versus gold.
- `reports/interactive/metals_relative_dashboard.html`: Plotly metals dashboard with live COMEX futures GSR rotation levels and legacy long-history analog context.
- `reports/interactive/pipeline_health_dashboard.html`: data-source and generated-artifact health dashboard.

## Checkonchain Cohort Metrics

The project parses Checkonchain's public static Plotly chart pages for cohort
context. The generated dataset is:

```text
data/processed/checkonchain_cohort_metrics.csv
```

Current fields used in dashboards include:

- STH realised price, STH-MVRV, and STH-MVRV Z-score.
- STH price-equivalent Z-score bands, especially -1.0sd, -1.5sd, and -2.0sd.
- LTH realised price, LTH true realised price, LTH-MVRV, and LTH-SOPR.
- Cointime Price from Checkonchain's cointime pricing chart.
- LTH realised loss in BTC plus 7D and 28D EMA variants.
- Optional CVDD from Bitbo's API when `BITBO_API_KEY` is configured.
- Public Looknode classic-formula CVDD fallback when Bitbo access is not
  configured; dashboards should label this as a third-party fallback rather
  than canonical Bitbo CVDD.

These metrics are strategy context, not floor-model inputs. They help separate
"value zone" from "final capitulation" evidence.

## Metals Relative Strength

The metals dashboard uses Yahoo Finance COMEX futures (`GC=F`, `SI=F`) for
live gold/silver and GSR decision monitoring. It also rebuilds the local gold
and silver analog studies inside the project from legacy long-history LBMA
series. It writes:

```text
data/processed/yahoo_gold_futures.csv
data/processed/yahoo_silver_futures.csv
data/processed/metals_gsr_daily.csv
reports/metals_relative_summary.json
```

GSR is the primary gold-vs-silver switch variable. The dashboard marks:

- 60.0: initial silver rotation trigger.
- 58.5: silver leadership confirmation.
- 56.0, 53.0, 48.0: silver outperformance target / reassessment zones.

The gold and silver analog charts are context. LBMA is not the preferred live
decision source. The allocation signal should be read through live GSR plus
silver price confirmation, not raw silver upside alone.

## BTC/Gold Rotation

The BTC/gold dashboard uses processed BTC daily closes and Yahoo Finance COMEX
gold futures (`GC=F`) to calculate:

```text
BTC/XAU = BTC_USD / gold_USD_per_oz
```

It writes:

```text
data/processed/btc_gold_ratio_daily.csv
data/processed/btc_gold_ratio_weekly.csv
reports/btc_gold_rotation_summary.json
```

The ratio is evaluated on the latest shared BTC/gold trading date. Daily 20D is
treated as a tactical probe, daily 50D plus weekly 20W as stronger rotation
evidence, and daily 200D or weekly 50W as broader BTC/gold regime repair.

## Dashboard Commentary

Interactive dashboard commentary is generated from current CSV/JSON outputs at
update time. It should stay quantitative: current spot, hard/warning floors,
forward-floor distance, SMA/channel state, Checkonchain STH/LTH stress, and the
resulting deployment interpretation.

The pipeline health dashboard is intentionally separate from commentary. It
answers whether the data sources, fallbacks, and generated artifacts are fresh
enough to trust before reading the market dashboards.
