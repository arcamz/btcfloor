# Methodology

This project compares several ways to estimate BTC floor pressure.

## Giovanni fixed power-law floor

The fixed floor is:

```text
trend = 1.0117e-17 * days_since_genesis^5.82
floor = trend * 0.42
```

This model is not fit to the local dataset, so it is useful as a fixed lower
rail. Its main weakness is that it cannot adapt if future BTC market structure
permanently changes.

## Burger 2019 RANSAC support

The Burger support reference is a pre-2022 provenance check based on Harold
Christopher Burger's September 2019 power-law corridor work. The current
snapshot always fits on the same vintage window:

```text
2010-07-17 <= date <= 2019-09-03
```

The approximation uses the article's robust-fit idea:

```text
y = log10(price_usd)
x = log10(days_since_genesis)
iteratively remove the largest absolute residual until 50% of rows remain
fit y = a + b*x on retained rows
floor = fitted line - 1.5 * retained-residual sigma
```

This line is not treated as an exact replication of Burger's chart. It is a
reproducible, same-window RANSAC-style lower support reference that cannot be
moved by the 2022 or 2026 lows.

## Weekly expectile floor

The LuxAlgo-inspired approximation fits:

```text
log10(price) = intercept + slope * log10(days_since_genesis)
```

with asymmetric least squares expectile regression on weekly closes. The bottom
expectile used in the main report is tau `0.0001`, displayed as `0.01%`.

## Data freshness

Coin Metrics is the canonical long-history source. When Coin Metrics lags the
current date, the pipeline appends only missing recent BTC/USD daily rows from
CoinGecko's public market-chart range endpoint. The processed series reports
source row counts in `reports/data_quality.md`.

Daily rows are UTC-dated. If a local timezone has moved into a new calendar day
before UTC has, the latest processed daily row can still be the prior local
date. The analysis should state the latest processed date rather than inventing
a partial daily close.

## Forward floor overlap

The forward-floor signal asks whether today's spot price is below a floor value
projected at a future horizon, such as 12 months. This is different from a
same-day floor touch:

```text
spot_today < floor(today + horizon)
```

The signal is intended to identify periods where current price overlaps the
future bottom rail before the expected cycle low.

## Tactical interpretation layer

The floor models answer whether price is in a historically interesting value
zone. They do not confirm that downside momentum has ended. Tactical reads
should therefore separate:

- floor and expectile pressure,
- 50D/200D SMA state,
- post-breach channel position,
- support/resistance reclaim,
- sweep/failure pattern evidence.

A strong floor signal with price below key moving averages is best treated as a
staged value/failed-breakdown setup, not as confirmed trend continuation.

## Checkonchain cohort layer

The on-chain cohort layer is pulled from Checkonchain's public static Plotly
charts during `uv run scripts/update_daily.py`. The core cohort metrics are:

- STH-MVRV and STH-MVRV Z-score: short-holder cost-basis stress.
- STH price-equivalent Z-score bands: tactical stress zones around current spot.
- LTH-MVRV and LTH-SOPR: long-holder unrealised and realised stress.
- LTH realised loss in BTC, including 7D/28D EMAs: capitulation impulse.
- Cointime Price: Checkonchain's cointime pricing model, added to the weekly
  realised-price stress map as another low-zone cost-basis reference.
- Classic CVDD: optional Bitbo API overlay when `BITBO_API_KEY` is configured;
  otherwise fetched from Looknode's public classic-formula CVDD endpoint and
  labelled as a third-party fallback. BGeometrics CVDD is intentionally not
  used here because its documented current-supply normalization puts it on a
  different tactical scale.

These metrics do not define the power-law floor. They are used to decide
whether the current floor-overlap zone is only "good value" or also shows the
kind of long-holder capitulation often seen near final lows.

## Metals relative-strength layer

The metals layer is a separate swing-allocation context. It uses Yahoo Finance
COMEX futures (`GC=F`, `SI=F`) for live gold/silver decision monitoring and
calculates daily GSR:

```text
GSR = gold_price_usd_per_oz / silver_price_usd_per_oz
```

GSR is used as the primary gold-vs-silver switch measure because it directly
answers whether silver is outperforming gold. The monitored levels are:

- 60.0: initial silver rotation trigger.
- 58.5: silver leadership confirmation.
- 56.0: first silver outperformance target.
- 53.0: strong silver outperformance target.
- 48.0: aggressive silver catch-up target.

Legacy LBMA series are retained only for long-history analog context. Raw gold
and silver analog returns are useful context, but rotation should be based on
live GSR breakdown plus silver price confirmation.

## BTC/gold rotation layer

The BTC/gold dashboard answers a different allocation question:

```text
BTC/XAU = BTC_USD_close / gold_USD_close
```

This is the number of gold ounces one BTC buys. It is calculated on the latest
shared BTC/gold trading date using processed BTC daily closes and Yahoo Finance
COMEX gold futures (`GC=F`). Weekend gold closes are not fabricated.

The intended read is:

- Daily 20D reclaim: tactical BTC probe versus gold.
- Daily 50D plus weekly 20W reclaim: stronger rotation evidence.
- Daily 200D or weekly 50W reclaim: broader BTC/gold regime repair.

This ratio does not replace the BTC floor model. It answers whether BTC is
starting to outperform a gold parking position while BTC floor pressure is
already elevated.

## Interactive dashboards

The main update command writes six HTML artifacts:

```text
reports/interactive/btc_floor_weekly.html
reports/interactive/btc_market_dashboard.html
reports/interactive/btc_roi_dashboard.html
reports/interactive/btc_gold_rotation_dashboard.html
reports/interactive/metals_relative_dashboard.html
reports/interactive/pipeline_health_dashboard.html
```

The weekly chart is the original floor/cycle Plotly view and now includes the
200-day SMA. The market dashboard combines the price/floor/SMA view with
zoomable Checkonchain STH/LTH panels. The ROI dashboard compares 1.0x and 1.5x
exposure tables and includes a deployment decision matrix.

The BTC/gold dashboard monitors relative BTC leadership versus gold. The metals
dashboard monitors live COMEX futures GSR for decisions, plus legacy LBMA
gold/silver analog panels for historical context.

The pipeline health dashboard reports source freshness, fallback status,
required artifact freshness, and missing/stale outputs. It is an operations
view, not an additional market model.

Each dashboard includes regenerated agent commentary from the latest report
data. Commentary should be treated as a quantitative status read, not as a
separate model.

## Cycle timing

Cycle timing is seeded from a common low-to-peak and peak-to-low cadence:

- 1064 days from low to peak,
- 364 days from peak to the next low,
- 1428 days from low to low.

These are research assumptions, not guarantees.

## Stability checks

Stability reports refit adaptive models after excluding recent market cycles
and evaluate how the resulting floor behaves afterward. This helps distinguish
fixed model behavior from adaptive models that may chase recent lows.
