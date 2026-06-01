# Model Notes

## Giovanni Power-Law Floor

The fixed Giovanni floor implementation uses:

```text
trend = 1.0117e-17 * days_since_genesis^5.82
floor = trend * 0.42
```

`days_since_genesis` is measured from the Bitcoin genesis block date, `2009-01-03`.

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
- `reports/interactive/btc_floor_weekly.html`: weekly candle chart with floor overlays.
