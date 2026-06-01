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

## Weekly expectile floor

The LuxAlgo-inspired approximation fits:

```text
log10(price) = intercept + slope * log10(days_since_genesis)
```

with asymmetric least squares expectile regression on weekly closes. The bottom
expectile used in the main report is tau `0.0001`, displayed as `0.01%`.

## Forward floor overlap

The forward-floor signal asks whether today's spot price is below a floor value
projected at a future horizon, such as 12 months. This is different from a
same-day floor touch:

```text
spot_today < floor(today + horizon)
```

The signal is intended to identify periods where current price overlaps the
future bottom rail before the expected cycle low.

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

