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
- LuxAlgo-inspired weekly expectile power-law approximation, including the
  requested `0.01%` bottom expectile.
- Forward floor-distance checks for 1, 3, 6, 9, and 12 month horizons.
- Historical forward-floor overlap episodes.
- 4chan-style cycle timing table using 1064 low-to-peak days and 364
  peak-to-low days.
- Full-sample and walk-forward cycle-low validation.
- Stability checks that refit adaptive models after excluding recent cycles.
- Static diagnostic plots and a standalone interactive Plotly weekly chart.
- `uv`-managed package, CLI, tests, docs, CI workflow, and git hygiene.

## Quick Start

Install dependencies and run tests:

```powershell
uv venv
uv sync --all-groups
uv run pytest -q
```

Run the full analysis:

```powershell
uv run btcfloor analyze
```

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

These files are reproducible outputs, not source files.

## Methodology

See [docs/methodology.md](docs/methodology.md) and [MODEL_NOTES.md](MODEL_NOTES.md)
for formulas, assumptions, and model limitations.

Short version:

- The Giovanni floor is fixed and does not refit to recent market data.
- The expectile floor is a transparent Python asymmetric least-squares fit over
  weekly BTC closes.
- The forward-floor signal asks whether current price overlaps a future floor:
  `spot_today < floor(today + horizon)`.
- Cycle timing is an explicit research assumption, not a forecast guarantee.

## Data Source

The default data source is the Coin Metrics community BTC CSV archive:

```text
https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv
```

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
