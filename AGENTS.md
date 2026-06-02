# Agent Instructions

- Use Python for project scripts, tools, and application code unless a task clearly requires another language.
- Use `uv` for all Python dependency management and Python execution.
- Prefer `uv add` / `uv remove` for dependency changes instead of editing dependency metadata by hand.
- Run Python commands through `uv run` so they use the project-managed environment.
- If a uv environment does not exist, create one with `uv venv` before installing or running Python dependencies.
- To refresh market data and reports, run `uv run btcfloor analyze --force-download`.
- The canonical long-history source is Coin Metrics. If Coin Metrics lags, the code appends only missing recent BTC/USD daily rows from CoinGecko.
- Treat processed daily bars as UTC-dated. If Europe/Stockholm has rolled into a new calendar day before UTC has, do not fabricate a new daily close; report the latest processed UTC date.
- Regenerate the tactical SMA/channel image with `uv run scripts/plot_sma_channel_decision.py` after refreshing analysis data.
- Read `reports/current_bottom_summary.csv`, `reports/risk_role_based.csv`, `reports/forward_floor_overlap_episodes.csv`, and `reports/sma_channel_decision_metrics.csv` before giving market interpretation.
- Interpret the setup as two separate layers: floor/expectile pressure for value context, and SMA/channel/reclaim/SFP behavior for tactical timing.
- Generated `data/`, `reports/`, and `dist/` artifacts are intentionally ignored and should not be committed.
