# Contributing

Contributions are welcome if they keep the project reproducible and explicit.

## Development

This project uses `uv` for dependency management and Python execution.

```powershell
uv venv
uv sync --all-groups
uv run pytest -q
```

Run analysis commands through the package CLI:

```powershell
uv run btcfloor download
uv run btcfloor analyze
uv run btcfloor chart
```

Generated `data/` and `reports/` outputs are intentionally ignored by git.
If a contribution changes report behavior, include tests for the underlying
calculation rather than committing regenerated market data or HTML outputs.

## Pull request checklist

- Add or update tests for changed model logic.
- Keep generated artifacts out of commits.
- Document new assumptions in `MODEL_NOTES.md`.
- Avoid financial advice language; this is a research toolkit.

