# Agent Instructions

- Use Python for project scripts, tools, and application code unless a task clearly requires another language.
- Use `uv` for all Python dependency management and Python execution.
- Prefer `uv add` / `uv remove` for dependency changes instead of editing dependency metadata by hand.
- Run Python commands through `uv run` so they use the project-managed environment.
- If a uv environment does not exist, create one with `uv venv` before installing or running Python dependencies.
