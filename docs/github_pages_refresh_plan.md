# 4-Hour Dashboard Refresh With GitHub Pages And Artifact Fallback

## Summary

Implement a GitHub Actions pipeline that refreshes BTC, metals, BTC/gold, and
on-chain data every 4 hours inside GitHub Actions, rebuilds all generated
reports and dashboards, packages them into `dist/site/`, publishes that tree to
GitHub Pages, and uploads the same tree as a private workflow artifact fallback.

The local machine is not part of the production refresh path. Local commands
exist only for manual reproduction and debugging. The only local step after
implementation is normal git sync.

## Key Changes

- Add `scripts/build_static_site.py` to package generated output into
  `dist/site/`.
- Add `.github/workflows/refresh-site.yml` for scheduled, manual, and
  post-push refreshes plus GitHub Pages deployment.
- Keep `data/`, `reports/`, and `dist/` ignored and uncommitted.
- Update `README.md`, `docs/usage.md`, and `AGENTS.md` to document the
  Actions-owned refresh path and the local reproduction commands.

## Workflow Behavior

- Trigger on:
  - `workflow_dispatch`,
  - `schedule: '17 */4 * * *'`,
  - `push` to `main`.
- Run on GitHub-hosted Ubuntu:
  - checkout,
  - setup `uv`,
  - install Python 3.12,
  - `uv sync --all-groups`,
  - `uv run pytest -q`,
  - `uv run scripts/update_daily.py`,
  - `uv run scripts/build_static_site.py`,
  - upload `dist/site` with `actions/upload-artifact`,
  - upload `dist/site` with `actions/upload-pages-artifact`,
  - deploy the Pages artifact with `actions/deploy-pages`.
- GitHub Pages is the primary browsable output:
  - `https://arcamz.github.io/btcfloor/`
- The repository must have Pages configured with Source set to GitHub Actions.
- `BITBO_API_KEY` is optional. If it is missing, the existing Looknode CVDD
  fallback remains expected and is labelled by the dashboards and health report.

## Static Artifact Contract

`dist/site/` must include:

- `.nojekyll`
- `index.html`
- `reports/interactive/*.html`
- `reports/figures/*`
- top-level generated `reports/*.csv`
- top-level generated `reports/*.json`
- top-level generated `reports/*.md`

`dist/site/` must exclude:

- `data/raw`
- `data/processed`
- `.env`
- local caches
- virtual environments
- secrets

## Test And Acceptance Plan

- Add tests for `scripts/build_static_site.py`:
  - creates `dist/site/index.html`,
  - writes `.nojekyll`,
  - copies all primary dashboard HTML files,
  - copies figure files,
  - copies top-level CSV/JSON/MD reports,
  - excludes raw and processed data,
  - rebuilds cleanly over an existing `dist/site`.
- Run:
  - `uv run pytest -q`
  - `uv run python -m compileall src scripts`
- Manual local acceptance:
  - `uv run scripts/update_daily.py`
  - `uv run scripts/build_static_site.py`
  - serve `dist/site`
  - verify dashboard nav and image links
- GitHub acceptance:
  - manual workflow run succeeds,
  - artifact is downloadable,
  - Pages deploy job succeeds,
  - Pages URL opens `index.html`,
  - unpacked artifact opens at `index.html`,
  - pipeline health dashboard shows the latest workflow-generated timestamp.

## Explicit Assumptions

- Production refresh happens inside GitHub Actions, not locally.
- The user does not need to run `update_daily.py` locally for the hosted site
  or artifact to update.
- GitHub Pages output is public unless the GitHub account/organization supports
  private Pages access control.
- The private artifact remains available as a fallback for non-public access.
- Cloudflare is out of scope for this step, but `dist/site/` remains reusable
  later for Cloudflare Pages or Direct Upload.
- Generated outputs remain ignored and should not be committed.
- Existing dashboard relative links must keep working in the artifact.
- GitHub's scheduler may delay runs, so "every 4 hours" means scheduled every
  4 hours, not guaranteed exact wall-clock execution.
