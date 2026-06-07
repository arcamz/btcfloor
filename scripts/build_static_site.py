from __future__ import annotations

import html
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from btcfloor.dashboard_common import DASHBOARD_LINKS
from btcfloor.paths import ProjectPaths


TOP_LEVEL_REPORT_EXTENSIONS = {".csv", ".json", ".md"}
PRIMARY_DASHBOARD_NAMES = tuple(link.href for link in DASHBOARD_LINKS)


def _site_root(paths: ProjectPaths, output_root: Path | None = None) -> Path:
    return (output_root or paths.root / "dist" / "site").resolve()


def _require_source_tree(paths: ProjectPaths) -> None:
    missing = [
        path
        for path in (paths.report_dir, paths.interactive_dir, paths.figure_dir)
        if not path.exists()
    ]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Generated report tree is incomplete. Run "
            "`uv run scripts/update_daily.py` before packaging. Missing: "
            f"{joined}"
        )

    missing_dashboards = [
        name for name in PRIMARY_DASHBOARD_NAMES if not (paths.interactive_dir / name).exists()
    ]
    if missing_dashboards:
        joined = ", ".join(missing_dashboards)
        raise FileNotFoundError(f"Missing primary dashboard HTML files: {joined}")


def _generated_at(paths: ProjectPaths) -> str:
    health_path = paths.report_dir / "pipeline_health.json"
    if health_path.exists():
        try:
            payload = json.loads(health_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        generated_at = payload.get("generated_at")
        if generated_at:
            return str(generated_at)
    return datetime.now(UTC).isoformat(timespec="seconds")


def _copy_directory(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing source directory: {source}")
    shutil.copytree(source, destination)


def _copy_top_level_reports(report_dir: Path, destination_dir: Path) -> list[str]:
    copied: list[str] = []
    for source in sorted(report_dir.iterdir()):
        if source.is_dir() or source.suffix.lower() not in TOP_LEVEL_REPORT_EXTENSIONS:
            continue
        shutil.copy2(source, destination_dir / source.name)
        copied.append(source.name)
    return copied


def _dashboard_links_html() -> str:
    cards = []
    for link in DASHBOARD_LINKS:
        href = f"reports/interactive/{link.href}"
        cards.append(
            f"""
        <a class="card" href="{html.escape(href)}">
          <span class="card-title">{html.escape(link.label)}</span>
          <span class="card-path">{html.escape(link.href)}</span>
        </a>"""
        )
    return "\n".join(cards)


def _report_links_html(report_files: list[str]) -> str:
    if not report_files:
        return '<p class="empty">No top-level report files were packaged.</p>'
    return "\n".join(
        f'<a class="report-link" href="{html.escape(f"reports/{name}")}">{html.escape(name)}</a>'
        for name in report_files
    )


def _index_html(generated_at: str, report_files: list[str]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>btcfloor Private Dashboard Artifact</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5b6472;
      --line: #d9dee7;
      --band: #f7f9fc;
      --accent: #155e75;
    }}
    body {{
      margin: 0;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .lead {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.5;
      max-width: 880px;
    }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 24px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      background: white;
      color: var(--muted);
      font-size: 13px;
      padding: 8px 12px;
    }}
    section {{
      margin: 0 0 28px;
    }}
    .dashboard-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .report-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .card, .report-link {{
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      color: inherit;
      text-decoration: none;
      padding: 14px 15px;
    }}
    .card:hover, .report-link:hover {{
      border-color: var(--accent);
      box-shadow: 0 2px 12px rgba(21, 94, 117, 0.08);
    }}
    .card-title {{
      display: block;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .card-path {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .note {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      color: var(--muted);
      line-height: 1.5;
      padding: 14px 15px;
    }}
    .empty {{
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      main {{
        padding: 18px;
      }}
      .dashboard-grid, .report-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>btcfloor private dashboard artifact</h1>
    <p class="lead">
      This package was generated by GitHub Actions. The workflow refreshed the
      data, rebuilt the reports and dashboards, then packaged the static output
      for private download from the Actions run.
    </p>
    <div class="meta">
      <span class="pill">Generated at {html.escape(generated_at)}</span>
      <span class="pill">Private Actions artifact</span>
      <span class="pill">Scheduled every 4 hours</span>
    </div>
    <section>
      <h2>Primary dashboards</h2>
      <div class="dashboard-grid">
{_dashboard_links_html()}
      </div>
    </section>
    <section>
      <h2>Top-level reports</h2>
      <div class="report-grid">
        {_report_links_html(report_files)}
      </div>
    </section>
    <section>
      <div class="note">
        Open this file after downloading and unpacking the artifact. The
        dashboards keep their existing relative links to <code>reports/figures</code>.
        Public hosting is intentionally not enabled in this version.
      </div>
    </section>
  </main>
</body>
</html>
"""


def build_static_site(paths: ProjectPaths, output_root: Path | None = None) -> Path:
    _require_source_tree(paths)

    site_root = _site_root(paths, output_root)
    if site_root.exists():
        shutil.rmtree(site_root)
    site_root.parent.mkdir(parents=True, exist_ok=True)

    site_reports_dir = site_root / "reports"
    site_reports_dir.mkdir(parents=True, exist_ok=True)

    _copy_directory(paths.interactive_dir, site_reports_dir / "interactive")
    _copy_directory(paths.figure_dir, site_reports_dir / "figures")
    report_files = _copy_top_level_reports(paths.report_dir, site_reports_dir)

    (site_root / ".nojekyll").write_text("", encoding="utf-8")
    (site_root / "index.html").write_text(
        _index_html(_generated_at(paths), report_files),
        encoding="utf-8",
    )
    return site_root


def main() -> None:
    print(build_static_site(ProjectPaths.from_cwd()))


if __name__ == "__main__":
    main()
