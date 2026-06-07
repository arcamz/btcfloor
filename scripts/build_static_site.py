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
    payload = _health_payload(paths)
    generated_at = payload.get("generated_at")
    if generated_at:
        return str(generated_at)
    return datetime.now(UTC).isoformat(timespec="seconds")


def _health_payload(paths: ProjectPaths) -> dict:
    health_path = paths.report_dir / "pipeline_health.json"
    if not health_path.exists():
        return {}
    try:
        payload = json.loads(health_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _status_badge(status: str) -> str:
    normalized = status.lower()
    labels = {
        "ok": "OK",
        "warn": "WARN",
        "stale": "STALE",
        "missing": "MISSING",
    }
    return (
        f'<span class="status-badge {html.escape(normalized)}">'
        f"{html.escape(labels.get(normalized, normalized.upper()))}</span>"
    )


def _health_counts_html(health: dict) -> str:
    counts = health.get("status_counts") or {}
    parts = [
        ("OK", counts.get("ok", 0), "ok"),
        ("Warn", counts.get("warn", 0), "warn"),
        ("Stale", counts.get("stale", 0), "stale"),
        ("Missing", counts.get("missing", 0), "missing"),
    ]
    return "\n".join(
        f'<span class="health-count {css_class}">{label}: {int(value or 0)}</span>'
        for label, value, css_class in parts
    )


def _public_path(path_value: object) -> str:
    if not path_value:
        return ""
    try:
        path = Path(str(path_value))
    except (TypeError, ValueError):
        return str(path_value)

    parts = list(path.parts)
    if "reports" in parts:
        return "/".join(parts[parts.index("reports") :])
    return path.name


def _health_rows_html(health: dict) -> str:
    items = health.get("items")
    if not isinstance(items, list) or not items:
        return """
              <tr>
                <td colspan="6" class="empty">No pipeline health payload was packaged.</td>
              </tr>"""

    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            f"""
              <tr>
                <td>{_status_badge(str(item.get("status", "")))}</td>
                <td>{html.escape(str(item.get("category") or ""))}</td>
                <td>{html.escape(str(item.get("name") or ""))}</td>
                <td>{html.escape(str(item.get("date") or ""))}</td>
                <td>{html.escape(str(item.get("source") or ""))}</td>
                <td>{html.escape(_public_path(item.get("path")))}</td>
              </tr>"""
        )
    return "\n".join(rows)


def _health_section_html(health: dict) -> str:
    if not health:
        return """
    <section>
      <h2>Data health</h2>
      <div class="note">
        No pipeline health payload was packaged. Open the Data Health dashboard
        after the next successful refresh.
      </div>
    </section>"""

    overall = str(health.get("overall_status") or "unknown")
    generated_at = str(health.get("generated_at") or "")
    today_utc = str(health.get("today_utc") or "")
    return f"""
    <section>
      <div class="section-heading">
        <h2>Data health</h2>
        <a class="section-link" href="reports/interactive/pipeline_health_dashboard.html">Open full health dashboard</a>
      </div>
      <div class="health-summary">
        <span>Overall {_status_badge(overall)}</span>
        <span>Generated {html.escape(generated_at)}</span>
        <span>UTC date {html.escape(today_utc)}</span>
        {_health_counts_html(health)}
      </div>
      <div class="table-wrap">
        <table class="health-table">
          <thead>
            <tr>
              <th>Status</th>
              <th>Category</th>
              <th>Name</th>
              <th>Latest</th>
              <th>Source</th>
              <th>Packaged path</th>
            </tr>
          </thead>
          <tbody>
{_health_rows_html(health)}
          </tbody>
        </table>
      </div>
    </section>"""


def _index_html(generated_at: str, report_files: list[str], health: dict) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>btcfloor Dashboard Site</title>
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
    .section-heading {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 12px;
    }}
    .section-link {{
      color: var(--accent);
      font-weight: 650;
      text-decoration: none;
      border-bottom: 1px solid var(--accent);
      white-space: nowrap;
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
    .health-summary {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 12px 14px;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .status-badge, .health-count {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 750;
      letter-spacing: 0;
    }}
    .status-badge.ok, .health-count.ok {{
      color: #14532d;
      background: #dcfce7;
    }}
    .status-badge.warn, .health-count.warn {{
      color: #854d0e;
      background: #fef3c7;
    }}
    .status-badge.stale, .status-badge.missing, .health-count.stale, .health-count.missing {{
      color: #7f1d1d;
      background: #fee2e2;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }}
    .health-table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 920px;
      font-size: 13px;
    }}
    .health-table th, .health-table td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .health-table th {{
      color: var(--muted);
      font-weight: 750;
      background: var(--band);
    }}
    .health-table td {{
      color: var(--ink);
    }}
    .health-table tr:last-child td {{
      border-bottom: 0;
    }}
    @media (max-width: 900px) {{
      main {{
        padding: 18px;
      }}
      .dashboard-grid, .report-grid {{
        grid-template-columns: 1fr;
      }}
      .section-heading {{
        display: block;
      }}
      .section-link {{
        display: inline-block;
        margin-top: 6px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>btcfloor dashboard site</h1>
    <p class="lead">
      This site was generated by GitHub Actions. The workflow refreshed the
      data, rebuilt the reports and dashboards, then published the static
      output to GitHub Pages with a private Actions artifact fallback.
    </p>
    <div class="meta">
      <span class="pill">Generated at {html.escape(generated_at)}</span>
      <span class="pill">GitHub Pages</span>
      <span class="pill">Private artifact fallback</span>
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
{_health_section_html(health)}
    <section>
      <div class="note">
        The dashboards keep their existing relative links to
        <code>reports/figures</code>. If GitHub Pages is unavailable, download
        the private workflow artifact and open <code>index.html</code> from the
        unpacked artifact root.
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
        _index_html(_generated_at(paths), report_files, _health_payload(paths)),
        encoding="utf-8",
    )
    return site_root


def main() -> None:
    print(build_static_site(ProjectPaths.from_cwd()))


if __name__ == "__main__":
    main()
