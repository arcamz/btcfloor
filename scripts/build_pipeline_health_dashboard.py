from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from btcfloor.dashboard_common import dashboard_nav, dashboard_nav_css
from btcfloor.paths import ProjectPaths


REQUIRED_REPORTS = [
    "data_quality.md",
    "current_bottom_summary.csv",
    "risk_role_based.csv",
    "forward_floor_overlap_episodes.csv",
    "sma_channel_decision_metrics.csv",
    "checkonchain_cohort_summary.json",
    "metals_relative_summary.json",
    "btc_gold_rotation_summary.json",
]

REQUIRED_INTERACTIVE = [
    "btc_floor_weekly.html",
    "btc_market_dashboard.html",
    "btc_roi_dashboard.html",
    "btc_gold_rotation_dashboard.html",
    "metals_relative_dashboard.html",
    "pipeline_health_dashboard.html",
]

REQUIRED_FIGURES = [
    "floor_convergence_decision_dashboard.png",
    "sma_channel_decision_plot.png",
    "tactical_trigger_strip.png",
    "checkonchain_cohort_current_bands.png",
    "checkonchain_cohort_cycle_lows.png",
    "checkonchain_lth_realised_loss_cycle.png",
    "checkonchain_low_signal_compare.png",
]


def _date(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).tz_localize(None).normalize()


def _today_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC)).tz_localize(None).normalize()


def _status_for_lag_days(lag_days: int | None, ok_days: int, warn_days: int) -> str:
    if lag_days is None:
        return "missing"
    if lag_days <= ok_days:
        return "ok"
    if lag_days <= warn_days:
        return "warn"
    return "stale"


def _mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _file_age_hours(path: Path) -> float | None:
    if not path.exists():
        return None
    modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    now = datetime.now().astimezone()
    return max((now - modified).total_seconds() / 3600.0, 0.0)


def _artifact_status(path: Path, max_ok_hours: float = 36.0, max_warn_hours: float = 96.0) -> str:
    age = _file_age_hours(path)
    if age is None:
        return "missing"
    if age <= max_ok_hours:
        return "ok"
    if age <= max_warn_hours:
        return "warn"
    return "stale"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _add_item(
    items: list[dict[str, Any]],
    category: str,
    name: str,
    status: str,
    detail: str,
    *,
    source: str | None = None,
    date: str | None = None,
    path: Path | None = None,
) -> None:
    items.append(
        {
            "category": category,
            "name": name,
            "status": status,
            "detail": detail,
            "source": source,
            "date": date,
            "path": str(path) if path is not None else None,
            "modified_at": _mtime(path) if path is not None else None,
        }
    )


def _btc_items(paths: ProjectPaths, items: list[dict[str, Any]], today: pd.Timestamp) -> None:
    path = paths.processed_btc_csv
    if not path.exists():
        _add_item(items, "Data sources", "BTC daily prices", "missing", "Processed BTC CSV is missing.", path=path)
        return

    data = pd.read_csv(path, parse_dates=["date"])
    latest = _date(data["date"].max())
    lag = None if latest is None else int((today - latest).days)
    status = _status_for_lag_days(lag, ok_days=1, warn_days=2)
    source_counts = (
        data.get("source", pd.Series(dtype=str))
        .fillna("unknown")
        .value_counts()
        .to_dict()
    )
    source_text = ", ".join(f"{key}: {value}" for key, value in source_counts.items())
    _add_item(
        items,
        "Data sources",
        "BTC daily prices",
        status,
        f"Latest UTC row {latest:%Y-%m-%d}; source rows {source_text}.",
        source="Coin Metrics + CoinGecko recent-fill",
        date=f"{latest:%Y-%m-%d}" if latest is not None else None,
        path=path,
    )


def _checkonchain_items(
    paths: ProjectPaths,
    items: list[dict[str, Any]],
    today: pd.Timestamp,
) -> None:
    summary_path = paths.report_dir / "checkonchain_cohort_summary.json"
    summary = _load_json(summary_path)
    latest = summary.get("latest", {})
    if not latest:
        _add_item(items, "Data sources", "Checkonchain cohorts", "missing", "Summary JSON is missing or empty.", path=summary_path)
        return

    metric_dates = [_date(value.get("date")) for value in latest.values() if isinstance(value, dict)]
    max_latest = max((date for date in metric_dates if date is not None), default=None)
    lag = None if max_latest is None else int((today - max_latest).days)
    status = _status_for_lag_days(lag, ok_days=1, warn_days=3)
    _add_item(
        items,
        "Data sources",
        "Checkonchain cohorts",
        status,
        f"{len(latest)} metrics; latest metric date {max_latest:%Y-%m-%d}.",
        source="Checkonchain static Plotly pages",
        date=f"{max_latest:%Y-%m-%d}" if max_latest is not None else None,
        path=summary_path,
    )

    cvdd = summary.get("source", {}).get("cvdd", {})
    bitbo = cvdd.get("bitbo", {})
    looknode = cvdd.get("looknode", {})
    if bitbo.get("available"):
        status = "ok"
        detail = "Bitbo CVDD is active."
        source = "Bitbo"
    elif looknode.get("available"):
        status = "warn"
        detail = "Using Looknode public CVDD fallback because Bitbo is unavailable."
        source = "Looknode fallback"
    else:
        status = "missing"
        detail = "No CVDD source is available."
        source = None
    cvdd_latest = latest.get("CVDD", {})
    _add_item(
        items,
        "Data sources",
        "CVDD source",
        status,
        detail,
        source=source,
        date=cvdd_latest.get("date") if isinstance(cvdd_latest, dict) else None,
        path=summary_path,
    )


def _metals_items(paths: ProjectPaths, items: list[dict[str, Any]], today: pd.Timestamp) -> None:
    summary_path = paths.report_dir / "metals_relative_summary.json"
    summary = _load_json(summary_path)
    latest = summary.get("latest", {})
    if not latest:
        _add_item(items, "Data sources", "Metals and GSR", "missing", "Metals summary JSON is missing or empty.", path=summary_path)
        return

    for key, label in [("gold", "Gold futures"), ("silver", "Silver futures"), ("gsr", "Gold/silver ratio")]:
        entry = latest.get(key, {})
        latest_date = _date(entry.get("date"))
        lag = None if latest_date is None else int((today - latest_date).days)
        status = _status_for_lag_days(lag, ok_days=3, warn_days=5)
        _add_item(
            items,
            "Data sources",
            label,
            status,
            "Weekend/market-close lag up to three calendar days is expected for futures feeds.",
            source=entry.get("source"),
            date=f"{latest_date:%Y-%m-%d}" if latest_date is not None else None,
            path=summary_path,
        )

    legacy = summary.get("sources", {}).get("legacy_analog_context", {})
    if legacy:
        _add_item(
            items,
            "Data sources",
            "LBMA legacy analog context",
            "ok" if legacy.get("available", True) else "warn",
            str(legacy.get("detail") or legacy.get("note") or "Legacy analog context status."),
            source="LBMA",
            path=summary_path,
        )


def _btc_gold_items(paths: ProjectPaths, items: list[dict[str, Any]], today: pd.Timestamp) -> None:
    summary_path = paths.report_dir / "btc_gold_rotation_summary.json"
    summary = _load_json(summary_path)
    latest = summary.get("latest", {})
    if not latest:
        _add_item(items, "Data sources", "BTC/gold rotation", "missing", "BTC/gold summary JSON is missing or empty.", path=summary_path)
        return

    latest_date = _date(latest.get("shared_date"))
    lag = None if latest_date is None else int((today - latest_date).days)
    status = _status_for_lag_days(lag, ok_days=3, warn_days=5)
    _add_item(
        items,
        "Data sources",
        "BTC/gold rotation",
        status,
        "Uses the latest shared BTC/USD and COMEX gold futures trading date.",
        source=summary.get("sources", {}).get("gold", {}).get("label"),
        date=f"{latest_date:%Y-%m-%d}" if latest_date is not None else None,
        path=summary_path,
    )


def _artifact_items(paths: ProjectPaths, items: list[dict[str, Any]]) -> None:
    for name in REQUIRED_REPORTS:
        path = paths.report_dir / name
        _add_item(
            items,
            "Pipeline artifacts",
            name,
            _artifact_status(path),
            "Required report artifact.",
            path=path,
        )

    for name in REQUIRED_FIGURES:
        path = paths.figure_dir / name
        _add_item(
            items,
            "Pipeline artifacts",
            name,
            _artifact_status(path),
            "Required static figure artifact.",
            path=path,
        )

    for name in REQUIRED_INTERACTIVE:
        path = paths.interactive_dir / name
        _add_item(
            items,
            "Pipeline artifacts",
            name,
            _artifact_status(path),
            "Required interactive dashboard artifact.",
            path=path,
        )


def _overall_status(items: list[dict[str, Any]]) -> str:
    statuses = {item["status"] for item in items}
    if "missing" in statuses or "stale" in statuses:
        return "stale"
    if "warn" in statuses:
        return "warn"
    return "ok"


def build_health_payload(paths: ProjectPaths) -> dict[str, Any]:
    today = _today_utc()
    items: list[dict[str, Any]] = []
    _btc_items(paths, items, today)
    _checkonchain_items(paths, items, today)
    _metals_items(paths, items, today)
    _btc_gold_items(paths, items, today)
    _artifact_items(paths, items)
    status_counts = pd.Series([item["status"] for item in items]).value_counts().to_dict()
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "today_utc": f"{today:%Y-%m-%d}",
        "overall_status": _overall_status(items),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "items": items,
    }


def _status_badge(status: str) -> str:
    labels = {
        "ok": "OK",
        "warn": "WARN",
        "stale": "STALE",
        "missing": "MISSING",
    }
    return f'<span class="badge {html.escape(status)}">{labels.get(status, status.upper())}</span>'


def _row(item: dict[str, Any]) -> str:
    cells = [
        _status_badge(str(item["status"])),
        html.escape(str(item["name"])),
        html.escape(str(item.get("date") or "")),
        html.escape(str(item.get("source") or "")),
        html.escape(str(item["detail"])),
        html.escape(str(item.get("modified_at") or "")),
    ]
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def _category_table(category: str, items: list[dict[str, Any]]) -> str:
    rows = "\n".join(_row(item) for item in items if item["category"] == category)
    return f"""
      <section>
        <div class="section-heading">
          <h2>{html.escape(category)}</h2>
        </div>
        <table>
          <thead>
            <tr>
              <th>Status</th>
              <th>Name</th>
              <th>Latest date</th>
              <th>Source</th>
              <th>Detail</th>
              <th>Modified</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </section>
    """


def _summary_cards(payload: dict[str, Any]) -> str:
    counts = payload["status_counts"]
    cards = [
        ("Overall", str(payload["overall_status"]).upper()),
        ("Generated", str(payload["generated_at"])),
        ("UTC Date", str(payload["today_utc"])),
        ("OK", str(counts.get("ok", 0))),
        ("Warn", str(counts.get("warn", 0))),
        ("Stale/Missing", str(counts.get("stale", 0) + counts.get("missing", 0))),
    ]
    return "\n".join(
        f"""
        <div class="card">
          <div class="card-label">{html.escape(label)}</div>
          <div class="card-value">{html.escape(value)}</div>
        </div>
        """
        for label, value in cards
    )


def _html(payload: dict[str, Any]) -> str:
    items = payload["items"]
    categories = list(dict.fromkeys(item["category"] for item in items))
    tables = "\n".join(_category_table(category, items) for category in categories)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Data And Pipeline Health</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5b6472;
      --line: #d9dee7;
      --band: #f7f9fc;
      --accent: #155e75;
      --ok: #0f8b61;
      --warn: #b7791f;
      --bad: #b42318;
    }}
    body {{
      margin: 0;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: white;
    }}
    header {{
      padding: 18px 28px 10px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      letter-spacing: 0;
      font-weight: 700;
    }}
{dashboard_nav_css()}
    main {{
      padding: 18px 28px 32px;
      max-width: 1680px;
      margin: 0 auto;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 22px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      background: var(--band);
      min-height: 62px;
    }}
    .card-label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 7px;
    }}
    .card-value {{
      font-size: 16px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    section {{
      margin: 0 0 26px;
    }}
    .section-heading {{
      border-bottom: 1px solid var(--line);
      margin: 0 0 8px;
      padding: 0 0 8px;
    }}
    h2 {{
      margin: 0;
      font-size: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #e7ebf0;
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: #fbfcfe;
    }}
    .badge {{
      display: inline-block;
      min-width: 58px;
      text-align: center;
      border-radius: 4px;
      padding: 3px 6px;
      font-size: 11px;
      font-weight: 800;
      color: white;
    }}
    .ok {{ background: var(--ok); }}
    .warn {{ background: var(--warn); }}
    .stale, .missing {{ background: var(--bad); }}
    @media (max-width: 900px) {{
      .cards {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      header {{ align-items: start; flex-direction: column; }}
      table {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Data And Pipeline Health</h1>
    {dashboard_nav("health")}
  </header>
  <main>
    <div class="cards">{_summary_cards(payload)}</div>
    {tables}
  </main>
</body>
</html>
"""


def build_pipeline_health_dashboard(paths: ProjectPaths) -> list[Path]:
    paths.ensure_dirs()
    json_path = paths.report_dir / "pipeline_health.json"
    html_path = paths.interactive_dir / "pipeline_health_dashboard.html"

    payload = build_health_payload(paths)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(_html(payload), encoding="utf-8")

    payload = build_health_payload(paths)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(_html(payload), encoding="utf-8")
    return [json_path, html_path]


def main() -> None:
    for output in build_pipeline_health_dashboard(ProjectPaths.from_cwd()):
        print(output)


if __name__ == "__main__":
    main()
