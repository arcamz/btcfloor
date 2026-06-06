from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from btcfloor.cycle import current_cycle_phase
from btcfloor.dashboard_common import dashboard_nav, dashboard_nav_css
from btcfloor.data import load_price_history
from btcfloor.paths import ProjectPaths
from btcfloor.powerlaw import giovanni_power_law_floor_model
from build_metals_dashboard import (
    GOLD_FUTURES_SYMBOL,
    PRIMARY_SOURCE_LABEL,
    YAHOO_CHART_URL,
    _load_yahoo_futures,
)


def _linear_range(values: pd.Series, pad_fraction: float = 0.08) -> list[float] | None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return None
    low = float(finite.min())
    high = float(finite.max())
    if math.isclose(low, high):
        pad = abs(low) * pad_fraction or 1.0
    else:
        pad = (high - low) * pad_fraction
    return [low - pad, high + pad]


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _money_2(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _ratio(value: float) -> str:
    return f"{value:.2f} oz"


def _load_gold(paths: ProjectPaths) -> pd.DataFrame:
    gold = _load_yahoo_futures(GOLD_FUTURES_SYMBOL, "Gold futures")
    gold.to_csv(paths.processed_dir / "yahoo_gold_futures.csv", index=False)
    return gold


def _build_ratio_frame(btc: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    btc_frame = btc.loc[:, ["date", "price_usd"]].rename(columns={"price_usd": "btc_usd"})
    gold_frame = gold.loc[:, ["date", "price"]].rename(columns={"price": "gold_usd"})
    frame = pd.merge(btc_frame, gold_frame, on="date", how="inner").sort_values("date")
    frame = frame.dropna(subset=["btc_usd", "gold_usd"]).copy()
    frame = frame.loc[(frame["btc_usd"] > 0) & (frame["gold_usd"] > 0)]
    frame["btc_xau"] = frame["btc_usd"] / frame["gold_usd"]
    frame["btc_xau_sma20"] = frame["btc_xau"].rolling(20).mean()
    frame["btc_xau_sma50"] = frame["btc_xau"].rolling(50).mean()
    frame["btc_xau_sma200"] = frame["btc_xau"].rolling(200).mean()
    frame["btc_return_index"] = frame["btc_usd"] / frame["btc_usd"].iloc[0] * 100.0
    frame["gold_return_index"] = frame["gold_usd"] / frame["gold_usd"].iloc[0] * 100.0
    frame["btc_gold_relative_index"] = frame["btc_xau"] / frame["btc_xau"].iloc[0] * 100.0

    floor = giovanni_power_law_floor_model().predict_price(frame["date"], floor=True)
    frame["btc_to_giovanni_floor"] = frame["btc_usd"].to_numpy(dtype=float) / floor
    return frame.reset_index(drop=True)


def _build_weekly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        frame.set_index("date")
        .resample("W-FRI")
        .last()
        .dropna(subset=["btc_xau"])
        .reset_index()
    )
    weekly["btc_xau_sma20w"] = weekly["btc_xau"].rolling(20).mean()
    weekly["btc_xau_sma50w"] = weekly["btc_xau"].rolling(50).mean()
    weekly["btc_xau_sma200w"] = weekly["btc_xau"].rolling(200).mean()
    return weekly


def _normalised_since(frame: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    window = frame.loc[frame["date"] >= anchor].copy()
    if window.empty:
        return window
    first = window.iloc[0]
    window["btc_since_anchor"] = window["btc_usd"] / float(first["btc_usd"]) * 100.0
    window["gold_since_anchor"] = window["gold_usd"] / float(first["gold_usd"]) * 100.0
    window["relative_since_anchor"] = window["btc_xau"] / float(first["btc_xau"]) * 100.0
    return window


def _above(value: float, reference: float) -> bool:
    return pd.notna(reference) and value >= reference


def _rotation_state(latest: pd.Series, weekly_latest: pd.Series) -> str:
    ratio = float(latest["btc_xau"])
    daily20 = float(latest["btc_xau_sma20"])
    daily50 = float(latest["btc_xau_sma50"])
    daily200 = float(latest["btc_xau_sma200"])
    weekly20 = float(weekly_latest["btc_xau_sma20w"])
    weekly50 = float(weekly_latest["btc_xau_sma50w"])

    if _above(ratio, daily200) and _above(float(weekly_latest["btc_xau"]), weekly50):
        return "BTC/gold regime repair"
    if _above(ratio, daily50) and _above(float(weekly_latest["btc_xau"]), weekly20):
        return "BTC rotation improving"
    if _above(ratio, daily20):
        return "BTC tactical bounce only"
    return "gold still leading BTC"


def _make_rotation_chart(
    frame: pd.DataFrame,
    breach_date: pd.Timestamp,
    expected_low_date: pd.Timestamp,
) -> go.Figure:
    latest = frame.iloc[-1]
    anchor_window = _normalised_since(frame, breach_date)
    view_start = max(pd.Timestamp(latest["date"]) - pd.Timedelta(days=365), frame["date"].min())
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=[
            "BTC priced in gold ounces",
            "Opportunity cost since BTC future-floor breach",
            "BTC value pressure: close / Giovanni hard floor",
        ],
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["btc_xau"],
            name="BTC/XAU",
            line={"color": "#111111", "width": 2.2},
            hovertemplate="%{x|%Y-%m-%d}<br>BTC/XAU %{y:.2f} oz<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for column, name, color, width in [
        ("btc_xau_sma20", "20D", "#0f9f8f", 1.6),
        ("btc_xau_sma50", "50D", "#4c5fd7", 1.5),
        ("btc_xau_sma200", "200D", "#8a8f98", 1.7),
    ]:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame[column],
                name=name,
                line={"color": color, "width": width},
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{name} %{{y:.2f}} oz<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if not anchor_window.empty:
        fig.add_trace(
            go.Scatter(
                x=anchor_window["date"],
                y=anchor_window["btc_since_anchor"],
                name="BTC since breach",
                line={"color": "#f97316", "width": 2.0},
                hovertemplate="%{x|%Y-%m-%d}<br>BTC %{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=anchor_window["date"],
                y=anchor_window["gold_since_anchor"],
                name="Gold since breach",
                line={"color": "#b7791f", "width": 2.0},
                hovertemplate="%{x|%Y-%m-%d}<br>Gold %{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=anchor_window["date"],
                y=anchor_window["relative_since_anchor"],
                name="BTC/gold relative",
                line={"color": "#111111", "width": 1.6, "dash": "dot"},
                hovertemplate="%{x|%Y-%m-%d}<br>BTC/gold %{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=100.0, line={"color": "#a8b1c2", "width": 1, "dash": "dash"}, row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["btc_to_giovanni_floor"],
            name="BTC / hard floor",
            line={"color": "#155e75", "width": 2.0},
            hovertemplate="%{x|%Y-%m-%d}<br>BTC / hard floor %{y:.2f}x<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=1.0, line={"color": "#b22222", "width": 1.2, "dash": "dash"}, row=3, col=1)
    fig.add_hline(y=1.1, line={"color": "#b7791f", "width": 1.0, "dash": "dot"}, row=3, col=1)

    for row in (1, 2, 3):
        fig.add_vline(
            x=breach_date,
            line={"color": "#b7791f", "width": 1.0, "dash": "dot"},
            row=row,
            col=1,
        )
        fig.add_vline(
            x=expected_low_date,
            line={"color": "#b22222", "width": 1.0, "dash": "dash"},
            row=row,
            col=1,
        )

    fig.add_annotation(
        x=pd.Timestamp(latest["date"]),
        y=float(latest["btc_xau"]),
        text=f"Latest {_ratio(float(latest['btc_xau']))}",
        showarrow=True,
        arrowhead=2,
        ax=-80,
        ay=-26,
        row=1,
        col=1,
    )
    fig.update_layout(
        template="plotly_white",
        height=1080,
        autosize=True,
        hovermode="x unified",
        dragmode="pan",
        margin={"l": 74, "r": 36, "t": 82, "b": 48},
        legend={"orientation": "h", "y": 1.03, "x": 0.01, "font": {"size": 11}},
    )
    view_end = pd.Timestamp(latest["date"]) + pd.Timedelta(days=20)
    visible = frame.loc[frame["date"].between(view_start, pd.Timestamp(latest["date"]))]
    ratio_range = _linear_range(
        visible[["btc_xau", "btc_xau_sma20", "btc_xau_sma50", "btc_xau_sma200"]].stack()
    )
    if not anchor_window.empty:
        visible_anchor = anchor_window.loc[
            anchor_window["date"].between(view_start, pd.Timestamp(latest["date"]))
        ]
        index_range = _linear_range(
            visible_anchor[
                ["btc_since_anchor", "gold_since_anchor", "relative_since_anchor"]
            ].stack()
        )
    else:
        index_range = None
    floor_range = _linear_range(
        pd.concat([visible["btc_to_giovanni_floor"], pd.Series([1.0, 1.1])], ignore_index=True)
    )

    fig.update_xaxes(range=[view_start, view_end])
    fig.update_yaxes(title_text="Gold oz / BTC", range=ratio_range, row=1, col=1)
    fig.update_yaxes(title_text="Indexed to 100", range=index_range, row=2, col=1)
    fig.update_yaxes(title_text="Multiple", range=floor_range, row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    return fig


def _make_weekly_chart(weekly: pd.DataFrame, expected_low_date: pd.Timestamp) -> go.Figure:
    latest = weekly.iloc[-1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weekly["date"],
            y=weekly["btc_xau"],
            name="BTC/XAU weekly",
            line={"color": "#111111", "width": 2.3},
            hovertemplate="%{x|%Y-%m-%d}<br>BTC/XAU %{y:.2f} oz<extra></extra>",
        )
    )
    for column, name, color in [
        ("btc_xau_sma20w", "20W", "#0f9f8f"),
        ("btc_xau_sma50w", "50W", "#4c5fd7"),
        ("btc_xau_sma200w", "200W", "#8a8f98"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=weekly["date"],
                y=weekly[column],
                name=name,
                line={"color": color, "width": 1.7},
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{name} %{{y:.2f}} oz<extra></extra>",
            )
        )
    fig.add_vline(
        x=expected_low_date,
        line={"color": "#b22222", "width": 1.0, "dash": "dash"},
    )
    fig.add_annotation(
        x=pd.Timestamp(latest["date"]),
        y=float(latest["btc_xau"]),
        text=f"Latest {_ratio(float(latest['btc_xau']))}",
        showarrow=True,
        arrowhead=2,
        ax=-72,
        ay=-28,
    )
    fig.update_layout(
        template="plotly_white",
        height=650,
        autosize=True,
        hovermode="x unified",
        dragmode="pan",
        margin={"l": 74, "r": 36, "t": 54, "b": 48},
        legend={"orientation": "h", "y": 1.04, "x": 0.01, "font": {"size": 11}},
    )
    fig.update_yaxes(title_text="Gold oz / BTC", type="log")
    fig.update_xaxes(title_text="Week")
    return fig


def _make_decision_table(summary: dict[str, object]) -> go.Figure:
    latest = summary["latest"]
    daily = latest["daily_signal"]
    weekly = latest["weekly_signal"]
    rows = [
        (
            "Daily 20D reclaim",
            daily["above_20d"],
            "First BTC probe if sustained above 20D",
            f"20D {_ratio(float(latest['btc_xau_sma20']))}",
        ),
        (
            "Daily 50D reclaim",
            daily["above_50d"],
            "Increase rotation if 20D and 50D both repair",
            f"50D {_ratio(float(latest['btc_xau_sma50']))}",
        ),
        (
            "Daily 200D reclaim",
            daily["above_200d"],
            "Treat as stronger regime repair",
            f"200D {_ratio(float(latest['btc_xau_sma200']))}",
        ),
        (
            "Weekly 20W reclaim",
            weekly["above_20w"],
            "Weekly confirmation for more than a bounce",
            f"20W {_ratio(float(latest['btc_xau_sma20w']))}",
        ),
        (
            "Weekly 50W reclaim",
            weekly["above_50w"],
            "Higher-conviction BTC leadership",
            f"50W {_ratio(float(latest['btc_xau_sma50w']))}",
        ),
        (
            "BTC floor pressure",
            summary["btc_floor_context"]["risk_state"],
            "Keep value context separate from relative momentum",
            f"Pressure {float(summary['btc_floor_context']['pressure_score']):.1f}",
        ),
    ]
    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": ["Check", "Current", "Use", "Level/context"],
                    "fill_color": "#e8eef7",
                    "align": "left",
                },
                cells={
                    "values": [
                        [row[0] for row in rows],
                        [str(row[1]) for row in rows],
                        [row[2] for row in rows],
                        [row[3] for row in rows],
                    ],
                    "align": "left",
                    "height": 30,
                },
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=330,
        margin={"l": 20, "r": 20, "t": 10, "b": 10},
    )
    return fig


def _plotly_div(fig: go.Figure, *, include_plotlyjs: bool, height: int) -> str:
    return fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        config={"responsive": True, "displaylogo": False},
        default_width="100%",
        default_height=f"{height}px",
    )


def _section(title: str, subtitle: str, content: str) -> str:
    return f"""
    <section class="section">
      <div class="section-heading">
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
      {content}
    </section>
    """


def _shell(title: str, commentary: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
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
    .commentary {{
      margin: 14px 28px 4px;
      padding: 12px 14px;
      background: var(--band);
      border: 1px solid var(--line);
      border-radius: 6px;
      font-size: 14px;
      line-height: 1.45;
      color: #263242;
    }}
    .chart-wrap {{
      padding: 8px 20px 28px;
    }}
    .section {{
      width: 100%;
      max-width: 1920px;
      margin: 0 auto 26px;
    }}
    .section-heading {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      border-bottom: 1px solid var(--line);
      margin: 0 0 8px;
      padding: 0 0 8px;
    }}
    .section-heading h2 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .section-heading p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      text-align: right;
      max-width: 760px;
      line-height: 1.35;
    }}
    @media (max-width: 900px) {{
      header {{
        align-items: start;
        flex-direction: column;
      }}
      .section-heading {{
        align-items: start;
        flex-direction: column;
      }}
      .section-heading p {{
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    {dashboard_nav("btc_gold")}
  </header>
  <section class="commentary"><strong>Agent commentary:</strong> {commentary}</section>
  <main class="chart-wrap">{body}</main>
</body>
</html>
"""


def _build_summary(
    paths: ProjectPaths,
    frame: pd.DataFrame,
    weekly: pd.DataFrame,
    btc: pd.DataFrame,
    gold: pd.DataFrame,
) -> dict[str, object]:
    latest = frame.iloc[-1]
    weekly_latest = weekly.iloc[-1]
    current = pd.read_csv(paths.report_dir / "current_bottom_summary.csv").iloc[0]
    breach_date = pd.Timestamp(
        pd.read_csv(paths.report_dir / "sma_channel_decision_metrics.csv").iloc[0][
            "future_floor_breach_date"
        ]
    )
    expected_low_date = pd.Timestamp(current_cycle_phase(pd.Timestamp(latest["date"]))["expected_next_low_date"])
    ratio = float(latest["btc_xau"])
    weekly_ratio = float(weekly_latest["btc_xau"])
    summary = {
        "sources": {
            "btc": {
                "label": "Processed BTC daily CSV",
                "path": str(paths.processed_btc_csv),
                "latest_date": f"{pd.Timestamp(btc['date'].iloc[-1]):%Y-%m-%d}",
            },
            "gold": {
                "label": PRIMARY_SOURCE_LABEL,
                "symbol": GOLD_FUTURES_SYMBOL,
                "url": YAHOO_CHART_URL.format(symbol=GOLD_FUTURES_SYMBOL),
                "latest_date": f"{pd.Timestamp(gold['date'].iloc[-1]):%Y-%m-%d}",
            },
        },
        "latest": {
            "shared_date": f"{pd.Timestamp(latest['date']):%Y-%m-%d}",
            "btc_usd": float(latest["btc_usd"]),
            "gold_usd": float(latest["gold_usd"]),
            "btc_xau": ratio,
            "btc_xau_sma20": float(latest["btc_xau_sma20"]),
            "btc_xau_sma50": float(latest["btc_xau_sma50"]),
            "btc_xau_sma200": float(latest["btc_xau_sma200"]),
            "btc_xau_sma20w": float(weekly_latest["btc_xau_sma20w"]),
            "btc_xau_sma50w": float(weekly_latest["btc_xau_sma50w"]),
            "btc_xau_sma200w": float(weekly_latest["btc_xau_sma200w"]),
            "daily_signal": {
                "above_20d": _above(ratio, float(latest["btc_xau_sma20"])),
                "above_50d": _above(ratio, float(latest["btc_xau_sma50"])),
                "above_200d": _above(ratio, float(latest["btc_xau_sma200"])),
            },
            "weekly_signal": {
                "above_20w": _above(weekly_ratio, float(weekly_latest["btc_xau_sma20w"])),
                "above_50w": _above(weekly_ratio, float(weekly_latest["btc_xau_sma50w"])),
                "above_200w": _above(weekly_ratio, float(weekly_latest["btc_xau_sma200w"])),
            },
            "rotation_state": _rotation_state(latest, weekly_latest),
        },
        "timeline": {
            "future_floor_breach_date": f"{breach_date:%Y-%m-%d}",
            "expected_low_date": f"{expected_low_date:%Y-%m-%d}",
        },
        "btc_floor_context": {
            "spot_usd": float(current["spot_price_usd"]),
            "hard_floor_usd": float(current["hard_floor_usd"]),
            "warning_floor_usd": float(current["warning_floor_usd"]),
            "pressure_score": float(current["current_bottom_pressure_score"]),
            "risk_state": str(current["current_risk_state"]),
        },
    }
    return summary


def build_btc_gold_dashboard(paths: ProjectPaths) -> list[Path]:
    paths.ensure_dirs()
    btc = load_price_history(paths.processed_btc_csv)
    gold = _load_gold(paths)
    frame = _build_ratio_frame(btc, gold)
    weekly = _build_weekly_frame(frame)

    if frame.empty or weekly.empty:
        raise RuntimeError("BTC/gold dashboard requires non-empty BTC and gold overlap data")

    summary = _build_summary(paths, frame, weekly, btc, gold)
    summary_path = paths.report_dir / "btc_gold_rotation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest = summary["latest"]
    floor = summary["btc_floor_context"]
    breach_date = pd.Timestamp(summary["timeline"]["future_floor_breach_date"])
    expected_low_date = pd.Timestamp(summary["timeline"]["expected_low_date"])
    commentary = (
        f"Latest shared BTC/gold date is {latest['shared_date']}: BTC {_money(float(latest['btc_usd']))}, "
        f"gold {_money_2(float(latest['gold_usd']))}, BTC/XAU {_ratio(float(latest['btc_xau']))}. "
        f"Daily reads: 20D {_ratio(float(latest['btc_xau_sma20']))}, "
        f"50D {_ratio(float(latest['btc_xau_sma50']))}, 200D {_ratio(float(latest['btc_xau_sma200']))}; "
        f"weekly 20W {_ratio(float(latest['btc_xau_sma20w']))}. "
        f"State: {latest['rotation_state']}. BTC floor pressure remains high "
        f"({float(floor['pressure_score']):.1f}, {floor['risk_state']}), so this page separates "
        f"value context from the relative-momentum trigger for rotating from gold back into BTC."
    )

    frame.to_csv(paths.processed_dir / "btc_gold_ratio_daily.csv", index=False)
    weekly.to_csv(paths.processed_dir / "btc_gold_ratio_weekly.csv", index=False)

    body = "\n".join(
        [
            _section(
                "Daily BTC/gold rotation monitor",
                (
                    "Close-to-close BTC/XAU avoids synthetic weekend candles. Reclaiming 20D is a probe; "
                    "50D and weekly 20W matter more for a real rotation from gold into BTC."
                ),
                _plotly_div(
                    _make_rotation_chart(frame, breach_date, expected_low_date),
                    include_plotlyjs=True,
                    height=1080,
                ),
            ),
            _section(
                "Weekly BTC/gold regime view",
                (
                    "Use weekly moving averages to avoid overreacting to one or two daily ratio closes. "
                    "This is the cleaner confirmation layer."
                ),
                _plotly_div(_make_weekly_chart(weekly, expected_low_date), include_plotlyjs=False, height=650),
            ),
            _section(
                "Rotation checklist",
                (
                    "A compact rules table keeps the cross-asset switch separate from BTC floor-pressure analysis."
                ),
                _plotly_div(_make_decision_table(summary), include_plotlyjs=False, height=330),
            ),
        ]
    )
    dashboard = paths.interactive_dir / "btc_gold_rotation_dashboard.html"
    dashboard.write_text(
        _shell("BTC/Gold Rotation Dashboard", commentary, body),
        encoding="utf-8",
    )
    return [
        paths.processed_dir / "yahoo_gold_futures.csv",
        paths.processed_dir / "btc_gold_ratio_daily.csv",
        paths.processed_dir / "btc_gold_ratio_weekly.csv",
        summary_path,
        dashboard,
    ]


def main() -> None:
    for output in build_btc_gold_dashboard(ProjectPaths.from_cwd()):
        print(output)


if __name__ == "__main__":
    main()
