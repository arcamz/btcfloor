from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from btcfloor.cycle import current_cycle_phase
from btcfloor.data import load_price_history, to_weekly_ohlc
from btcfloor.expectile import expectile_model_name, fit_expectile_power_law
from btcfloor.forward_floor import future_floor_overlap_daily
from btcfloor.paths import ProjectPaths
from btcfloor.powerlaw import giovanni_power_law_floor_model


ROI_ENTRIES = [44_000, 50_000, 55_000, 60_000, 63_000, 66_000, 70_000]
ROI_TARGETS = [250_000, 300_000, 350_000]
STARTING_CAPITAL_SEK = 1_000_000
DEEP_LEVELS = [60_000, 57_000, 51_000, 44_000]
CYCLE_LOW_WINDOWS = [
    ("2015 low", pd.Timestamp("2015-01-14"), "#8f99a8"),
    ("2018 low", pd.Timestamp("2018-12-15"), "#5d6c7a"),
    ("2022 low", pd.Timestamp("2022-11-21"), "#b45f4d"),
]


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _money_k(value: float) -> str:
    return f"${value / 1000:.1f}k"


def _sek_m(value: float) -> str:
    return f"{value / 1_000_000:.2f}M SEK"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=True)


def _latest_metric(summary: dict, name: str) -> tuple[pd.Timestamp, float]:
    item = summary["latest"][name]
    return pd.Timestamp(item["date"]), float(item["value"])


def _cvdd_display_name(summary: dict) -> str:
    status = summary.get("source", {}).get("cvdd", {})
    bitbo = status.get("bitbo", {})
    looknode = status.get("looknode", {})
    if bitbo.get("available"):
        return "CVDD (Bitbo)"
    if looknode.get("available"):
        return "CVDD (Looknode fallback)"
    return "CVDD"


def _load_checkonchain(paths: ProjectPaths) -> pd.DataFrame:
    path = paths.processed_dir / "checkonchain_cohort_metrics.csv"
    data = pd.read_csv(path, parse_dates=["date"])
    return data.sort_values("date").reset_index(drop=True)


def _agent_commentary(
    paths: ProjectPaths,
    checkonchain_summary: dict,
) -> tuple[str, str]:
    current = pd.read_csv(paths.report_dir / "current_bottom_summary.csv").iloc[0]
    sma = pd.read_csv(paths.report_dir / "sma_channel_decision_metrics.csv").iloc[0]

    as_of = pd.Timestamp(current["as_of_date"])
    spot = float(current["spot_price_usd"])
    hard = float(current["hard_floor_usd"])
    warning = float(current["warning_floor_usd"])
    pressure = float(current["current_bottom_pressure_score"])
    future_floor = float(sma["future_floor_12m_usd"])
    sma200 = float(sma["sma200_usd"])
    channel_lower = float(sma["channel_lower_usd"])

    _, sth_mvrv = _latest_metric(checkonchain_summary, "STH-MVRV")
    _, sth_z = _latest_metric(checkonchain_summary, "STH-MVRV Z-Score")
    _, price_minus_1sd = _latest_metric(checkonchain_summary, "Price -1.0sd")
    _, price_minus_15sd = _latest_metric(checkonchain_summary, "Price -1.5sd")
    _, lth_mvrv = _latest_metric(checkonchain_summary, "LTH-MVRV")
    _, lth_sopr = _latest_metric(checkonchain_summary, "LTH-SOPR")
    _, lth_loss_ema = _latest_metric(checkonchain_summary, "LTH Realised Loss 7D EMA BTC")

    state = (
        "active value zone, incomplete capitulation"
        if spot > hard and lth_mvrv > 1.0
        else "deep capitulation zone"
    )
    market_commentary = (
        f"As of {as_of:%Y-%m-%d}, spot is {_money_k(spot)}, "
        f"{_pct(spot / hard - 1)} above the Giovanni hard floor and "
        f"{_pct(spot / warning - 1)} versus the 0.01% expectile warning floor. "
        f"The 12m-forward Giovanni floor is {_money_k(future_floor)}, leaving spot "
        f"{_pct(spot / future_floor - 1)} below that forward floor. "
        f"Checkonchain has STH-MVRV {sth_mvrv:.2f}, STH Z {sth_z:.2f}, "
        f"and the -1sd STH price band at {_money_k(price_minus_1sd)}; the -1.5sd band is "
        f"{_money_k(price_minus_15sd)}. LTH-MVRV is {lth_mvrv:.2f}, LTH-SOPR is "
        f"{lth_sopr:.2f}, and LTH realised-loss 7D EMA is {lth_loss_ema:,.0f} BTC. "
        f"Read: {state}. First tranche is defensible; full-size capitulation evidence "
        f"would need a deeper LTH loss impulse or a sweep toward the lower stress bands."
    )
    roi_commentary = (
        f"Current spot is {_money_k(spot)} with bottom-pressure score {pressure:.1f}. "
        f"Price is {_pct(spot / sma200 - 1)} versus the 200D SMA and "
        f"{_pct(spot / channel_lower - 1)} versus the fitted post-rejection channel lower bound. "
        f"Deployment logic: use current floor overlap for limited exposure, reserve larger "
        f"tranches for {_money_k(hard)} to {_money_k(price_minus_15sd)}, and treat "
        f"$51k/$44k as deep-stress tail bids rather than the only acceptable plan."
    )
    return market_commentary, roi_commentary


def _dashboard_shell(
    title: str,
    commentary: str,
    body: str,
    peer_link: str,
    peer_label: str,
) -> str:
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
    nav a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
      border-bottom: 1px solid var(--accent);
    }}
    nav {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
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
      max-width: 720px;
      line-height: 1.35;
    }}
    .weekly-frame {{
      width: 100%;
      height: 1060px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: white;
    }}
    .plain-link {{
      color: var(--accent);
      text-decoration: none;
      border-bottom: 1px solid var(--accent);
      font-weight: 650;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <nav>
      <a href="{peer_link}">{peer_label}</a>
      <a href="pipeline_health_dashboard.html">Data health</a>
    </nav>
  </header>
  <section class="commentary"><strong>Agent commentary:</strong> {commentary}</section>
  <main class="chart-wrap">{body}</main>
</body>
</html>
"""


def _add_weekly_price_panel(fig: go.Figure, daily: pd.DataFrame, row: int) -> None:
    weekly = to_weekly_ohlc(daily)
    price = daily.loc[:, ["date", "price_usd"]].copy().sort_values("date")
    price["sma200"] = price["price_usd"].rolling(200).mean()
    price["sma50"] = price["price_usd"].rolling(50).mean()
    weekly_sma = (
        price.set_index("date").resample("W-SUN").last().dropna(subset=["sma200"]).reset_index()
    )
    weekly_close = weekly.rename(columns={"close": "price_usd"}).loc[
        :, ["date", "days_since_genesis", "price_usd"]
    ]
    giovanni = giovanni_power_law_floor_model()
    expectile = fit_expectile_power_law(
        weekly_close,
        tau=0.0001,
        name=expectile_model_name(0.0001),
    )
    overlap = future_floor_overlap_daily(daily, giovanni, horizon_months=12)

    fig.add_trace(
        go.Candlestick(
            x=weekly["date"],
            open=weekly["open"],
            high=weekly["high"],
            low=weekly["low"],
            close=weekly["close"],
            name="BTC weekly candles",
            increasing_line_color="#138a61",
            increasing_fillcolor="rgba(19,138,97,0.55)",
            decreasing_line_color="#c24136",
            decreasing_fillcolor="rgba(194,65,54,0.55)",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=weekly_sma["date"],
            y=weekly_sma["sma200"],
            name="200D SMA",
            line={"color": "#6d28d9", "width": 1.7},
            hovertemplate="%{x|%Y-%m-%d}<br>200D SMA %{y:$,.0f}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=weekly_sma["date"],
            y=weekly_sma["sma50"],
            name="50D SMA",
            visible="legendonly",
            line={"color": "#15803d", "width": 1.2},
            hovertemplate="%{x|%Y-%m-%d}<br>50D SMA %{y:$,.0f}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    for label, model, color, width in [
        ("Giovanni hard floor", giovanni, "#d95f02", 2.3),
        ("0.01% expectile warning floor", expectile, "#1b9e77", 2.0),
    ]:
        fig.add_trace(
            go.Scatter(
                x=weekly["date"],
                y=model.predict_price(weekly["date"], floor=True),
                name=label,
                line={"color": color, "width": width},
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:$,.0f}<extra>%{fullData.name}</extra>",
            ),
            row=row,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=overlap["date"],
            y=overlap["future_floor_usd"],
            name="12m-forward Giovanni floor",
            line={"color": "#ef4444", "width": 1.2, "dash": "dot"},
            hovertemplate="%{x|%Y-%m-%d}<br>12m floor %{y:$,.0f}<extra></extra>",
        ),
        row=row,
        col=1,
    )


def _plotly_html(fig: go.Figure, height: int) -> str:
    return fig.to_html(
        include_plotlyjs=True,
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


def _weekly_checkonchain(check: pd.DataFrame) -> pd.DataFrame:
    numeric = check.select_dtypes(include="number").columns
    return (
        check.set_index("date")[numeric]
        .resample("W-SUN")
        .last()
        .dropna(how="all")
        .reset_index()
    )


def _latest_non_null(data: pd.DataFrame, column: str) -> tuple[pd.Timestamp, float]:
    row = data.loc[data[column].notna(), ["date", column]].iloc[-1]
    return pd.Timestamp(row["date"]), float(row[column])


def _add_right_label(
    fig: go.Figure,
    x: pd.Timestamp,
    y: float,
    text: str,
    color: str,
    row: int,
) -> None:
    fig.add_annotation(
        x=x,
        y=y,
        text=text,
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font={"size": 11, "color": color},
        bgcolor="rgba(255,255,255,0.78)",
        row=row,
        col=1,
    )


def _build_weekly_realised_price_chart(
    daily: pd.DataFrame,
    check: pd.DataFrame,
    checkonchain_summary: dict,
) -> str:
    weekly_price = to_weekly_ohlc(daily)
    weekly_check = _weekly_checkonchain(check)
    latest_date = pd.Timestamp(weekly_price["date"].iloc[-1])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=[0.72, 0.28],
        subplot_titles=[
            "Weekly BTC price against realised-price stress bands",
            "Weekly holder stress ratios",
        ],
    )
    fig.add_trace(
        go.Candlestick(
            x=weekly_price["date"],
            open=weekly_price["open"],
            high=weekly_price["high"],
            low=weekly_price["low"],
            close=weekly_price["close"],
            name="BTC weekly candles",
            increasing_line_color="#0f8b61",
            increasing_fillcolor="rgba(15,139,97,0.55)",
            decreasing_line_color="#c0392b",
            decreasing_fillcolor="rgba(192,57,43,0.55)",
        ),
        row=1,
        col=1,
    )

    price_lines = [
        ("STH Realised Price", "STH Realised Price", "#2563eb", 2.0, "solid", True),
        ("Price -1.0sd", "Price -1.0sd", "#f59e0b", 1.5, "dash", True),
        ("Price -1.5sd", "Price -1.5sd", "#dc2626", 1.5, "dash", True),
        ("Price -2.0sd", "Price -2.0sd", "#7c3aed", 1.3, "dash", "legendonly"),
        ("Cointime Price", "Cointime Price", "#0891b2", 1.6, "dashdot", True),
        ("CVDD", _cvdd_display_name(checkonchain_summary), "#9333ea", 1.5, "dashdot", True),
        ("LTH Realised Price", "LTH Realised Price", "#111111", 1.8, "dot", True),
        (
            "LTH True Realised Price",
            "LTH True Realised Price",
            "#64748b",
            1.3,
            "dot",
            "legendonly",
        ),
    ]
    for column, display_name, color, width, dash, visible in price_lines:
        if column not in weekly_check:
            continue
        fig.add_trace(
            go.Scatter(
                x=weekly_check["date"],
                y=weekly_check[column],
                name=display_name,
                visible=visible,
                line={"color": color, "width": width, "dash": dash},
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>%{y:$,.0f}<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1,
            col=1,
        )
        label_date, value = _latest_non_null(weekly_check, column)
        if visible is True:
            _add_right_label(
                fig,
                label_date + pd.Timedelta(days=17),
                value,
                f"{display_name.replace('Price ', '')}: {_money_k(value)}",
                color,
                row=1,
            )

    ratio_lines = [
        ("STH-MVRV", "#2563eb", 1.7),
        ("STH-MVRV Z-Score", "#7c3aed", 1.4),
        ("LTH-MVRV", "#0f766e", 1.7),
        ("LTH-SOPR", "#c2410c", 1.4),
    ]
    for column, color, width in ratio_lines:
        if column not in weekly_check:
            continue
        fig.add_trace(
            go.Scatter(
                x=weekly_check["date"],
                y=weekly_check[column],
                name=column,
                line={"color": color, "width": width},
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>%{fullData.name}</extra>",
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=1.0, line={"color": "#a8b1c2", "width": 0.9, "dash": "dash"}, row=2, col=1)
    fig.add_hline(y=-1.0, line={"color": "#cbd5e1", "width": 0.8, "dash": "dot"}, row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=900,
        autosize=True,
        hovermode="x unified",
        dragmode="pan",
        margin={"l": 75, "r": 130, "t": 76, "b": 50},
        legend={"orientation": "h", "y": 1.04, "x": 0.01, "font": {"size": 10}},
    )
    fig.update_yaxes(type="log", title_text="USD", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Ratio / z", row=2, col=1)
    fig.update_xaxes(
        rangeselector={
            "buttons": [
                {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"count": 3, "label": "3Y", "step": "year", "stepmode": "backward"},
                {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                {"label": "All", "step": "all"},
            ]
        },
        rangeslider={"visible": False},
        row=1,
        col=1,
    )
    fig.update_xaxes(
        range=[latest_date - pd.Timedelta(days=540), latest_date + pd.Timedelta(days=95)]
    )
    return _plotly_html(fig, 900)


def _cycle_window(
    data: pd.DataFrame,
    column: str,
    anchor: pd.Timestamp,
    min_weeks: int = -60,
    max_weeks: int = 36,
) -> pd.DataFrame:
    frame = data.loc[:, ["date", column]].dropna().copy()
    frame = frame.loc[
        frame["date"].between(
            anchor + pd.Timedelta(weeks=min_weeks),
            anchor + pd.Timedelta(weeks=max_weeks),
        )
    ]
    if frame.empty:
        return frame
    weekly = (
        frame.set_index("date")
        .resample("W-SUN")
        .last()
        .dropna()
        .reset_index()
    )
    weekly["weeks_from_low"] = (weekly["date"] - anchor).dt.days / 7
    return weekly


def _build_lth_cycle_chart(daily: pd.DataFrame, check: pd.DataFrame) -> str:
    latest_date = pd.Timestamp(daily["date"].iloc[-1])
    expected_low = pd.Timestamp(current_cycle_phase(latest_date)["expected_next_low_date"])
    cycles = [*CYCLE_LOW_WINDOWS, ("2026 expected low", expected_low, "#111111")]
    metrics = [
        ("LTH-MVRV", "Long-holder cost-basis cushion", [1.0], (0.55, 4.2)),
        ("LTH-SOPR", "Long-holder spending at profit/loss", [1.0], (0.45, 2.1)),
        ("LTH Realised Loss 7D EMA BTC", "Long-holder realised-loss impulse", [], None),
    ]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f"{metric}: {subtitle}" for metric, subtitle, _, _ in metrics],
    )
    for row, (metric, _, refs, y_range) in enumerate(metrics, start=1):
        for label, anchor, color in cycles:
            window = _cycle_window(check, metric, anchor)
            if window.empty:
                continue
            is_current = label.startswith("2026")
            fig.add_trace(
                go.Scatter(
                    x=window["weeks_from_low"],
                    y=window[metric],
                    name=label if row == 1 else f"{label} {metric}",
                    showlegend=row == 1,
                    legendgroup=label,
                    line={"color": color, "width": 3.0 if is_current else 1.7},
                    opacity=1.0 if is_current else 0.72,
                    hovertemplate=(
                        "Weeks from low %{x:.0f}<br>%{y:,.2f}"
                        f"<extra>{label} - {metric}</extra>"
                    ),
                ),
                row=row,
                col=1,
            )
        for ref in refs:
            fig.add_hline(
                y=ref,
                line={"color": "#a8b1c2", "width": 0.9, "dash": "dash"},
                row=row,
                col=1,
            )
        fig.add_vline(
            x=0,
            line={"color": "#111111", "width": 0.9, "dash": "dot"},
            row=row,
            col=1,
        )
        if y_range is not None:
            fig.update_yaxes(range=list(y_range), row=row, col=1)

    fig.update_layout(
        template="plotly_white",
        height=900,
        autosize=True,
        hovermode="x unified",
        dragmode="pan",
        margin={"l": 75, "r": 35, "t": 76, "b": 50},
        legend={"orientation": "h", "y": 1.04, "x": 0.01, "font": {"size": 10}},
    )
    fig.update_xaxes(
        range=[-60, 36],
        title_text="Weeks from actual / expected cycle low",
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="MVRV", row=1, col=1)
    fig.update_yaxes(title_text="SOPR", row=2, col=1)
    fig.update_yaxes(title_text="BTC", row=3, col=1)
    return _plotly_html(fig, 900)


def build_market_dashboard(paths: ProjectPaths) -> Path:
    daily = load_price_history(paths.processed_btc_csv)
    check = _load_checkonchain(paths)
    summary = _read_json(paths.report_dir / "checkonchain_cohort_summary.json")
    market_commentary, _ = _agent_commentary(paths, summary)

    weekly_frame = """
      <iframe
        class="weekly-frame"
        title="Original weekly BTC floor dashboard"
        src="btc_floor_weekly.html"
      ></iframe>
    """
    body = "\n".join(
        [
            _section(
                "Original weekly floor dashboard",
                (
                    "The prior wide layout is preserved here: candles, floor variants, "
                    "cycle distance, expected-low projections, and the 200D SMA."
                ),
                weekly_frame,
            ),
            _section(
                "Weekly realised-price stress map",
                (
                    "Candles stay on the weekly frame while STH/LTH realised-price and "
                    "STH sigma bands, Cointime Price, and source-labelled CVDD show "
                    "how close price is to holder cost-basis stress."
                ),
                _build_weekly_realised_price_chart(daily, check, summary),
            ),
            _section(
                "Weekly LTH low-tracking comparison",
                (
                    "The same LTH metrics are reanchored around prior cycle lows and the "
                    "October 2026 expected low, so current stress can be compared like-for-like."
                ),
                _build_lth_cycle_chart(daily, check),
            ),
            _section(
                "Static report images",
                (
                    "These are regenerated by the daily updater and remain useful for quick "
                    "mobile review when the interactive charts are too heavy."
                ),
                """
                <p>
                  <a class="plain-link" href="../figures/checkonchain_cohort_current_bands.png">Cohort current bands</a>
                  &nbsp;&middot;&nbsp;
                  <a class="plain-link" href="../figures/checkonchain_low_signal_compare.png">LTH signal comparison</a>
                  &nbsp;&middot;&nbsp;
                  <a class="plain-link" href="../figures/checkonchain_lth_realised_loss_cycle.png">LTH realised-loss cycle</a>
                  &nbsp;&middot;&nbsp;
                  <a class="plain-link" href="../figures/sma_channel_decision_plot.png">SMA/channel tactical plot</a>
                  &nbsp;&middot;&nbsp;
                  <a class="plain-link" href="metals_relative_dashboard.html">Metals relative dashboard</a>
                </p>
                """,
            ),
        ]
    )
    output = paths.interactive_dir / "btc_market_dashboard.html"
    output.write_text(
        _dashboard_shell(
            "BTC Market And On-Chain Dashboard",
            market_commentary,
            body,
            "btc_roi_dashboard.html",
            "Open ROI dashboard",
        ),
        encoding="utf-8",
    )
    return output


def _roi_table(multiplier: float) -> go.Table:
    header = ["Entry"] + [f"Exit {_money_k(target)}" for target in ROI_TARGETS]
    cells = [
        [_money(entry) for entry in ROI_ENTRIES],
        *[
            [f"{target / entry * multiplier:.2f}x" for entry in ROI_ENTRIES]
            for target in ROI_TARGETS
        ],
    ]
    return go.Table(
        header={"values": header, "fill_color": "#e8eef7", "align": "left"},
        cells={"values": cells, "align": "left", "height": 26},
        name=f"{multiplier:g}x return multiples",
    )


def _ending_table(multiplier: float) -> go.Table:
    header = ["Entry"] + [f"Exit {_money_k(target)}" for target in ROI_TARGETS]
    cells = [
        [_money(entry) for entry in ROI_ENTRIES],
        *[
            [_sek_m(STARTING_CAPITAL_SEK * target / entry * multiplier) for entry in ROI_ENTRIES]
            for target in ROI_TARGETS
        ],
    ]
    return go.Table(
        header={"values": header, "fill_color": "#edf7ed", "align": "left"},
        cells={"values": cells, "align": "left", "height": 26},
        name=f"{multiplier:g}x ending value",
    )


def _delta_table(multiplier: float) -> go.Table:
    baseline = {target: STARTING_CAPITAL_SEK * target / 70_000 * multiplier for target in ROI_TARGETS}
    header = ["Entry"] + [f"Extra vs $70k at {_money_k(target)}" for target in ROI_TARGETS]
    cells = [
        [_money(entry) for entry in ROI_ENTRIES if entry != 70_000],
        *[
            [
                _sek_m(STARTING_CAPITAL_SEK * target / entry * multiplier - baseline[target])
                for entry in ROI_ENTRIES
                if entry != 70_000
            ]
            for target in ROI_TARGETS
        ],
    ]
    return go.Table(
        header={"values": header, "fill_color": "#fff4e6", "align": "left"},
        cells={"values": cells, "align": "left", "height": 26},
        name=f"{multiplier:g}x delta vs $70k",
    )


def _drawdown_table() -> go.Table:
    entries = [70_000, 66_000, 63_000, 60_000, 57_000, 51_000]
    header = ["Entry"] + [f"Drop to {_money_k(level)}" for level in DEEP_LEVELS]
    values = [[_money(entry) for entry in entries]]
    for level in DEEP_LEVELS:
        column = []
        for entry in entries:
            if level >= entry:
                column.append("in profit / breakeven")
            else:
                drawdown = level / entry - 1.0
                recovery = entry / level - 1.0
                column.append(f"{drawdown:.1%} dd; {recovery:.1%} recover")
        values.append(column)
    return go.Table(
        header={"values": header, "fill_color": "#f1f5f9", "align": "left"},
        cells={"values": values, "align": "left", "height": 28},
        name="downside/recovery matrix",
    )


def _decision_table(paths: ProjectPaths, summary: dict) -> go.Table:
    current = pd.read_csv(paths.report_dir / "current_bottom_summary.csv").iloc[0]
    hard = float(current["hard_floor_usd"])
    spot = float(current["spot_price_usd"])
    _, minus_15 = _latest_metric(summary, "Price -1.5sd")
    _, minus_2 = _latest_metric(summary, "Price -2.0sd")
    rows = [
        ("Now", spot, "First tranche only", "Spot overlaps warning floor and STH -1sd"),
        ("Hard-floor zone", hard, "Add larger tranche", "Giovanni same-day floor proximity"),
        ("STH -1.5sd", minus_15, "High-conviction stress add", "Deeper short-holder capitulation"),
        ("STH -2sd", minus_2, "Deep-stress reserve", "Rare short-holder stress band"),
        ("Tail bid", 44_000, "Only if liquidation cascade", "Great ROI but high non-fill risk"),
        ("Reclaim", 68_800, "Momentum/SFP add", "Failed breakdown / local reclaim"),
        ("Macro repair", 73_600, "Stop chasing lower bids", "Range repair and reduced non-fill risk"),
    ]
    return go.Table(
        header={"values": ["Zone", "Level", "Action", "Evidence"], "fill_color": "#e0f2fe", "align": "left"},
        cells={
            "values": [
                [row[0] for row in rows],
                [_money_k(row[1]) for row in rows],
                [row[2] for row in rows],
                [row[3] for row in rows],
            ],
            "align": "left",
            "height": 28,
        },
        name="decision matrix",
    )


def build_roi_dashboard(paths: ProjectPaths) -> Path:
    summary = _read_json(paths.report_dir / "checkonchain_cohort_summary.json")
    _, roi_commentary = _agent_commentary(paths, summary)
    specs = [
        [{"type": "table"}, {"type": "table"}],
        [{"type": "table"}, {"type": "table"}],
        [{"type": "table", "colspan": 2}, None],
    ]
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=specs,
        subplot_titles=[
            "Return multiple",
            "Ending value from 1M SEK",
            "Extra profit vs $70k entry",
            "Downside / recovery burden",
            "Deployment decision matrix",
        ],
        vertical_spacing=0.11,
        horizontal_spacing=0.045,
    )
    base_traces = [_roi_table(1.0), _ending_table(1.0), _delta_table(1.0)]
    levered_traces = [_roi_table(1.5), _ending_table(1.5), _delta_table(1.5)]
    positions = [(1, 1), (1, 2), (2, 1)]
    for trace, pos in zip(base_traces, positions):
        fig.add_trace(trace, row=pos[0], col=pos[1])
    for trace, pos in zip(levered_traces, positions):
        trace.visible = False
        fig.add_trace(trace, row=pos[0], col=pos[1])
    fig.add_trace(_drawdown_table(), row=2, col=2)
    fig.add_trace(_decision_table(paths, summary), row=3, col=1)

    visible_1x = [True, True, True, False, False, False, True, True]
    visible_15x = [False, False, False, True, True, True, True, True]
    fig.update_layout(
        template="plotly_white",
        height=1130,
        margin={"l": 35, "r": 35, "t": 105, "b": 35},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 1.08,
                "showactive": True,
                "buttons": [
                    {"label": "1.0x exposure", "method": "update", "args": [{"visible": visible_1x}]},
                    {"label": "1.5x exposure", "method": "update", "args": [{"visible": visible_15x}]},
                ],
            }
        ],
        annotations=[
            *fig.layout.annotations,
            {
                "x": 1.0,
                "y": 1.08,
                "xref": "paper",
                "yref": "paper",
                "text": "Assumes USD BTC entries/exits, 1M SEK starting capital, no tax/fees/slippage.",
                "showarrow": False,
                "font": {"size": 12, "color": "#5b6472"},
                "xanchor": "right",
            },
        ],
    )
    chart = fig.to_html(
        include_plotlyjs=True,
        full_html=False,
        config={"responsive": True, "displaylogo": False},
        default_width="100%",
        default_height="1130px",
    )
    output = paths.interactive_dir / "btc_roi_dashboard.html"
    output.write_text(
        _dashboard_shell(
            "BTC ROI And Deployment Dashboard",
            roi_commentary,
            chart,
            "btc_market_dashboard.html",
            "Open market dashboard",
        ),
        encoding="utf-8",
    )
    return output


def build_dashboards(paths: ProjectPaths) -> list[Path]:
    paths.interactive_dir.mkdir(parents=True, exist_ok=True)
    return [build_market_dashboard(paths), build_roi_dashboard(paths)]


def main() -> None:
    for output in build_dashboards(ProjectPaths.from_cwd()):
        print(output)


if __name__ == "__main__":
    main()
