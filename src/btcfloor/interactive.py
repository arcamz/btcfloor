from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from btcfloor.cycle import (
    DEFAULT_CYCLE_ANCHORS,
    FOURCHAN_FULL_LOW_TO_LOW_DAYS,
    FOURCHAN_LOW_TO_PEAK_DAYS,
)
from btcfloor.data import to_weekly_ohlc
from btcfloor.powerlaw import PowerLawModel
from btcfloor.validation import DEFAULT_CYCLE_LOWS


MODEL_COLORS = {
    "giovanni_power_law_floor": "#d35400",
    "weekly_expectile_power_law_tau_0_0001": "#1b9e77",
    "weekly_expectile_power_law_tau_0_0005": "#4daf4a",
    "weekly_expectile_power_law_tau_0_001": "#377eb8",
    "weekly_expectile_power_law_tau_0_005": "#984ea3",
    "weekly_expectile_power_law_tau_0_01": "#e41a1c",
}

CYCLE_COLORS = {
    "2015_cycle": "#2b8cbe",
    "2018_cycle": "#756bb1",
    "2022_cycle": "#e6550d",
}


def _model_label(model: PowerLawModel) -> str:
    if model.name == "giovanni_power_law_floor":
        return "Giovanni fixed floor"
    if "expectile" in model.name:
        tau_text = model.name.removeprefix("weekly_expectile_power_law_tau_")
        tau_text = tau_text.replace("_", ".")
        tau = float(tau_text)
        return f"Expectile {tau * 100:g}% floor"
    return f"{model.name} floor"


def _nearest_price(daily: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    distances = (daily["date"] - pd.Timestamp(date)).abs()
    return daily.loc[distances.idxmin()]


def _add_cycle_calendar_guides(fig: go.Figure, latest_date: pd.Timestamp) -> None:
    for anchor in DEFAULT_CYCLE_ANCHORS:
        peak_date = anchor.low_date + pd.Timedelta(days=FOURCHAN_LOW_TO_PEAK_DAYS)
        next_low_date = anchor.low_date + pd.Timedelta(days=FOURCHAN_FULL_LOW_TO_LOW_DAYS)
        advance_end = min(peak_date, latest_date)
        decline_start = peak_date
        decline_end = min(next_low_date, latest_date)

        for row in (1, 2):
            if anchor.low_date < advance_end:
                fig.add_vrect(
                    x0=anchor.low_date,
                    x1=advance_end,
                    fillcolor="#2ca25f",
                    opacity=0.045,
                    line_width=0,
                    row=row,
                    col=1,
                )
            if decline_start < decline_end:
                fig.add_vrect(
                    x0=decline_start,
                    x1=decline_end,
                    fillcolor="#de2d26",
                    opacity=0.045,
                    line_width=0,
                    row=row,
                    col=1,
                )
            fig.add_vline(
                x=anchor.low_date,
                line={"color": "#636363", "width": 1, "dash": "dot"},
                row=row,
                col=1,
            )
            fig.add_vline(
                x=peak_date,
                line={"color": "#b7791f", "width": 1, "dash": "dash"},
                row=row,
                col=1,
            )
            fig.add_vline(
                x=next_low_date,
                line={"color": "#b22222", "width": 1, "dash": "dash"},
                row=row,
                col=1,
            )


def _add_cycle_event_markers(fig: go.Figure, daily: pd.DataFrame) -> None:
    low_dates = [low.date for low in DEFAULT_CYCLE_LOWS]
    low_rows = [_nearest_price(daily, low_date) for low_date in low_dates]
    fig.add_trace(
        go.Scatter(
            x=[row["date"] for row in low_rows],
            y=[row["price_usd"] for row in low_rows],
            mode="markers+text",
            name="Observed cycle lows",
            showlegend=False,
            marker={
                "symbol": "triangle-down",
                "size": 11,
                "color": "#111111",
                "line": {"color": "white", "width": 1},
            },
            text=[low.name.replace("_low", "") for low in DEFAULT_CYCLE_LOWS],
            textposition="bottom center",
            textfont={"size": 11, "color": "#111111"},
            hovertemplate=(
                "%{text} low<br>%{x|%Y-%m-%d}<br>Price: %{y:$,.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    peak_dates = [
        anchor.observed_peak_date
        for anchor in DEFAULT_CYCLE_ANCHORS
        if anchor.observed_peak_date is not None
    ]
    peak_rows = [_nearest_price(daily, peak_date) for peak_date in peak_dates]
    fig.add_trace(
        go.Scatter(
            x=[row["date"] for row in peak_rows],
            y=[row["price_usd"] for row in peak_rows],
            mode="markers",
            name="Observed cycle peaks",
            showlegend=False,
            marker={
                "symbol": "triangle-up",
                "size": 11,
                "color": "#b7791f",
                "line": {"color": "white", "width": 1},
            },
            hovertemplate="%{x|%Y-%m-%d}<br>Price: %{y:$,.2f}<extra>Observed peak</extra>",
        ),
        row=1,
        col=1,
    )


def _add_cycle_aligned_panel(
    fig: go.Figure,
    weekly: pd.DataFrame,
    giovanni: PowerLawModel | None,
) -> None:
    if giovanni is None:
        return

    for anchor in DEFAULT_CYCLE_ANCHORS:
        cycle_end = anchor.low_date + pd.Timedelta(days=FOURCHAN_FULL_LOW_TO_LOW_DAYS)
        cycle = weekly.loc[
            weekly["date"].between(anchor.low_date, min(cycle_end, weekly["date"].max())),
            ["date", "close"],
        ].copy()
        if cycle.empty:
            continue
        cycle["cycle_day"] = (cycle["date"] - anchor.low_date).dt.days
        cycle["ratio_to_giovanni_floor"] = (
            cycle["close"].to_numpy(dtype=float)
            / giovanni.predict_price(cycle["date"], floor=True)
        )
        fig.add_trace(
            go.Scatter(
                x=cycle["cycle_day"],
                y=cycle["ratio_to_giovanni_floor"],
                mode="lines",
                name=anchor.name.replace("_", " "),
                showlegend=False,
                line={
                    "width": 2.2,
                    "color": CYCLE_COLORS.get(anchor.name, None),
                },
                hovertemplate=(
                    "Cycle day %{x:,}<br>"
                    "Close / Giovanni floor: %{y:.2f}x"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1,
            col=2,
        )
        fig.add_annotation(
            x=float(cycle["cycle_day"].iloc[-1]),
            y=float(cycle["ratio_to_giovanni_floor"].iloc[-1]),
            text=anchor.name.replace("_cycle", ""),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font={
                "size": 11,
                "color": CYCLE_COLORS.get(anchor.name, "#333333"),
            },
            row=1,
            col=2,
        )

    fig.add_hline(
        y=1.0,
        line={"color": "#b22222", "width": 1.3, "dash": "dash"},
        row=1,
        col=2,
    )
    fig.add_vline(
        x=FOURCHAN_LOW_TO_PEAK_DAYS,
        line={"color": "#b7791f", "width": 1.2, "dash": "dash"},
        row=1,
        col=2,
    )
    fig.add_vline(
        x=FOURCHAN_FULL_LOW_TO_LOW_DAYS,
        line={"color": "#b22222", "width": 1.2, "dash": "dash"},
        row=1,
        col=2,
    )


def write_interactive_weekly_floor_chart(
    daily: pd.DataFrame,
    models: list[PowerLawModel],
    output_path: Path,
) -> Path:
    weekly = to_weekly_ohlc(daily)
    giovanni = next(
        (model for model in models if model.name == "giovanni_power_law_floor"),
        None,
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy", "rowspan": 2}],
            [{"type": "xy"}, None],
        ],
        row_heights=[0.66, 0.34],
        column_widths=[0.74, 0.26],
        horizontal_spacing=0.055,
        vertical_spacing=0.075,
        subplot_titles=[
            "Weekly BTC price and floor variants",
            "Cycle-aligned distance to Giovanni floor",
            "Price / floor distance over time",
        ],
    )
    fig.add_trace(
        go.Candlestick(
            x=weekly["date"],
            open=weekly["open"],
            high=weekly["high"],
            low=weekly["low"],
            close=weekly["close"],
            name="BTC weekly candles",
            increasing_line_color="#0f8b61",
            increasing_fillcolor="rgba(15, 139, 97, 0.65)",
            decreasing_line_color="#c0392b",
            decreasing_fillcolor="rgba(192, 57, 43, 0.65)",
            hoverlabel={"namelength": -1},
        )
        ,
        row=1,
        col=1,
    )

    expectile_floors = [
        (model, model.predict_price(weekly["date"], floor=True))
        for model in models
        if "expectile" in model.name
    ]
    if len(expectile_floors) >= 2:
        lower_floor = expectile_floors[0][1]
        upper_floor = expectile_floors[-1][1]
        fig.add_trace(
            go.Scatter(
                x=weekly["date"],
                y=lower_floor,
                mode="lines",
                name="Expectile variant band lower",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=weekly["date"],
                y=upper_floor,
                mode="lines",
                name="Expectile variant band",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(55, 126, 184, 0.10)",
                hoverinfo="skip",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    for model in models:
        floor = model.predict_price(weekly["date"], floor=True)
        label = _model_label(model)
        color = MODEL_COLORS.get(model.name, None)
        width = 3.0 if model.name == "giovanni_power_law_floor" else 1.8
        opacity = 1.0 if model.name in {
            "giovanni_power_law_floor",
            "weekly_expectile_power_law_tau_0_0001",
        } else 0.78
        visible = (
            "legendonly"
            if model.name
            in {
                "weekly_expectile_power_law_tau_0_0005",
                "weekly_expectile_power_law_tau_0_005",
            }
            else True
        )
        fig.add_trace(
            go.Scatter(
                x=weekly["date"],
                y=floor,
                mode="lines",
                name=label,
                legendgroup=model.name,
                line={
                    "width": width,
                    "color": color,
                },
                opacity=opacity,
                visible=visible,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:$,.2f}<extra>%{fullData.name}</extra>",
            )
            ,
            row=1,
            col=1,
        )
        ratio = weekly["close"].to_numpy(dtype=float) / floor
        fig.add_trace(
            go.Scatter(
                x=weekly["date"],
                y=ratio,
                mode="lines",
                name=f"{label} distance",
                legendgroup=model.name,
                showlegend=False,
                line={
                    "width": width,
                    "color": color,
                },
                opacity=opacity,
                visible=visible,
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Close / floor: %{y:.2f}x"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=2,
            col=1,
        )

        low_x = []
        low_y = []
        low_text = []
        for low in DEFAULT_CYCLE_LOWS:
            low_row = _nearest_price(daily, low.date)
            floor_at_low = float(model.predict_price(pd.Timestamp(low_row["date"]), floor=True)[0])
            low_x.append(low_row["date"])
            low_y.append(float(low_row["price_usd"]) / floor_at_low)
            low_text.append(low.name.replace("_", " "))
        fig.add_trace(
            go.Scatter(
                x=low_x,
                y=low_y,
                mode="markers",
                name=f"{label} low ratios",
                legendgroup=model.name,
                showlegend=False,
                marker={
                    "size": 7,
                    "color": color,
                    "line": {"color": "white", "width": 1},
                },
                visible=visible,
                hovertemplate=(
                    "%{text}<br>%{x|%Y-%m-%d}<br>Price / floor: %{y:.2f}x"
                    "<extra>%{fullData.name}</extra>"
                ),
                text=low_text,
            ),
            row=2,
            col=1,
        )

    latest = daily.iloc[-1]
    _add_cycle_calendar_guides(fig, pd.Timestamp(latest["date"]))
    _add_cycle_event_markers(fig, daily)
    _add_cycle_aligned_panel(fig, weekly, giovanni)

    fig.add_hline(
        y=1.0,
        line={"color": "#b22222", "width": 1.2, "dash": "dash"},
        row=2,
        col=1,
    )

    fig.update_layout(
        title={
            "text": (
                "BTC Floor Variants And Cycle Distance"
                f"<br><sup>Last daily close: {latest['date']:%Y-%m-%d}"
                f" at ${float(latest['price_usd']):,.2f}</sup>"
            ),
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 18},
        },
        template="plotly_white",
        autosize=True,
        width=1900,
        height=1030,
        hovermode="x unified",
        dragmode="pan",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.075,
            "xanchor": "left",
            "x": 0.16,
            "groupclick": "togglegroup",
            "font": {"size": 10},
        },
        margin={"l": 75, "r": 45, "t": 140, "b": 55},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.01,
                "y": 1.085,
                "showactive": True,
                "buttons": [
                    {
                        "label": "Log price",
                        "method": "relayout",
                        "args": [{"yaxis.type": "log"}],
                    },
                    {
                        "label": "Linear price",
                        "method": "relayout",
                        "args": [{"yaxis.type": "linear"}],
                    },
                ],
            }
        ],
    )
    fig.update_xaxes(
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"count": 3, "label": "3Y", "step": "year", "stepmode": "backward"},
                {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                {"count": 10, "label": "10Y", "step": "year", "stepmode": "backward"},
                {"label": "All", "step": "all"},
            ]
        },
        showticklabels=False,
        rangeslider={"visible": False},
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Week",
        matches="x",
        rangeslider={"visible": False},
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="Days since cycle low",
        range=[0, FOURCHAN_FULL_LOW_TO_LOW_DAYS],
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="USD",
        type="log",
        tickprefix="$",
        fixedrange=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Price / floor",
        type="log",
        tickvals=[0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 25, 50],
        ticktext=["0.5x", "0.75x", "1x", "1.5x", "2x", "3x", "5x", "10x", "25x", "50x"],
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Close / Giovanni floor",
        type="log",
        tickvals=[1, 1.25, 1.5, 2, 3, 5, 10, 25, 50],
        ticktext=["1x", "1.25x", "1.5x", "2x", "3x", "5x", "10x", "25x", "50x"],
        row=1,
        col=2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config={"responsive": True},
        default_width="100%",
        default_height="1030px",
    )
    return output_path
