from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

from btcfloor.dashboard_common import dashboard_nav, dashboard_nav_css
from btcfloor.paths import ProjectPaths


GOLD_PM_URL = "https://prices.lbma.org.uk/json/gold_pm.json"
SILVER_URL = "https://prices.lbma.org.uk/json/silver.json"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; btcfloor/0.1)"}
GOLD_FUTURES_SYMBOL = "GC=F"
SILVER_FUTURES_SYMBOL = "SI=F"
PRIMARY_SOURCE_LABEL = "Yahoo Finance COMEX futures"
YAHOO_LIVE_RANGE = "5y"

CURRENT_ANCHOR = pd.Timestamp("2026-01-29")
GOLD_PROJECTION_DAYS = 270
SILVER_PROJECTION_DAYS = 240

GOLD_ANALOG_ANCHORS = {
    "1973 analog, LBMA PM fix": pd.Timestamp("1973-07-06"),
    "2006 analog, LBMA PM fix": pd.Timestamp("2006-05-12"),
}
SILVER_ANALOG_ANCHORS = {
    "1974": pd.Timestamp("1974-02-26"),
    "1980": pd.Timestamp("1980-01-18"),
    "2004": pd.Timestamp("2004-04-02"),
    "2006": pd.Timestamp("2006-05-12"),
    "2011": pd.Timestamp("2011-04-28"),
    "2008": pd.Timestamp("2008-03-17"),
}

CHANNEL_BASE_WINDOW = ("2023-08-01", "2023-11-15")
CHANNEL_SPACING_USD = 300.0
CHANNEL_2_3_BOUNDARY_LATEST = 4150.0

GSR_LEVELS = [
    (61.5, "Gold-heavy / wait"),
    (60.0, "Initial silver rotation"),
    (58.5, "Silver leadership confirm"),
    (56.0, "First target"),
    (53.0, "Strong target"),
    (48.0, "Aggressive catch-up"),
]


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


@dataclass(frozen=True)
class ChannelModel:
    base_date: pd.Timestamp
    base_price: float
    latest_date: pd.Timestamp
    line0_latest: float
    spacing: float

    @property
    def slope_per_day(self) -> float:
        days = max((self.latest_date - self.base_date).days, 1)
        return (self.line0_latest - self.base_price) / days

    def line(self, level: int, dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
        date_index = pd.to_datetime(dates)
        day_offsets = (date_index - self.latest_date).days
        values = self.line0_latest + self.spacing * level + self.slope_per_day * day_offsets
        return pd.Series(values, index=date_index)


def _fetch_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "btcfloor/0.1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_lbma(url: str) -> pd.DataFrame:
    payload = _fetch_json(url)
    rows = [
        {"date": item["d"], "price": item["v"][0]}
        for item in payload
        if item.get("v") and item["v"][0] is not None
    ]
    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data["date"])
    data["price"] = pd.to_numeric(data["price"])
    return data.sort_values("date").reset_index(drop=True)


def _load_yahoo_futures(symbol: str, label: str) -> pd.DataFrame:
    response = requests.get(
        YAHOO_CHART_URL.format(symbol=symbol),
        params={"range": YAHOO_LIVE_RANGE, "interval": "1d"},
        headers=YAHOO_HEADERS,
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {symbol}: {chart['error']}")
    result = chart["result"][0]
    quote = result["indicators"]["quote"][0]
    timestamps = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    data = pd.DataFrame(
        {
            "date": timestamps.tz_convert(None).normalize(),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    data = data.dropna(subset=["close"]).copy()
    data["price"] = pd.to_numeric(data["close"])
    data["symbol"] = symbol
    data["label"] = label

    meta = result.get("meta", {})
    latest_price = meta.get("regularMarketPrice")
    latest_time = meta.get("regularMarketTime")
    if latest_price is not None and latest_time is not None:
        latest_date = (
            pd.to_datetime(int(latest_time), unit="s", utc=True)
            .tz_convert(None)
            .normalize()
        )
        existing = data["date"].eq(latest_date)
        if existing.any():
            data.loc[existing, "close"] = float(latest_price)
            data.loc[existing, "price"] = float(latest_price)
        elif data.empty or latest_date > data["date"].max():
            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        [
                            {
                                "date": latest_date,
                                "open": pd.NA,
                                "high": pd.NA,
                                "low": pd.NA,
                                "close": float(latest_price),
                                "volume": pd.NA,
                                "price": float(latest_price),
                                "symbol": symbol,
                                "label": label,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    return (
        data.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )


def _price_on_or_after(data: pd.DataFrame, date: pd.Timestamp, label: str) -> tuple[pd.Timestamp, float]:
    available = data[data["date"] >= date]
    if available.empty:
        raise ValueError(f"No {label} price available on or after {date.date()}")
    row = available.iloc[0]
    return pd.Timestamp(row["date"]), float(row["price"])


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _money_2(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _build_gold_analogs(lbma: pd.DataFrame) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    current_anchor_date, current_anchor_price = _price_on_or_after(lbma, CURRENT_ANCHOR, "gold")
    lines: list[pd.DataFrame] = []

    for name, anchor in GOLD_ANALOG_ANCHORS.items():
        anchor_date, anchor_price = _price_on_or_after(lbma, anchor, "gold")
        segment = lbma[
            (lbma["date"] >= anchor_date)
            & (lbma["date"] <= anchor_date + pd.Timedelta(days=GOLD_PROJECTION_DAYS))
        ].copy()
        segment["projection_date"] = current_anchor_date + (segment["date"] - anchor_date)
        segment["scaled_price"] = segment["price"] / anchor_price * current_anchor_price
        segment["series"] = name
        segment["source_date"] = segment["date"]
        segment["source_price"] = segment["price"]
        lines.append(
            segment[
                [
                    "projection_date",
                    "scaled_price",
                    "series",
                    "source_date",
                    "source_price",
                ]
            ]
        )

    current = lbma[
        (lbma["date"] >= current_anchor_date)
        & (lbma["date"] <= current_anchor_date + pd.Timedelta(days=GOLD_PROJECTION_DAYS))
    ].copy()
    current["projection_date"] = current["date"]
    current["scaled_price"] = current["price"]
    current["series"] = "2026 observed, LBMA PM fix"
    current["source_date"] = current["date"]
    current["source_price"] = current["price"]
    lines.append(
        current[
            ["projection_date", "scaled_price", "series", "source_date", "source_price"]
        ]
    )

    combined = pd.concat(lines, ignore_index=True)
    calendar = pd.date_range(
        current_anchor_date,
        current_anchor_date + pd.Timedelta(days=GOLD_PROJECTION_DAYS),
        freq="D",
    )
    average_frame = pd.DataFrame(index=calendar)
    for series in GOLD_ANALOG_ANCHORS:
        analog = combined[combined["series"].eq(series)].set_index("projection_date")[
            "scaled_price"
        ]
        average_frame[series] = analog.reindex(calendar).interpolate(method="time")
    average = average_frame.mean(axis=1).dropna().reset_index()
    average.columns = ["projection_date", "scaled_price"]
    average["series"] = "Analog average, 1973 + 2006"
    average["source_date"] = pd.NaT
    average["source_price"] = pd.NA

    combined = pd.concat([combined, average], ignore_index=True)
    return combined.sort_values(["series", "projection_date"]), current_anchor_price, current_anchor_date


def _build_channel_model(lbma: pd.DataFrame) -> ChannelModel:
    latest = lbma.iloc[-1]
    base_window = lbma[
        (lbma["date"] >= CHANNEL_BASE_WINDOW[0])
        & (lbma["date"] <= CHANNEL_BASE_WINDOW[1])
    ]
    base = base_window.loc[base_window["price"].idxmin()]
    line0_latest = CHANNEL_2_3_BOUNDARY_LATEST - 2 * CHANNEL_SPACING_USD
    return ChannelModel(
        base_date=pd.Timestamp(base["date"]),
        base_price=float(base["price"]),
        latest_date=pd.Timestamp(latest["date"]),
        line0_latest=line0_latest,
        spacing=CHANNEL_SPACING_USD,
    )


def _add_channel_zone(
    fig: go.Figure,
    x: pd.DatetimeIndex,
    lower: pd.Series,
    upper: pd.Series,
    name: str,
    fill_color: str,
    showlegend: bool = True,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower.values,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper.values,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=fill_color,
            name=name,
            showlegend=showlegend,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:$,.0f}<extra>" + name + "</extra>",
        )
    )


def _make_gold_chart(
    analogs: pd.DataFrame,
    lbma: pd.DataFrame,
    channel: ChannelModel,
    current_gold: pd.Series,
) -> go.Figure:
    latest = lbma.iloc[-1]
    projection_x = pd.date_range(
        CURRENT_ANCHOR,
        CURRENT_ANCHOR + pd.Timedelta(days=GOLD_PROJECTION_DAYS),
        freq="D",
    )
    fig = go.Figure()
    colors = {
        "1973 analog, LBMA PM fix": "#22a7f0",
        "2006 analog, LBMA PM fix": "#d62728",
        "2026 observed, LBMA PM fix": "#111111",
        "Analog average, 1973 + 2006": "#8a8a8a",
    }
    widths = {
        "1973 analog, LBMA PM fix": 2.5,
        "2006 analog, LBMA PM fix": 2.5,
        "2026 observed, LBMA PM fix": 4.0,
        "Analog average, 1973 + 2006": 2.0,
    }
    series_order = [
        "2026 observed, LBMA PM fix",
        "Analog average, 1973 + 2006",
        "2006 analog, LBMA PM fix",
        "1973 analog, LBMA PM fix",
    ]

    channel_2_low = channel.line(1, projection_x)
    channel_2_high = channel.line(2, projection_x)
    channel_3_high = channel.line(3, projection_x)
    channel_3_mid = (channel_2_high + channel_3_high) / 2
    _add_channel_zone(
        fig,
        projection_x,
        channel_2_high,
        channel_3_mid,
        "Peter target: lower Channel #3",
        "rgba(255, 165, 0, 0.16)",
    )
    _add_channel_zone(
        fig,
        projection_x,
        channel_2_low,
        channel_2_high,
        "Peter risk: Channel #2",
        "rgba(220, 20, 60, 0.13)",
    )

    for name in series_order:
        series = analogs[
            (analogs["series"].eq(name))
            & (analogs["projection_date"] >= CURRENT_ANCHOR)
            & (
                analogs["projection_date"]
                <= CURRENT_ANCHOR + pd.Timedelta(days=GOLD_PROJECTION_DAYS)
            )
        ]
        if series.empty:
            continue
        legend_name = name.replace(", LBMA PM fix", "")
        fig.add_trace(
            go.Scatter(
                x=series["projection_date"],
                y=series["scaled_price"],
                mode="lines",
                name=legend_name,
                line={"color": colors[name], "width": widths[name]},
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Scaled price: %{y:$,.0f}<extra>"
                    + legend_name
                    + "</extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[latest["date"]],
            y=[latest["price"]],
            mode="markers+text",
            name="Latest legacy LBMA PM fix",
            marker={"size": 9, "color": "#111111"},
            text=[f"LBMA {_money(float(latest['price']))}"],
            textposition="top left",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:$,.2f}<extra>Legacy LBMA</extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[current_gold["date"]],
            y=[current_gold["price"]],
            mode="markers+text",
            name=f"Live {GOLD_FUTURES_SYMBOL}",
            marker={"size": 10, "color": "#f97316", "symbol": "diamond"},
            text=[f"{GOLD_FUTURES_SYMBOL} {_money(float(current_gold['price']))}"],
            textposition="bottom right",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>%{y:$,.2f}<extra>"
                + f"Live {GOLD_FUTURES_SYMBOL}"
                + "</extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_vline(x=latest["date"], line_width=1, line_dash="dot", line_color="#555555")
    fig.add_vline(
        x=current_gold["date"],
        line_width=1,
        line_dash="dot",
        line_color="#f97316",
    )
    fig.update_layout(
        template="plotly_white",
        title=(
            "Gold analog context + live futures marker<br>"
            f"<sup>Analog lines use legacy LBMA PM history through {pd.Timestamp(latest['date']).date()}; "
            f"decision marker uses {GOLD_FUTURES_SYMBOL} through {pd.Timestamp(current_gold['date']).date()}</sup>"
        ),
        height=760,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(255,255,255,0.82)",
            "bordercolor": "rgba(0,0,0,0.08)",
            "borderwidth": 1,
        },
        margin={"l": 80, "r": 50, "t": 105, "b": 120},
    )
    fig.update_yaxes(title_text="Gold price, USD/oz", tickprefix="$")
    fig.update_xaxes(title_text="Projected 2026 date")
    return fig


def _make_gold_zoom_chart(
    analogs: pd.DataFrame,
    lbma: pd.DataFrame,
    channel: ChannelModel,
    current_gold: pd.Series,
) -> go.Figure:
    latest = lbma.iloc[-1]
    zoom_start = pd.Timestamp("2026-06-01")
    zoom_end = pd.Timestamp("2026-08-31")
    zoom_x = pd.date_range(zoom_start, zoom_end, freq="D")
    fig = go.Figure()
    colors = {
        "1973 analog, LBMA PM fix": "#22a7f0",
        "2006 analog, LBMA PM fix": "#d62728",
        "2026 observed, LBMA PM fix": "#111111",
        "Analog average, 1973 + 2006": "#8a8a8a",
    }
    widths = {
        "1973 analog, LBMA PM fix": 2.8,
        "2006 analog, LBMA PM fix": 2.8,
        "2026 observed, LBMA PM fix": 4.5,
        "Analog average, 1973 + 2006": 2.4,
    }
    series_order = [
        "2026 observed, LBMA PM fix",
        "Analog average, 1973 + 2006",
        "2006 analog, LBMA PM fix",
        "1973 analog, LBMA PM fix",
    ]

    channel_2_low = channel.line(1, zoom_x)
    channel_2_high = channel.line(2, zoom_x)
    channel_3_high = channel.line(3, zoom_x)
    channel_3_mid = (channel_2_high + channel_3_high) / 2
    _add_channel_zone(
        fig,
        zoom_x,
        channel_2_high,
        channel_3_mid,
        "Peter target: lower Channel #3",
        "rgba(255, 165, 0, 0.16)",
    )
    _add_channel_zone(
        fig,
        zoom_x,
        channel_2_low,
        channel_2_high,
        "Peter risk: Channel #2",
        "rgba(220, 20, 60, 0.13)",
    )
    for level in [1, 2, 3]:
        y = channel.line(level, zoom_x)
        fig.add_trace(
            go.Scatter(
                x=zoom_x,
                y=y.values,
                mode="lines",
                name=f"Channel boundary L{level}",
                line={"color": "#686868", "width": 1.5, "dash": "dot"},
                opacity=0.75,
                hovertemplate="%{x|%Y-%m-%d}<br>L"
                + str(level)
                + ": %{y:$,.0f}<extra></extra>",
                showlegend=False,
            )
        )

    for name in series_order:
        series = analogs[
            (analogs["series"].eq(name))
            & (analogs["projection_date"] >= zoom_start)
            & (analogs["projection_date"] <= zoom_end)
        ]
        if series.empty:
            continue
        legend_name = name.replace(", LBMA PM fix", "")
        fig.add_trace(
            go.Scatter(
                x=series["projection_date"],
                y=series["scaled_price"],
                mode="lines",
                name=legend_name,
                line={"color": colors[name], "width": widths[name]},
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Scaled price: %{y:$,.0f}<extra>"
                    + legend_name
                    + "</extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[latest["date"]],
            y=[latest["price"]],
            mode="markers+text",
            name="Latest legacy LBMA PM fix",
            marker={"size": 9, "color": "#111111"},
            text=[f"LBMA {_money(float(latest['price']))}"],
            textposition="top right",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:$,.2f}<extra>Legacy LBMA</extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[current_gold["date"]],
            y=[current_gold["price"]],
            mode="markers+text",
            name=f"Live {GOLD_FUTURES_SYMBOL}",
            marker={"size": 10, "color": "#f97316", "symbol": "diamond"},
            text=[f"{GOLD_FUTURES_SYMBOL} {_money(float(current_gold['price']))}"],
            textposition="bottom right",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>%{y:$,.2f}<extra>"
                + f"Live {GOLD_FUTURES_SYMBOL}"
                + "</extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_vline(x=latest["date"], line_width=1, line_dash="dot", line_color="#555555")
    fig.add_vline(
        x=current_gold["date"],
        line_width=1,
        line_dash="dot",
        line_color="#f97316",
    )
    fig.update_layout(
        template="plotly_white",
        title=(
            "Gold June-August zoom<br>"
            f"<sup>Analog context uses legacy LBMA history; live marker uses "
            f"{GOLD_FUTURES_SYMBOL} through {pd.Timestamp(current_gold['date']).date()}</sup>"
        ),
        height=620,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.17,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(255,255,255,0.82)",
            "bordercolor": "rgba(0,0,0,0.08)",
            "borderwidth": 1,
        },
        margin={"l": 80, "r": 50, "t": 100, "b": 115},
    )
    fig.update_yaxes(title_text="Gold price, USD/oz", tickprefix="$")
    fig.update_xaxes(title_text="Projected 2026 date")
    return fig


def _build_silver_analogs(lbma: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    current_anchor_date, current_anchor_price = _price_on_or_after(lbma, CURRENT_ANCHOR, "silver")
    lines: list[pd.DataFrame] = []

    for name, anchor in SILVER_ANALOG_ANCHORS.items():
        anchor_date, anchor_price = _price_on_or_after(lbma, anchor, "silver")
        segment = lbma[
            (lbma["date"] >= anchor_date)
            & (lbma["date"] <= anchor_date + pd.Timedelta(days=SILVER_PROJECTION_DAYS))
        ].copy()
        segment["projection_date"] = current_anchor_date + (segment["date"] - anchor_date)
        segment["scaled_price"] = segment["price"] / anchor_price * current_anchor_price
        segment["series"] = name
        segment["source_date"] = segment["date"]
        segment["source_price"] = segment["price"]
        segment["anchor_date"] = anchor_date
        segment["anchor_price"] = anchor_price
        lines.append(segment)

    current = lbma[
        (lbma["date"] >= current_anchor_date)
        & (lbma["date"] <= current_anchor_date + pd.Timedelta(days=SILVER_PROJECTION_DAYS))
    ].copy()
    current["projection_date"] = current["date"]
    current["scaled_price"] = current["price"]
    current["series"] = "2026"
    current["source_date"] = current["date"]
    current["source_price"] = current["price"]
    current["anchor_date"] = current_anchor_date
    current["anchor_price"] = current_anchor_price
    lines.append(current)

    combined = pd.concat(lines, ignore_index=True)
    return (
        combined[
            [
                "projection_date",
                "scaled_price",
                "series",
                "source_date",
                "source_price",
                "anchor_date",
                "anchor_price",
            ]
        ].sort_values(["series", "projection_date"]),
        current_anchor_date,
        current_anchor_price,
    )


def _make_silver_chart(
    analogs: pd.DataFrame,
    latest: pd.Series,
    current_silver: pd.Series,
) -> go.Figure:
    colors = {
        "1974": "#f28e2b",
        "1980": "#8ab34f",
        "2004": "#7560a8",
        "2006": "#45b5c4",
        "2011": "#c94c4c",
        "2008": "#264e75",
        "2026": "#111111",
    }
    fig = go.Figure()
    for series in ["1974", "1980", "2004", "2006", "2011", "2008", "2026"]:
        data = analogs[analogs["series"].eq(series)]
        fig.add_trace(
            go.Scatter(
                x=data["projection_date"],
                y=data["scaled_price"],
                mode="lines",
                name=series,
                line={"color": colors[series], "width": 4 if series == "2026" else 2.5},
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Scaled silver: %{y:$,.2f}<extra>"
                    + series
                    + "</extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[latest["date"]],
            y=[latest["price"]],
            mode="markers+text",
            name="Latest legacy LBMA silver",
            marker={"size": 9, "color": "#111111"},
            text=[f"LBMA {_money_2(float(latest['price']))}"],
            textposition="top right",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:$,.3f}<extra>Legacy LBMA</extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[current_silver["date"]],
            y=[current_silver["price"]],
            mode="markers+text",
            name=f"Live {SILVER_FUTURES_SYMBOL}",
            marker={"size": 10, "color": "#f97316", "symbol": "diamond"},
            text=[f"{SILVER_FUTURES_SYMBOL} {_money_2(float(current_silver['price']))}"],
            textposition="bottom right",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>%{y:$,.3f}<extra>"
                + f"Live {SILVER_FUTURES_SYMBOL}"
                + "</extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_vline(x=latest["date"], line_width=1, line_dash="dot", line_color="#777777")
    fig.add_vline(
        x=current_silver["date"],
        line_width=1,
        line_dash="dot",
        line_color="#f97316",
    )
    fig.update_layout(
        template="plotly_white",
        title=(
            "Silver correction analog context + live futures marker<br>"
            f"<sup>Analog lines use legacy LBMA silver history through {pd.Timestamp(latest['date']).date()}; "
            f"decision marker uses {SILVER_FUTURES_SYMBOL} through {pd.Timestamp(current_silver['date']).date()}</sup>"
        ),
        height=760,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.12,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255,255,255,0.86)",
            "bordercolor": "rgba(0,0,0,0.08)",
            "borderwidth": 1,
        },
        margin={"l": 80, "r": 50, "t": 100, "b": 115},
    )
    fig.update_xaxes(title_text="Projected 2026 date")
    fig.update_yaxes(title_text="Silver price, USD/oz", tickprefix="$", range=[20, 125])
    return fig


def _build_gsr(gold: pd.DataFrame, silver: pd.DataFrame) -> pd.DataFrame:
    merged = gold[["date", "price"]].rename(columns={"price": "gold_usd"}).merge(
        silver[["date", "price"]].rename(columns={"price": "silver_usd"}),
        on="date",
        how="inner",
    )
    merged["gsr"] = merged["gold_usd"] / merged["silver_usd"]
    merged["gsr_sma20"] = merged["gsr"].rolling(20).mean()
    merged["gsr_sma50"] = merged["gsr"].rolling(50).mean()
    merged["gold_100"] = merged["gold_usd"] / merged["gold_usd"].iloc[0] * 100
    merged["silver_100"] = merged["silver_usd"] / merged["silver_usd"].iloc[0] * 100
    return merged


def _gsr_state(gsr: float, sma20: float, sma50: float) -> str:
    if gsr <= 48:
        level = "aggressive silver catch-up target; trim/rebalance zone"
    elif gsr <= 53:
        level = "strong silver outperformance target"
    elif gsr <= 56:
        level = "first silver target / reassess zone"
    elif gsr <= 58.5:
        level = "silver leadership confirmed"
    elif gsr <= 60:
        level = "initial rotation zone"
    elif gsr <= 61.5:
        level = "watch zone; gold still acceptable unless 60 breaks"
    else:
        level = "gold-heavy regime; wait for silver confirmation"

    trend = (
        "20D below 50D, ratio trend supports silver"
        if pd.notna(sma20) and pd.notna(sma50) and sma20 < sma50
        else "ratio trend not yet cleanly silver-positive"
    )
    return f"{level}; {trend}"


def _make_gsr_chart(gsr: pd.DataFrame, source_label: str) -> go.Figure:
    latest = gsr.iloc[-1]
    latest_date = pd.Timestamp(latest["date"])
    view_start = max(latest_date - pd.Timedelta(days=365), gsr["date"].min())
    view = gsr[gsr["date"] >= view_start].copy()
    chart = gsr.copy()
    chart["gold_view_100"] = float("nan")
    chart["silver_view_100"] = float("nan")
    if not view.empty:
        base = view.iloc[0]
        visible_mask = chart["date"] >= view_start
        chart.loc[visible_mask, "gold_view_100"] = (
            chart.loc[visible_mask, "gold_usd"] / float(base["gold_usd"]) * 100
        )
        chart.loc[visible_mask, "silver_view_100"] = (
            chart.loc[visible_mask, "silver_usd"] / float(base["silver_usd"]) * 100
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.72, 0.28],
        subplot_titles=[
            "Gold/silver ratio decision map",
            f"Gold and silver indexed to 100 from the current view start ({view_start.date()})",
        ],
    )
    y_max = max(64.5, float(view["gsr"].max()) + 1.0)

    zones = [
        (61.5, y_max, "Gold-heavy", "rgba(100,116,139,0.10)"),
        (60.0, 61.5, "Watch", "rgba(245,158,11,0.10)"),
        (58.5, 60.0, "Initial silver rotation", "rgba(34,197,94,0.10)"),
        (56.0, 58.5, "Confirmed silver leadership", "rgba(16,185,129,0.12)"),
        (53.0, 56.0, "First target", "rgba(14,165,233,0.11)"),
        (48.0, 53.0, "Strong target", "rgba(124,58,237,0.10)"),
    ]
    for y0, y1, label, color in zones:
        fig.add_hrect(
            y0=y0,
            y1=y1,
            fillcolor=color,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=gsr["date"],
            y=gsr["gsr"],
            name="GSR",
            line={"color": "#111111", "width": 2.3},
            hovertemplate="%{x|%Y-%m-%d}<br>GSR %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=gsr["date"],
            y=gsr["gsr_sma20"],
            name="20D GSR",
            line={"color": "#2563eb", "width": 1.3},
            hovertemplate="%{x|%Y-%m-%d}<br>20D %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=gsr["date"],
            y=gsr["gsr_sma50"],
            name="50D GSR",
            line={"color": "#7c3aed", "width": 1.3},
            hovertemplate="%{x|%Y-%m-%d}<br>50D %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for level, label in GSR_LEVELS:
        fig.add_hline(
            y=level,
            line={"color": "#475569", "width": 0.9, "dash": "dash"},
            annotation_text=f"{level:g}: {label}",
            annotation_position="right",
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=chart["date"],
            y=chart["gold_view_100"],
            name="Gold indexed",
            line={"color": "#d97706", "width": 1.6},
            hovertemplate="%{x|%Y-%m-%d}<br>Gold index %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart["date"],
            y=chart["silver_view_100"],
            name="Silver indexed",
            line={"color": "#64748b", "width": 1.6},
            hovertemplate="%{x|%Y-%m-%d}<br>Silver index %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_annotation(
        x=latest["date"],
        y=latest["gsr"],
        text=f"Latest {latest['gsr']:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=45,
        ay=-35,
        bgcolor="rgba(255,255,255,0.86)",
        bordercolor="#d0d7de",
        borderwidth=1,
        row=1,
        col=1,
    )
    fig.update_layout(
        template="plotly_white",
        title=(
            "Daily GSR monitor with rotation levels<br>"
            f"<sup>{source_label}: latest shared date {pd.Timestamp(latest['date']).date()}, "
            f"GSR {latest['gsr']:.2f}</sup>"
        ),
        height=780,
        hovermode="x unified",
        dragmode="pan",
        legend={"orientation": "h", "y": 1.05, "x": 0.01, "font": {"size": 10}},
        margin={"l": 75, "r": 90, "t": 105, "b": 55},
    )
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
    fig.update_xaxes(range=[view_start, latest_date + pd.Timedelta(days=20)])
    gsr_range = _linear_range(
        pd.concat(
            [
                view["gsr"],
                view["gsr_sma20"],
                view["gsr_sma50"],
                pd.Series([48.0, 53.0, 56.0, 58.5, 60.0, 61.5]),
            ],
            ignore_index=True,
        )
    )
    index_range = _linear_range(
        chart.loc[chart["date"] >= view_start, ["gold_view_100", "silver_view_100"]].stack()
    )
    fig.update_yaxes(title_text="Gold / silver", range=gsr_range, row=1, col=1)
    fig.update_yaxes(title_text="Index", range=index_range, row=2, col=1)
    return fig


def _plotly_div(fig: go.Figure, include_plotlyjs: bool | str = False) -> str:
    return fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        config={"responsive": True, "displaylogo": False},
        default_width="100%",
        default_height=f"{int(fig.layout.height or 760)}px",
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
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    {dashboard_nav("metals")}
  </header>
  <section class="commentary"><strong>Agent commentary:</strong> {commentary}</section>
  <main class="chart-wrap">{body}</main>
</body>
</html>
"""


def _write_summary(
    path: Path,
    gold: pd.DataFrame,
    silver: pd.DataFrame,
    gsr: pd.DataFrame,
    source_label: str,
) -> dict[str, object]:
    latest_gold = gold.iloc[-1]
    latest_silver = silver.iloc[-1]
    latest_gsr = gsr.iloc[-1]
    summary = {
        "sources": {
            "primary": {
                "label": source_label,
                "gold_symbol": GOLD_FUTURES_SYMBOL,
                "gold_url": YAHOO_CHART_URL.format(symbol=GOLD_FUTURES_SYMBOL),
                "silver_symbol": SILVER_FUTURES_SYMBOL,
                "silver_url": YAHOO_CHART_URL.format(symbol=SILVER_FUTURES_SYMBOL),
                "note": "Primary metals/GSR decision monitor uses current COMEX futures bars.",
            },
            "legacy_analog_context": {
                "gold_pm": GOLD_PM_URL,
                "silver": SILVER_URL,
                "note": "LBMA is retained only for long-history analog panels.",
            },
        },
        "latest": {
            "gold": {
                "date": f"{pd.Timestamp(latest_gold['date']):%Y-%m-%d}",
                "value": float(latest_gold["price"]),
                "symbol": GOLD_FUTURES_SYMBOL,
                "source": source_label,
            },
            "silver": {
                "date": f"{pd.Timestamp(latest_silver['date']):%Y-%m-%d}",
                "value": float(latest_silver["price"]),
                "symbol": SILVER_FUTURES_SYMBOL,
                "source": source_label,
            },
            "gsr": {
                "date": f"{pd.Timestamp(latest_gsr['date']):%Y-%m-%d}",
                "value": float(latest_gsr["gsr"]),
                "sma20": float(latest_gsr["gsr_sma20"]),
                "sma50": float(latest_gsr["gsr_sma50"]),
                "source": source_label,
            },
        },
        "decision_levels": [
            {"gsr": level, "label": label}
            for level, label in GSR_LEVELS
        ],
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_metals_dashboard(paths: ProjectPaths) -> list[Path]:
    paths.ensure_dirs()
    live_gold = _load_yahoo_futures(GOLD_FUTURES_SYMBOL, "Gold futures")
    live_silver = _load_yahoo_futures(SILVER_FUTURES_SYMBOL, "Silver futures")
    legacy_gold = _load_lbma(GOLD_PM_URL)
    legacy_silver = _load_lbma(SILVER_URL)
    gsr = _build_gsr(live_gold, live_silver)
    gold_analogs, _, _ = _build_gold_analogs(legacy_gold)
    silver_analogs, _, _ = _build_silver_analogs(legacy_silver)
    channel = _build_channel_model(legacy_gold)
    current_gold = live_gold.iloc[-1]
    current_silver = live_silver.iloc[-1]

    live_gold.to_csv(paths.processed_dir / "yahoo_gold_futures.csv", index=False)
    live_silver.to_csv(paths.processed_dir / "yahoo_silver_futures.csv", index=False)
    legacy_gold.to_csv(paths.processed_dir / "lbma_gold_pm.csv", index=False)
    legacy_silver.to_csv(paths.processed_dir / "lbma_silver.csv", index=False)
    gsr.to_csv(paths.processed_dir / "metals_gsr_daily.csv", index=False)
    gold_analogs.to_csv(paths.processed_dir / "gold_analog_projection.csv", index=False)
    silver_analogs.to_csv(paths.processed_dir / "silver_analog_projection.csv", index=False)

    summary = _write_summary(
        paths.report_dir / "metals_relative_summary.json",
        live_gold,
        live_silver,
        gsr,
        PRIMARY_SOURCE_LABEL,
    )
    latest = summary["latest"]
    gsr_latest = latest["gsr"]
    gsr_state = _gsr_state(
        float(gsr_latest["value"]),
        float(gsr_latest["sma20"]),
        float(gsr_latest["sma50"]),
    )
    commentary = (
        f"Primary live source is {PRIMARY_SOURCE_LABEL}: "
        f"{GOLD_FUTURES_SYMBOL} {_money(float(latest['gold']['value']))} "
        f"on {latest['gold']['date']}, {SILVER_FUTURES_SYMBOL} "
        f"{_money_2(float(latest['silver']['value']))} on {latest['silver']['date']}. "
        f"Latest shared GSR is "
        f"{float(gsr_latest['value']):.2f} on {gsr_latest['date']}; "
        f"20D {float(gsr_latest['sma20']):.2f}, 50D {float(gsr_latest['sma50']):.2f}. "
        f"Read: {gsr_state}. Treat 60 as the first rotation trigger, "
        f"58.5 as weekly leadership confirmation, and 56/53/48 as silver outperformance targets. "
        f"The gold/silver analog panels retain LBMA only as long-history context, not as the live decision feed."
    )

    body = "\n".join(
        [
            _section(
                "Daily GSR monitor",
                (
                    "Primary switch tool. Falling GSR means silver is outperforming gold; "
                    "levels mark rotate, confirm, and trim/reassess zones."
                ),
                _plotly_div(_make_gsr_chart(gsr, PRIMARY_SOURCE_LABEL), include_plotlyjs=True),
            ),
            _section(
                "Gold analog monitor",
                (
                    "Long-history analog context rebuilt from local source data, with a live "
                    f"{GOLD_FUTURES_SYMBOL} marker added so it is not mistaken for the decision feed."
                ),
                _plotly_div(_make_gold_chart(gold_analogs, legacy_gold, channel, current_gold)),
            ),
            _section(
                "Gold June-August zoom",
                "Short-window view for whether gold trend is still following the analog path.",
                _plotly_div(
                    _make_gold_zoom_chart(gold_analogs, legacy_gold, channel, current_gold)
                ),
            ),
            _section(
                "Silver analog monitor",
                (
                    "Silver correction analogs scaled from the 2026 peak. Use this with live GSR; "
                    "raw silver upside matters only if it outperforms gold."
                ),
                _plotly_div(
                    _make_silver_chart(silver_analogs, legacy_silver.iloc[-1], current_silver)
                ),
            ),
        ]
    )

    dashboard = paths.interactive_dir / "metals_relative_dashboard.html"
    dashboard.write_text(
        _shell("Metals Relative Strength Dashboard", commentary, body),
        encoding="utf-8",
    )
    return [
        paths.processed_dir / "yahoo_gold_futures.csv",
        paths.processed_dir / "yahoo_silver_futures.csv",
        paths.processed_dir / "lbma_gold_pm.csv",
        paths.processed_dir / "lbma_silver.csv",
        paths.processed_dir / "metals_gsr_daily.csv",
        paths.processed_dir / "gold_analog_projection.csv",
        paths.processed_dir / "silver_analog_projection.csv",
        paths.report_dir / "metals_relative_summary.json",
        dashboard,
    ]


def main() -> None:
    for output in build_metals_dashboard(ProjectPaths.from_cwd()):
        print(output)


if __name__ == "__main__":
    main()
