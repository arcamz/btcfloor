from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from btcfloor.data import load_price_history
from btcfloor.expectile import expectile_model_name, fit_expectile_power_law
from btcfloor.paths import ProjectPaths
from btcfloor.powerlaw import giovanni_power_law_floor_model


WINDOW_DAYS = 72
RECENT_WICK_LOW_USD = 66_500.0
BROKEN_SUPPORT_USD = 68_850.0
FIRST_RANGE_RECLAIM_USD = 73_600.0


def money_k(value: float) -> str:
    return f"${value / 1000:.1f}k"


def add_daily_candles(ax: plt.Axes, frame: pd.DataFrame) -> None:
    width = 0.58
    for row in frame.itertuples(index=False):
        x = mdates.date2num(pd.Timestamp(row.date))
        open_price = float(row.open)
        close_price = float(row.close)
        high_price = float(row.high)
        low_price = float(row.low)
        up = close_price >= open_price
        color = "#111827" if not up else "#f9fafb"
        edge = "#111827"
        ax.vlines(x, low_price, high_price, color=edge, linewidth=0.75)
        body_bottom = min(open_price, close_price)
        body_height = max(abs(close_price - open_price), 45.0)
        ax.add_patch(
            Rectangle(
                (x - width / 2, body_bottom),
                width,
                body_height,
                facecolor=color,
                edgecolor=edge,
                linewidth=0.75,
            )
        )


def build_daily_ohlc_from_close(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.loc[:, ["date", "price_usd"]].copy().sort_values("date")
    frame["open"] = frame["price_usd"].shift(1).fillna(frame["price_usd"])
    frame["close"] = frame["price_usd"]
    frame["high"] = frame[["open", "close"]].max(axis=1)
    frame["low"] = frame[["open", "close"]].min(axis=1)
    return frame.loc[:, ["date", "open", "high", "low", "close"]]


def write_plot(paths: ProjectPaths) -> pd.DataFrame:
    daily = load_price_history(paths.processed_btc_csv)
    ohlc = build_daily_ohlc_from_close(daily)
    ohlc["sma50"] = daily["price_usd"].rolling(50).mean().to_numpy()
    ohlc["sma200"] = daily["price_usd"].rolling(200).mean().to_numpy()

    latest = daily.iloc[-1]
    latest_date = pd.Timestamp(latest["date"])
    spot_price = float(latest["price_usd"])
    view = ohlc.loc[ohlc["date"] >= latest_date - pd.Timedelta(days=WINDOW_DAYS)].copy()

    weekly_source = daily.resample("W-SUN", on="date").last().dropna().reset_index()
    expectile = fit_expectile_power_law(
        weekly_source.rename(columns={"price_usd": "price_usd"}),
        tau=0.0001,
        name=expectile_model_name(0.0001),
    )
    hard_floor = giovanni_power_law_floor_model()
    latest_hard_floor = float(hard_floor.predict_price(latest_date, floor=True)[0])
    latest_expectile = float(expectile.predict_price(latest_date, floor=True)[0])
    latest_sma50 = float(view["sma50"].dropna().iloc[-1])
    latest_sma200 = float(view["sma200"].dropna().iloc[-1])

    levels = [
        ("200D SMA repair", latest_sma200, "#7c3aed", "wait/confirm"),
        ("50D SMA repair", latest_sma50, "#2563eb", "momentum improves"),
        ("range repair", FIRST_RANGE_RECLAIM_USD, "#0284c7", "add on strength"),
        ("failed-breakdown reclaim", BROKEN_SUPPORT_USD, "#f59e0b", "SFP trigger"),
        ("latest close", spot_price, "#111827", "current"),
        ("recent wick low", RECENT_WICK_LOW_USD, "#9ca3af", "sweep level"),
        ("0.01% expectile", latest_expectile, "#1b9e77", "value band"),
        ("hard floor", latest_hard_floor, "#d95f02", "extreme value"),
    ]
    metrics = pd.DataFrame(
        [
            {
                "as_of_date": latest_date,
                "spot_price_usd": spot_price,
                "level": label,
                "level_usd": value,
                "gap_pct": spot_price / value - 1.0,
                "action_context": context,
            }
            for label, value, _, context in levels
        ]
    )
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(paths.report_dir / "tactical_trigger_strip_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    add_daily_candles(ax, view)

    for label, value, color, context in levels:
        linewidth = 2.0 if label == "latest close" else 1.15
        linestyle = "-" if label in {"latest close", "0.01% expectile", "hard floor"} else "--"
        alpha = 1.0 if label in {"latest close", "failed-breakdown reclaim", "0.01% expectile", "hard floor"} else 0.68
        ax.axhline(value, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        ax.text(
            latest_date + pd.Timedelta(days=3),
            value,
            f"{label} {money_k(value)} - {context}",
            color=color,
            fontsize=9,
            va="center",
            ha="left",
        )

    ax.text(
        view["date"].iloc[0],
        max(view["high"].max(), latest_sma200) * 1.015,
        "Tactical trigger strip",
        fontsize=17,
        color="#111827",
        ha="left",
    )
    ax.text(
        view["date"].iloc[0],
        max(view["high"].max(), latest_sma200) * 1.002,
        "Value is here; confirmation requires a failed breakdown reclaim or momentum repair.",
        fontsize=10,
        color="#4b5563",
        ha="left",
    )

    ax.annotate(
        "preferred value trigger:\nsweep low, reclaim $68.8k",
        xy=(latest_date, BROKEN_SUPPORT_USD),
        xytext=(latest_date - pd.Timedelta(days=22), BROKEN_SUPPORT_USD + 3_200),
        arrowprops={"arrowstyle": "->", "color": "#f59e0b", "linewidth": 1.0},
        color="#92400e",
        fontsize=9.5,
        ha="left",
    )

    ax.set_xlim(view["date"].iloc[0] - pd.Timedelta(days=2), latest_date + pd.Timedelta(days=32))
    ymin = min(latest_hard_floor, RECENT_WICK_LOW_USD, view["low"].min()) * 0.975
    ymax = max(latest_sma200, view["high"].max()) * 1.035
    ax.set_ylim(ymin, ymax)
    ax.text(
        view["date"].iloc[0],
        ymin + (ymax - ymin) * 0.012,
        "Candle bodies are daily close-to-close; wick/support levels are manual trading references.",
        fontsize=8.5,
        color="#6b7280",
        ha="left",
        va="bottom",
    )
    ax.yaxis.set_major_formatter(lambda value, _: money_k(value))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(axis="both", colors="#374151", labelsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")

    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    output = paths.figure_dir / "tactical_trigger_strip.png"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    result = write_plot(ProjectPaths.from_cwd())
    print("Wrote reports/figures/tactical_trigger_strip.png")
    print(result.to_string(index=False))
