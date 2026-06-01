from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcfloor.forward_floor import future_floor_overlap_daily
from btcfloor.paths import ProjectPaths
from btcfloor.powerlaw import giovanni_power_law_floor_model


START_DATE = pd.Timestamp("2013-01-01")
HORIZON_MONTHS = 12


def observed_peak_between(
    daily: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.Series:
    window = daily.loc[
        daily["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ]
    if window.empty:
        raise ValueError(f"No price rows between {start_date} and {end_date}")
    return window.loc[window["price_usd"].idxmax()]


def build_cycle_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    model = giovanni_power_law_floor_model()
    overlap = future_floor_overlap_daily(daily, model, horizon_months=HORIZON_MONTHS)
    latest_date = pd.Timestamp(daily["date"].max())
    cycle_specs = [
        ("2013->2015", "2011-11-18", "2015-01-14", pd.Timestamp("2015-01-14"), True),
        ("2017->2018", "2015-01-14", "2018-12-15", pd.Timestamp("2018-12-15"), True),
        ("2021->2022", "2018-12-15", "2022-11-21", pd.Timestamp("2022-11-21"), True),
        ("2025->2026e", "2022-11-21", "2026-10-19", pd.Timestamp("2026-10-19"), False),
    ]

    rows = []
    for label, start_date, peak_search_end, low_date, fully_observed_cycle in cycle_specs:
        observed_end = min(low_date, latest_date)
        peak = observed_peak_between(daily, start_date, str(observed_end.date()))
        peak_date = pd.Timestamp(peak["date"])
        peak_price = float(peak["price_usd"])
        peak_floor = float(model.predict_price(peak_date, floor=True)[0])
        peak_forward_floor = float(
            model.predict_price(peak_date + pd.DateOffset(months=HORIZON_MONTHS), floor=True)[
                0
            ]
        )

        post_peak_signal = overlap.loc[
            overlap["date"].between(peak_date, observed_end)
            & overlap["below_future_floor"].astype(bool)
        ]
        if post_peak_signal.empty:
            continue

        entry = post_peak_signal.iloc[0]
        entry_date = pd.Timestamp(entry["date"])
        entry_price = float(entry["price_usd"])
        entry_future_floor = float(entry["future_floor_usd"])
        entry_ratio = float(entry["ratio_to_future_floor"])
        opportunity_window = daily.loc[
            daily["date"].between(entry_date, observed_end)
        ].copy()
        cheapest = opportunity_window.loc[opportunity_window["price_usd"].idxmin()]
        cheapest_date = pd.Timestamp(cheapest["date"])
        cheapest_price = float(cheapest["price_usd"])
        bear_days = int((low_date - peak_date).days)
        days_peak_to_entry = int((entry_date - peak_date).days)
        days_entry_to_low = int((low_date - entry_date).days)
        lower_buy_pct = cheapest_price / entry_price - 1.0

        rows.append(
            {
                "cycle": label,
                "peak_date": peak_date,
                "peak_price_usd": peak_price,
                "peak_to_floor": peak_price / peak_floor,
                "peak_to_forward_floor": peak_price / peak_forward_floor,
                "bear_low_date": low_date,
                "observed_through_date": observed_end,
                "entry_date": entry_date,
                "entry_price_usd": entry_price,
                "entry_future_floor_usd": entry_future_floor,
                "entry_ratio_to_future_floor": entry_ratio,
                "cheapest_date": cheapest_date,
                "cheapest_price_usd": cheapest_price,
                "lower_buy_pct": lower_buy_pct,
                "days_peak_to_entry": days_peak_to_entry,
                "bear_days": bear_days,
                "bear_elapsed_pct": days_peak_to_entry / bear_days,
                "days_entry_to_low": days_entry_to_low,
                "days_entry_to_cheapest": int((cheapest_date - entry_date).days),
                "entry_drawdown_from_peak": entry_price / peak_price - 1.0,
                "fully_observed": bool(fully_observed_cycle and low_date <= latest_date),
            }
        )

    return pd.DataFrame(rows)


def write_plot(paths: ProjectPaths) -> Path:
    daily = pd.read_csv(paths.processed_btc_csv, parse_dates=["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    model = giovanni_power_law_floor_model()
    overlap = future_floor_overlap_daily(daily, model, horizon_months=HORIZON_MONTHS)
    metrics = build_cycle_metrics(daily)

    plot_frame = overlap.loc[overlap["date"] >= START_DATE].copy()
    dates = plot_frame["date"]
    ratio = plot_frame["ratio_to_future_floor"].to_numpy(dtype=float)
    future_floor = plot_frame["future_floor_usd"].to_numpy(dtype=float)
    price = plot_frame["price_usd"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(21, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[2.2, 1.0, 1.35])
    ax_price = fig.add_subplot(gs[0, :])
    ax_ratio = fig.add_subplot(gs[1, :], sharex=ax_price)
    ax_peak = fig.add_subplot(gs[2, 0])
    ax_timing = fig.add_subplot(gs[2, 1])
    ax_wait = fig.add_subplot(gs[2, 2])

    for row in metrics.itertuples(index=False):
        end = pd.Timestamp(row.bear_low_date)
        ax_price.axvspan(row.peak_date, end, color="#dce9f7", alpha=0.45, linewidth=0)
        ax_ratio.axvspan(row.peak_date, end, color="#dce9f7", alpha=0.45, linewidth=0)
        ax_price.annotate(
            f"{row.cycle}\npeak {row.peak_to_floor:.1f}x floor\nbreach at {row.bear_elapsed_pct:.0%}",
            (row.entry_date, row.entry_price_usd),
            xytext=(22, -6),
            textcoords="offset points",
            fontsize=9,
            color="#374151",
            ha="left",
            va="top",
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": 0.8},
        )

    ax_price.plot(dates, price, color="#18212f", linewidth=1.45, label="BTC daily close")
    ax_price.plot(
        dates,
        future_floor,
        color="#d94841",
        linewidth=1.8,
        label="Giovanni floor 12m forward",
    )
    ax_price.scatter(
        metrics["peak_date"],
        metrics["peak_price_usd"],
        marker="^",
        s=75,
        color="#111827",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="cycle peak",
    )
    ax_price.scatter(
        metrics["entry_date"],
        metrics["entry_price_usd"],
        marker="v",
        s=90,
        color="#d94841",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="first post-peak breach",
    )
    ax_price.scatter(
        metrics["cheapest_date"],
        metrics["cheapest_price_usd"],
        marker="o",
        s=58,
        color="#2563a6",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="cheapest later price before low/window end",
    )
    ax_price.set_yscale("log")
    ax_price.set_title(
        "Is 2026 different? Subdued peak, earlier future-floor breach, and cheaper-buy window",
        fontsize=16,
        pad=12,
    )
    ax_price.set_ylabel("USD, log scale")
    ax_price.grid(True, which="both", alpha=0.22)
    ax_price.legend(loc="upper left", frameon=False, ncols=3)

    ax_ratio.plot(dates, ratio, color="#2563a6", linewidth=1.25)
    ax_ratio.axhline(1.0, color="#d94841", linestyle="--", linewidth=1.15)
    ax_ratio.fill_between(
        dates,
        ratio,
        1.0,
        where=ratio < 1.0,
        color="#d94841",
        alpha=0.20,
        interpolate=True,
    )
    ax_ratio.set_ylim(0.35, max(2.4, float(np.nanpercentile(ratio, 88))))
    ax_ratio.set_ylabel("Price / 12m-forward floor")
    ax_ratio.grid(True, alpha=0.22)
    ax_ratio.set_title("Below 1.0 = price has crossed under the future floor", fontsize=12)

    labels = metrics["cycle"].tolist()
    x = np.arange(len(metrics))
    bar_colors = ["#506f8f", "#506f8f", "#506f8f", "#d94841"]
    ax_peak.bar(x, metrics["peak_to_floor"], color=bar_colors, width=0.68)
    ax_peak.set_xticks(x)
    ax_peak.set_xticklabels(labels, rotation=20, ha="right")
    ax_peak.set_ylabel("Peak / same-day floor")
    ax_peak.set_title("Peak extension was much smaller")
    ax_peak.grid(True, axis="y", alpha=0.22)
    for idx, value in enumerate(metrics["peak_to_floor"]):
        ax_peak.text(idx, value * 1.03, f"{value:.1f}x", ha="center", fontsize=9)

    ax_timing.bar(
        x,
        metrics["bear_elapsed_pct"] * 100.0,
        color=bar_colors,
        width=0.68,
        label="bear elapsed at breach",
    )
    ax_timing.set_xticks(x)
    ax_timing.set_xticklabels(labels, rotation=20, ha="right")
    ax_timing.set_ylim(0, 110)
    ax_timing.set_ylabel("% of peak-to-low window")
    ax_timing.set_title("First breach came earlier in 2026")
    ax_timing.grid(True, axis="y", alpha=0.22)
    for idx, row in metrics.iterrows():
        ax_timing.text(
            idx,
            row["bear_elapsed_pct"] * 100.0 + 3,
            f"{row['bear_elapsed_pct']:.0%}\n{int(row['days_entry_to_low'])}d left",
            ha="center",
            fontsize=9,
        )

    wait_values = metrics["lower_buy_pct"] * 100.0
    wait_colors = np.where(metrics["fully_observed"], "#b42318", "#94a3b8")
    ax_wait.bar(x, wait_values, color=wait_colors, width=0.68)
    ax_wait.axhline(0.0, color="#111827", linewidth=0.8)
    ax_wait.set_xticks(x)
    ax_wait.set_xticklabels(labels, rotation=20, ha="right")
    ax_wait.set_ylabel("Cheapest later buy vs breach")
    ax_wait.set_title("Did waiting after breach help?")
    ax_wait.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax_wait.grid(True, axis="y", alpha=0.22)
    y_min = min(-5.0, float(wait_values.min()) - 8.0)
    ax_wait.set_ylim(y_min, 5.0)
    for idx, row in metrics.iterrows():
        suffix = "*" if not row["fully_observed"] else ""
        value = float(row["lower_buy_pct"]) * 100.0
        ax_wait.text(
            idx,
            value - 2.5 if value < 0 else value + 1.5,
            f"{value:.0f}%{suffix}",
            ha="center",
            va="top" if value < 0 else "bottom",
            fontsize=9,
        )

    ax_ratio.set_xlim(
        START_DATE,
        max(pd.Timestamp(daily["date"].max()), pd.Timestamp("2026-10-19"))
        + pd.Timedelta(days=45),
    )
    ax_ratio.xaxis.set_major_locator(mdates.YearLocator(1))
    ax_ratio.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for label in ax_price.get_xticklabels():
        label.set_visible(False)

    note = (
        "* Current cycle bar is incomplete: data ends at "
        f"{pd.Timestamp(daily['date'].max()):%Y-%m-%d}; expected low is 2026-10-19."
    )
    fig.text(0.01, 0.01, note, fontsize=9, color="#4b5563")

    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = paths.report_dir / "cycle_floor_comparison_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    output = paths.figure_dir / "cycle_floor_comparison_dashboard.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(write_plot(ProjectPaths.from_cwd()))
