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
) -> pd.Timestamp:
    window = daily.loc[
        daily["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ]
    if window.empty:
        raise ValueError(f"No price rows between {start_date} and {end_date}")
    return pd.Timestamp(window.loc[window["price_usd"].idxmax(), "date"])


def build_bear_windows(daily: pd.DataFrame) -> pd.DataFrame:
    latest_date = pd.Timestamp(daily["date"].max())
    rows = [
        {
            "cycle_label": "2013 peak to 2015 low",
            "peak_date": observed_peak_between(daily, "2011-11-18", "2015-01-14"),
            "bear_low_date": pd.Timestamp("2015-01-14"),
            "low_type": "observed",
        },
        {
            "cycle_label": "2017 peak to 2018 low",
            "peak_date": observed_peak_between(daily, "2015-01-14", "2018-12-15"),
            "bear_low_date": pd.Timestamp("2018-12-15"),
            "low_type": "observed",
        },
        {
            "cycle_label": "2021 peak to 2022 low",
            "peak_date": observed_peak_between(daily, "2018-12-15", "2022-11-21"),
            "bear_low_date": pd.Timestamp("2022-11-21"),
            "low_type": "observed",
        },
        {
            "cycle_label": "2025 expected peak to 2026 expected low",
            "peak_date": pd.Timestamp("2025-10-20"),
            "bear_low_date": pd.Timestamp("2026-10-19"),
            "low_type": "expected",
        },
    ]
    windows = pd.DataFrame(rows)
    windows["observed_through_date"] = windows["bear_low_date"].map(
        lambda low_date: min(pd.Timestamp(low_date), latest_date)
    )
    windows["fully_observed"] = windows["bear_low_date"] <= latest_date
    windows["bear_window_days"] = (
        windows["bear_low_date"] - windows["peak_date"]
    ).dt.days
    return windows


def build_bear_entry_metrics(
    daily: pd.DataFrame,
    overlap: pd.DataFrame,
    bear_windows: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for window in bear_windows.itertuples(index=False):
        peak_date = pd.Timestamp(window.peak_date)
        bear_low_date = pd.Timestamp(window.bear_low_date)
        observed_end = pd.Timestamp(window.observed_through_date)

        signal_window = overlap.loc[
            overlap["date"].between(peak_date, observed_end)
            & overlap["below_future_floor"].astype(bool)
        ].copy()

        if signal_window.empty:
            rows.append(
                {
                    "cycle_label": window.cycle_label,
                    "peak_date": peak_date,
                    "bear_low_date": bear_low_date,
                    "observed_through_date": observed_end,
                    "low_type": window.low_type,
                    "fully_observed": bool(window.fully_observed),
                    "entry_date": pd.NaT,
                    "entry_price_usd": np.nan,
                    "first_breach_ratio": np.nan,
                    "future_floor_at_entry_usd": np.nan,
                    "cheapest_price_date": pd.NaT,
                    "cheapest_price_usd": np.nan,
                    "lower_price_opportunity_pct": np.nan,
                    "lower_price_available": False,
                    "days_to_cheapest_price": np.nan,
                    "days_from_entry_to_bear_low": np.nan,
                }
            )
            continue

        entry = signal_window.iloc[0]
        entry_date = pd.Timestamp(entry["date"])
        entry_price = float(entry["price_usd"])
        opportunity_window = daily.loc[
            daily["date"].between(entry_date, observed_end)
        ].copy()
        cheapest = opportunity_window.loc[opportunity_window["price_usd"].idxmin()]
        cheapest_date = pd.Timestamp(cheapest["date"])
        cheapest_price = float(cheapest["price_usd"])
        opportunity_pct = cheapest_price / entry_price - 1.0

        rows.append(
            {
                "cycle_label": window.cycle_label,
                "peak_date": peak_date,
                "bear_low_date": bear_low_date,
                "observed_through_date": observed_end,
                "low_type": window.low_type,
                "fully_observed": bool(window.fully_observed),
                "entry_date": entry_date,
                "entry_price_usd": entry_price,
                "first_breach_ratio": float(entry["ratio_to_future_floor"]),
                "future_floor_at_entry_usd": float(entry["future_floor_usd"]),
                "cheapest_price_date": cheapest_date,
                "cheapest_price_usd": cheapest_price,
                "lower_price_opportunity_pct": opportunity_pct,
                "lower_price_available": opportunity_pct < 0.0,
                "days_to_cheapest_price": int((cheapest_date - entry_date).days),
                "days_from_entry_to_bear_low": int((bear_low_date - entry_date).days),
            }
        )
    return pd.DataFrame(rows)


def shade_windows(
    ax: plt.Axes,
    bear_windows: pd.DataFrame,
    color: str = "#dce9f7",
    alpha: float = 0.42,
) -> None:
    for window in bear_windows.itertuples(index=False):
        ax.axvspan(
            pd.Timestamp(window.peak_date),
            pd.Timestamp(window.bear_low_date),
            color=color,
            alpha=alpha,
            linewidth=0,
        )


def write_plot(paths: ProjectPaths) -> Path:
    daily = pd.read_csv(paths.processed_btc_csv, parse_dates=["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    model = giovanni_power_law_floor_model()
    overlap = future_floor_overlap_daily(daily, model, horizon_months=HORIZON_MONTHS)
    bear_windows = build_bear_windows(daily)
    metrics = build_bear_entry_metrics(daily, overlap, bear_windows)

    plot_frame = overlap.loc[overlap["date"] >= START_DATE].copy()
    dates = plot_frame["date"]
    price = plot_frame["price_usd"].to_numpy(dtype=float)
    future_floor = plot_frame["future_floor_usd"].to_numpy(dtype=float)
    ratio = plot_frame["ratio_to_future_floor"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.25, 1.0, 1.3])
    ax_price = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_price)
    ax_outcome = fig.add_subplot(gs[2])

    shade_windows(ax_price, bear_windows)
    shade_windows(ax_ratio, bear_windows)

    ax_price.plot(dates, price, color="#18212f", linewidth=1.6, label="BTC daily close")
    ax_price.plot(
        dates,
        future_floor,
        color="#d94841",
        linewidth=1.8,
        label=f"Giovanni floor {HORIZON_MONTHS}m forward",
    )

    entries = metrics.dropna(subset=["entry_date"]).copy()
    ax_price.scatter(
        entries["entry_date"],
        entries["entry_price_usd"],
        marker="v",
        s=95,
        color="#d94841",
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
        label="first breach after peak",
    )
    ax_price.scatter(
        entries["cheapest_price_date"],
        entries["cheapest_price_usd"],
        marker="o",
        s=58,
        color="#2563a6",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="cheapest later price before low",
    )
    for entry in entries.itertuples(index=False):
        ax_price.annotate(
            f"{entry.entry_date:%Y}",
            (entry.entry_date, entry.entry_price_usd),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            color="#4b5563",
        )

    ax_price.set_yscale("log")
    ax_price.set_title(
        "Future-floor breach inside post-peak bear windows",
        fontsize=16,
        pad=12,
    )
    ax_price.set_ylabel("USD, log scale")
    ax_price.grid(True, which="both", alpha=0.22)
    ax_price.legend(loc="upper left", frameon=False, ncols=2)

    ax_ratio.plot(dates, ratio, color="#2563a6", linewidth=1.35)
    ax_ratio.axhline(1.0, color="#d94841", linestyle="--", linewidth=1.2)
    ax_ratio.fill_between(
        dates,
        ratio,
        1.0,
        where=ratio < 1.0,
        color="#d94841",
        alpha=0.20,
        interpolate=True,
    )
    ax_ratio.set_ylabel("Price / future floor")
    ax_ratio.set_ylim(0.35, max(2.3, float(np.nanpercentile(ratio, 88))))
    ax_ratio.grid(True, alpha=0.22)
    ax_ratio.set_title(
        "Signal threshold: below 1.0 means spot is below the 12m-forward floor",
        fontsize=12,
        pad=8,
    )

    metrics = metrics.sort_values("peak_date").reset_index(drop=True)
    x = np.arange(len(metrics))
    labels = [
        f"{row.cycle_label.replace(' to ', ' -> ')}\nentry "
        f"{pd.Timestamp(row.entry_date):%Y-%m-%d}"
        for row in metrics.itertuples(index=False)
    ]
    opportunity = metrics["lower_price_opportunity_pct"].fillna(0.0).to_numpy(dtype=float)
    bar_values = opportunity * 100.0
    colors = np.where(metrics["lower_price_available"], "#b42318", "#2b8a5f")
    colors = np.where(metrics["fully_observed"], colors, "#94a3b8")
    bars = ax_outcome.bar(
        x,
        bar_values,
        color=colors,
        width=0.68,
        label="best lower buy before bear low/window end",
    )
    for bar, fully_observed in zip(bars, metrics["fully_observed"], strict=True):
        if not fully_observed:
            bar.set_hatch("//")
            bar.set_edgecolor("#64748b")

    ax_days = ax_outcome.twinx()
    ax_days.plot(
        x,
        metrics["days_to_cheapest_price"].to_numpy(dtype=float),
        color="#2563a6",
        marker="o",
        linewidth=1.8,
        label="days from breach to cheapest price",
    )
    ax_outcome.axhline(0.0, color="#111827", linewidth=0.8)
    ax_outcome.set_xticks(x)
    ax_outcome.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax_outcome.set_ylabel("Cheapest price before low vs first breach")
    ax_days.set_ylabel("Days from first breach to cheapest price")
    ax_outcome.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}%")
    y_min = min(-5.0, float(np.nanmin(bar_values)) - 8.0)
    ax_outcome.set_ylim(y_min, 4.0)
    ax_outcome.grid(True, axis="y", alpha=0.22)
    ax_outcome.set_title(
        "Was waiting after the first post-peak breach rewarded with a lower price?",
        fontsize=12,
        pad=8,
    )

    for idx, row in metrics.iterrows():
        value = row["lower_price_opportunity_pct"]
        if pd.isna(value):
            text = "no signal"
            y = 2.0
        else:
            suffix = "*" if not row["fully_observed"] else ""
            text = f"{value * 100:,.0f}%{suffix}"
            y = value * 100.0
            y += 2 if y >= 0 else -3
        ax_outcome.text(
            idx,
            y,
            text,
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=9,
            color="#111827",
        )

    handles_1, labels_1 = ax_outcome.get_legend_handles_labels()
    handles_2, labels_2 = ax_days.get_legend_handles_labels()
    ax_outcome.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="lower right",
        frameon=False,
    )

    x_max = max(
        pd.Timestamp(daily["date"].max()),
        pd.Timestamp(bear_windows["bear_low_date"].max()),
    ) + pd.Timedelta(days=45)
    ax_ratio.set_xlim(START_DATE, x_max)
    ax_ratio.xaxis.set_major_locator(mdates.YearLocator(1))
    ax_ratio.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for label in ax_price.get_xticklabels():
        label.set_visible(False)

    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = paths.report_dir / "bear_window_forward_floor_decision_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    output = paths.figure_dir / "bear_window_forward_floor_decision_plot.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(write_plot(ProjectPaths.from_cwd()))
