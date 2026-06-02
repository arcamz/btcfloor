from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from btcfloor.cycle import DEFAULT_CYCLE_ANCHORS, current_cycle_phase
from btcfloor.data import load_price_history
from btcfloor.paths import ProjectPaths
from btcfloor.powerlaw import giovanni_power_law_floor_model


LOOKBACK_DAYS_TO_LOW = 365


def build_cycle_cloud(daily: pd.DataFrame) -> pd.DataFrame:
    model = giovanni_power_law_floor_model()
    latest = daily.iloc[-1]
    latest_date = pd.Timestamp(latest["date"])
    current_phase = current_cycle_phase(latest_date)
    current_expected_low = pd.Timestamp(current_phase["expected_next_low_date"])

    rows = []
    for anchor in DEFAULT_CYCLE_ANCHORS:
        peak_date = (
            anchor.observed_peak_date
            if anchor.observed_peak_date is not None
            else anchor.low_date + pd.Timedelta(days=1064)
        )
        low_date = (
            anchor.observed_next_low_date
            if anchor.observed_next_low_date is not None
            else current_expected_low
        )
        observed_end = min(pd.Timestamp(low_date), latest_date)
        window = daily.loc[
            daily["date"].between(pd.Timestamp(peak_date), observed_end),
            ["date", "price_usd"],
        ].copy()
        if window.empty:
            continue
        window["cycle"] = anchor.name.replace("_cycle", "")
        window["expected_or_observed_low"] = pd.Timestamp(low_date)
        window["days_to_low"] = (pd.Timestamp(low_date) - window["date"]).dt.days
        window = window.loc[window["days_to_low"].between(0, LOOKBACK_DAYS_TO_LOW)].copy()
        if window.empty:
            continue
        window["ratio_to_hard_floor"] = (
            window["price_usd"].to_numpy(dtype=float)
            / model.predict_price(window["date"], floor=True)
        )
        rows.append(window)

    if not rows:
        return pd.DataFrame(
            columns=["date", "price_usd", "cycle", "days_to_low", "ratio_to_hard_floor"]
        )
    return pd.concat(rows, ignore_index=True)


def build_cloud_quantiles(cycles: pd.DataFrame) -> pd.DataFrame:
    historical = cycles.loc[~cycles["cycle"].eq("2022")].copy()
    if historical.empty:
        return pd.DataFrame()
    grouped = historical.groupby("days_to_low")["ratio_to_hard_floor"]
    return grouped.agg(
        low="min",
        q25=lambda s: s.quantile(0.25),
        median="median",
        q75=lambda s: s.quantile(0.75),
        high="max",
    ).reset_index()


def write_plot(paths: ProjectPaths) -> pd.DataFrame:
    daily = load_price_history(paths.processed_btc_csv)
    cycles = build_cycle_cloud(daily)
    cloud = build_cloud_quantiles(cycles)
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    cycles.to_csv(paths.report_dir / "cycle_cloud_decision_daily.csv", index=False)
    cloud.to_csv(paths.report_dir / "cycle_cloud_decision_cloud.csv", index=False)

    latest = daily.iloc[-1]
    latest_date = pd.Timestamp(latest["date"])
    phase = current_cycle_phase(latest_date)
    days_to_low = int(phase["days_to_expected_next_low"])
    current_cycle = cycles.loc[cycles["cycle"].eq("2022")].copy()
    current_point = current_cycle.loc[current_cycle["date"].eq(latest_date)]

    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if not cloud.empty:
        x = cloud["days_to_low"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            cloud["low"].to_numpy(dtype=float),
            cloud["high"].to_numpy(dtype=float),
            color="#9ecae1",
            alpha=0.22,
            linewidth=0,
        )
        ax.fill_between(
            x,
            cloud["q25"].to_numpy(dtype=float),
            cloud["q75"].to_numpy(dtype=float),
            color="#3182bd",
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            x,
            cloud["median"].to_numpy(dtype=float),
            color="#08519c",
            linewidth=1.6,
            alpha=0.75,
        )
        label_row = cloud.iloc[(cloud["days_to_low"] - 120).abs().argmin()]
        ax.text(
            float(label_row["days_to_low"]) + 5,
            float(label_row["q75"]) * 1.04,
            "prior bear-window range",
            color="#377eb8",
            fontsize=9,
            va="bottom",
        )
        ax.text(
            float(label_row["days_to_low"]) + 5,
            float(label_row["median"]) * 0.98,
            "prior median",
            color="#08519c",
            fontsize=9,
            va="top",
        )

    for cycle, frame in cycles.groupby("cycle"):
        if cycle == "2022":
            continue
        ax.plot(
            frame["days_to_low"],
            frame["ratio_to_hard_floor"],
            color="#6b7280",
            linewidth=0.9,
            alpha=0.35,
        )

    if not current_cycle.empty:
        ax.plot(
            current_cycle["days_to_low"],
            current_cycle["ratio_to_hard_floor"],
            color="#111827",
            linewidth=2.4,
        )
        if not current_point.empty:
            ratio = float(current_point["ratio_to_hard_floor"].iloc[0])
            ax.scatter(
                [days_to_low],
                [ratio],
                color="#111827",
                s=52,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )
            ax.text(
                days_to_low + 8,
                ratio,
                f"now: {ratio:.2f}x hard floor\n{days_to_low} days to expected low",
                color="#111827",
                fontsize=10,
                va="center",
            )

    ax.axhline(1.0, color="#b91c1c", linewidth=1.0, linestyle="--")
    ax.text(360, 1.02, "hard floor touch", color="#b91c1c", fontsize=9, va="bottom")

    ax.set_xlim(LOOKBACK_DAYS_TO_LOW, 0)
    ax.set_yscale("log")
    y_upper_candidates = [3.8]
    if not cloud.empty:
        y_upper_candidates.append(float(cloud["high"].quantile(0.98)) * 1.05)
    if not current_cycle.empty:
        y_upper_candidates.append(float(current_cycle["ratio_to_hard_floor"].max()) * 1.08)
    ax.set_ylim(0.9, min(max(y_upper_candidates), 6.0))
    ax.yaxis.set_major_locator(FixedLocator([1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}x"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title("Historical bear-window cloud: distance to Giovanni hard floor", fontsize=15)
    ax.text(
        365,
        3.52,
        "Current cycle is already near the hard-floor zone relative to prior bear windows.",
        fontsize=10,
        color="#4b5563",
        ha="left",
    )
    ax.set_xlabel("Days to observed/expected cycle low")
    ax.set_ylabel("BTC price / Giovanni hard floor")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")

    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    output = paths.figure_dir / "cycle_cloud_decision.png"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return cycles


if __name__ == "__main__":
    result = write_plot(ProjectPaths.from_cwd())
    print("Wrote reports/figures/cycle_cloud_decision.png")
    print(result.tail(8).to_string(index=False))
