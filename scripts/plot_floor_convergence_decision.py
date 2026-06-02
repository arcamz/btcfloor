from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcfloor.cycle import (
    DEFAULT_CYCLE_ANCHORS,
    FOURCHAN_FULL_LOW_TO_LOW_DAYS,
    current_cycle_phase,
)
from btcfloor.data import load_price_history, to_weekly_close
from btcfloor.expectile import expectile_model_name, fit_expectile_power_law
from btcfloor.paths import ProjectPaths
from btcfloor.powerlaw import PowerLawModel, giovanni_power_law_floor_model


EXPECTED_LOW_EXPECTILE_TAUS = (0.0001, 0.001, 0.005, 0.01)
RECENT_LOW_LEVEL_USD = 66_500.0
BROKEN_SUPPORT_USD = 68_850.0
FIRST_RECLAIM_USD = 73_600.0
MOMENTUM_RECLAIM_USD = 79_300.0


def _money_k(value: float) -> str:
    return f"${value / 1000:.1f}k"


def _model_label(model: PowerLawModel) -> str:
    if model.name == "giovanni_power_law_floor":
        return "Giovanni hard floor"
    tau_text = model.name.removeprefix("weekly_expectile_power_law_tau_").replace("_", ".")
    tau = float(tau_text)
    return f"{tau * 100:g}% expectile"


def _first_cross_date(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    threshold: float,
) -> pd.Timestamp | None:
    crosses = np.flatnonzero(values >= threshold)
    if len(crosses) == 0:
        return None
    return pd.Timestamp(dates[int(crosses[0])])


def _build_models(daily: pd.DataFrame) -> list[PowerLawModel]:
    weekly = to_weekly_close(daily)
    return [
        giovanni_power_law_floor_model(),
        *[
            fit_expectile_power_law(weekly, tau=tau, name=expectile_model_name(tau))
            for tau in EXPECTED_LOW_EXPECTILE_TAUS
        ],
    ]


def _projection_frame(
    models: list[PowerLawModel],
    latest_date: pd.Timestamp,
    expected_low_date: pd.Timestamp,
    spot_price: float,
) -> pd.DataFrame:
    dates = pd.date_range(latest_date, expected_low_date, freq="D")
    rows = []
    for model in models:
        floors = model.predict_price(dates, floor=True).astype(float)
        cross_date = _first_cross_date(dates, floors, spot_price)
        for date, floor in zip(dates, floors):
            rows.append(
                {
                    "date": date,
                    "model": model.name,
                    "label": _model_label(model),
                    "floor_usd": float(floor),
                    "floor_crosses_current_spot_date": cross_date,
                }
            )
    return pd.DataFrame(rows)


def _cycle_decline_frame(
    daily: pd.DataFrame,
    hard_floor: PowerLawModel,
    latest_date: pd.Timestamp,
    current_expected_low: pd.Timestamp,
) -> pd.DataFrame:
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
        window["days_to_low"] = (pd.Timestamp(low_date) - window["date"]).dt.days
        window["ratio_to_hard_floor"] = (
            window["price_usd"].to_numpy(dtype=float)
            / hard_floor.predict_price(window["date"], floor=True)
        )
        rows.append(window)
    if not rows:
        return pd.DataFrame(columns=["date", "price_usd", "cycle", "days_to_low", "ratio_to_hard_floor"])
    return pd.concat(rows, ignore_index=True)


def write_plot(paths: ProjectPaths) -> Path:
    daily = load_price_history(paths.processed_btc_csv)
    latest = daily.iloc[-1]
    latest_date = pd.Timestamp(latest["date"])
    spot_price = float(latest["price_usd"])
    phase = current_cycle_phase(latest_date)
    expected_low_date = pd.Timestamp(phase["expected_next_low_date"])
    days_to_low = int(phase["days_to_expected_next_low"])

    models = _build_models(daily)
    hard_floor = models[0]
    projection = _projection_frame(models, latest_date, expected_low_date, spot_price)
    cycle_declines = _cycle_decline_frame(
        daily,
        hard_floor=hard_floor,
        latest_date=latest_date,
        current_expected_low=expected_low_date,
    )

    expected_rows = projection.loc[projection["date"].eq(expected_low_date)].copy()
    latest_rows = projection.loc[projection["date"].eq(latest_date)].copy()
    metrics = expected_rows.merge(
        latest_rows[["model", "floor_usd"]].rename(columns={"floor_usd": "latest_floor_usd"}),
        on="model",
        how="left",
    )
    metrics["latest_date"] = latest_date
    metrics["spot_price_usd"] = spot_price
    metrics["expected_low_date"] = expected_low_date
    metrics["days_to_expected_low"] = days_to_low
    metrics = metrics.rename(columns={"floor_usd": "expected_low_floor_usd"})
    metrics["latest_pct_above_floor"] = spot_price / metrics["latest_floor_usd"] - 1.0
    metrics["spot_pct_vs_expected_low_floor"] = (
        spot_price / metrics["expected_low_floor_usd"] - 1.0
    )
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(paths.report_dir / "floor_convergence_decision_metrics.csv", index=False)

    colors = {
        "giovanni_power_law_floor": "#d35400",
        "weekly_expectile_power_law_tau_0_0001": "#1b9e77",
        "weekly_expectile_power_law_tau_0_001": "#377eb8",
        "weekly_expectile_power_law_tau_0_005": "#984ea3",
        "weekly_expectile_power_law_tau_0_01": "#e41a1c",
    }

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], width_ratios=[1.45, 1.0])
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_gap = fig.add_subplot(gs[0, 1])
    ax_history = fig.add_subplot(gs[1, 0])
    ax_ladder = fig.add_subplot(gs[1, 1])

    for model in models:
        model_projection = projection.loc[projection["model"].eq(model.name)]
        color = colors.get(model.name, "#555555")
        linewidth = 2.5 if model.name == "giovanni_power_law_floor" else 1.8
        ax_curve.plot(
            model_projection["date"],
            model_projection["floor_usd"],
            color=color,
            linewidth=linewidth,
            label=_model_label(model),
        )
        expected_floor = float(
            model_projection.loc[
                model_projection["date"].eq(expected_low_date), "floor_usd"
            ].iloc[0]
        )
        ax_curve.scatter(
            [expected_low_date],
            [expected_floor],
            color=color,
            marker="D",
            s=58,
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
        ax_curve.text(
            expected_low_date + pd.Timedelta(days=3),
            expected_floor,
            _money_k(expected_floor),
            color=color,
            fontsize=9,
            va="center",
        )

    ax_curve.axhline(
        spot_price,
        color="#111827",
        linewidth=2.0,
        linestyle="-",
        label=f"latest spot {_money_k(spot_price)}",
    )
    ax_curve.axvline(
        expected_low_date,
        color="#b22222",
        linewidth=1.4,
        linestyle="--",
    )
    ax_curve.text(
        expected_low_date,
        ax_curve.get_ylim()[1] * 0.98,
        f"expected low\n{expected_low_date:%Y-%m-%d}",
        color="#7f1d1d",
        fontsize=9,
        ha="right",
        va="top",
    )
    ax_curve.set_title(
        "Rising floors versus current spot into the expected bear-market low",
        fontsize=14,
    )
    ax_curve.set_ylabel("USD")
    ax_curve.yaxis.set_major_formatter(lambda value, _: _money_k(value))
    ax_curve.grid(True, alpha=0.22)
    ax_curve.legend(loc="upper left", frameon=False, fontsize=9)
    ax_curve.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_curve.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    gap = metrics.sort_values("expected_low_floor_usd")
    labels = list(gap["label"])
    latest_gaps = (gap["latest_pct_above_floor"] * 100.0).to_numpy(dtype=float)
    expected_gaps = (gap["spot_pct_vs_expected_low_floor"] * 100.0).to_numpy(dtype=float)
    x = np.arange(len(labels))
    ax_gap.bar(x - 0.18, latest_gaps, width=0.36, color="#6b7280", label="today")
    ax_gap.bar(
        x + 0.18,
        expected_gaps,
        width=0.36,
        color=["#b42318" if v < 0 else "#2b8a5f" for v in expected_gaps],
        label="if spot unchanged at expected low",
    )
    ax_gap.axhline(0, color="#111827", linewidth=0.8)
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(labels, rotation=24, ha="right")
    ax_gap.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax_gap.set_ylabel("Spot gap to floor")
    ax_gap.set_title("How fast the floors catch up", fontsize=14)
    ax_gap.grid(True, axis="y", alpha=0.22)
    ax_gap.legend(frameon=False, fontsize=9)
    for idx, row in enumerate(gap.itertuples(index=False)):
        cross = row.floor_crosses_current_spot_date
        text = "no cross" if pd.isna(cross) else pd.Timestamp(cross).strftime("%b %d")
        ax_gap.text(
            idx + 0.18,
            expected_gaps[idx] + (-4 if expected_gaps[idx] < 0 else 2),
            text,
            ha="center",
            va="top" if expected_gaps[idx] < 0 else "bottom",
            fontsize=8,
            color="#374151",
        )

    for cycle, frame in cycle_declines.groupby("cycle"):
        recent_window = frame.loc[frame["days_to_low"].between(0, 365)].copy()
        if recent_window.empty:
            continue
        linewidth = 2.7 if cycle == "2022" else 1.7
        color = "#e6550d" if cycle == "2022" else None
        ax_history.plot(
            recent_window["days_to_low"],
            recent_window["ratio_to_hard_floor"],
            label=f"{cycle} cycle",
            linewidth=linewidth,
            color=color,
            alpha=0.9,
        )
    current_ratio = spot_price / float(hard_floor.predict_price(latest_date, floor=True)[0])
    ax_history.scatter(
        [days_to_low],
        [current_ratio],
        color="#111827",
        s=80,
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
        label="current",
    )
    ax_history.axhline(1.0, color="#b22222", linestyle="--", linewidth=1.1)
    ax_history.set_xlim(365, 0)
    ax_history.set_yscale("log")
    ax_history.set_title(
        "Post-peak closeness to Giovanni hard floor, aligned by days to low",
        fontsize=14,
    )
    ax_history.set_xlabel("Days to observed/expected cycle low")
    ax_history.set_ylabel("Price / hard floor")
    ax_history.yaxis.set_major_formatter(lambda value, _: f"{value:.2f}x")
    ax_history.grid(True, which="both", alpha=0.22)
    ax_history.legend(frameon=False, fontsize=9)

    ladder_rows = [
        ("Momentum reclaim / 200D area", MOMENTUM_RECLAIM_USD, "#7c3aed"),
        ("First range reclaim", FIRST_RECLAIM_USD, "#2563eb"),
        ("Broken support reclaim", BROKEN_SUPPORT_USD, "#f59e0b"),
        ("Latest spot", spot_price, "#111827"),
        ("Recent local wick low", RECENT_LOW_LEVEL_USD, "#64748b"),
        ("0.01% expectile today", float(gap.loc[gap["model"].eq("weekly_expectile_power_law_tau_0_0001"), "latest_floor_usd"].iloc[0]), "#1b9e77"),
        ("Giovanni hard floor today", float(gap.loc[gap["model"].eq("giovanni_power_law_floor"), "latest_floor_usd"].iloc[0]), "#d35400"),
        ("Giovanni expected-low floor", float(gap.loc[gap["model"].eq("giovanni_power_law_floor"), "expected_low_floor_usd"].iloc[0]), "#d35400"),
        ("0.01% expected-low floor", float(gap.loc[gap["model"].eq("weekly_expectile_power_law_tau_0_0001"), "expected_low_floor_usd"].iloc[0]), "#1b9e77"),
    ]
    min_price = min(value for _, value, _ in ladder_rows) * 0.96
    max_price = max(value for _, value, _ in ladder_rows) * 1.03
    ax_ladder.set_ylim(min_price, max_price)
    ax_ladder.set_xlim(0, 1)
    ax_ladder.set_xticks([])
    ax_ladder.set_title("Decision ladder: value zone versus confirmation", fontsize=14)
    ax_ladder.yaxis.set_major_formatter(lambda value, _: _money_k(value))
    ax_ladder.grid(True, axis="y", alpha=0.18)
    for idx, (label, value, color) in enumerate(ladder_rows):
        x0 = 0.08 if idx % 2 == 0 else 0.18
        ax_ladder.hlines(value, x0, 0.92, color=color, linewidth=2.0)
        ax_ladder.text(
            0.93,
            value,
            f"{label}  {_money_k(value)}",
            color=color,
            fontsize=9,
            va="center",
            ha="left",
        )
    ax_ladder.text(
        0.08,
        max_price * 0.992,
        "Read: floor pressure supports staged value entries;\n"
        "SFP/reclaim levels handle tactical timing.",
        fontsize=10,
        color="#24324a",
        va="top",
    )

    fig.suptitle(
        f"BTC floor convergence decision view | latest UTC daily close {latest_date:%Y-%m-%d} "
        f"at {_money_k(spot_price)} | {days_to_low} days to expected low",
        fontsize=16,
    )
    output = paths.figure_dir / "floor_convergence_decision_dashboard.png"
    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(write_plot(ProjectPaths.from_cwd()))
