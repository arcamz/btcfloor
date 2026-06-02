from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from btcfloor.cycle import current_cycle_phase
from btcfloor.data import load_price_history, to_weekly_close, to_weekly_ohlc
from btcfloor.expectile import expectile_model_name, fit_expectile_power_law
from btcfloor.forward_floor import floor_target_crossing_date
from btcfloor.paths import ProjectPaths
from btcfloor.powerlaw import PowerLawModel, giovanni_power_law_floor_model


EXPECTILE_TAUS = (0.0001, 0.001, 0.01)
TACTICAL_LEVELS = (
    ("recent wick low", 66_500.0),
    ("broken support reclaim", 68_850.0),
    ("first range reclaim", 73_600.0),
)
PRICE_CONTEXT_DAYS = 78


def money_k(value: float) -> str:
    return f"${value / 1000:.1f}k"


def model_label(model: PowerLawModel) -> str:
    if model.name == "giovanni_power_law_floor":
        return "Giovanni hard floor"
    tau_text = model.name.removeprefix("weekly_expectile_power_law_tau_").replace("_", ".")
    tau = float(tau_text)
    return f"{tau * 100:g}% expectile"


def build_models(daily: pd.DataFrame) -> list[PowerLawModel]:
    weekly = to_weekly_close(daily)
    return [
        giovanni_power_law_floor_model(),
        *[
            fit_expectile_power_law(weekly, tau=tau, name=expectile_model_name(tau))
            for tau in EXPECTILE_TAUS
        ],
    ]


def build_metrics(
    models: list[PowerLawModel],
    latest_date: pd.Timestamp,
    spot_price: float,
    expected_low_date: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for model in models:
        latest_floor = float(model.predict_price(latest_date, floor=True)[0])
        expected_low_floor = float(model.predict_price(expected_low_date, floor=True)[0])
        cross_date = floor_target_crossing_date(
            model,
            spot_price,
            start_date=latest_date,
        )
        rows.append(
            {
                "model": model.name,
                "label": model_label(model),
                "latest_date": latest_date,
                "spot_price_usd": spot_price,
                "latest_floor_usd": latest_floor,
                "expected_low_date": expected_low_date,
                "expected_low_floor_usd": expected_low_floor,
                "latest_pct_above_floor": spot_price / latest_floor - 1.0,
                "spot_pct_vs_expected_low_floor": spot_price / expected_low_floor - 1.0,
                "floor_crosses_current_spot_date": cross_date,
                "crosses_before_expected_low": cross_date <= expected_low_date,
            }
        )
    return pd.DataFrame(rows)


def add_direct_label(
    ax: plt.Axes,
    x: pd.Timestamp,
    y: float,
    text: str,
    color: str,
    dy: float = 0.0,
) -> None:
    ax.text(
        x + pd.Timedelta(days=5),
        y + dy,
        text,
        color=color,
        fontsize=10,
        va="center",
        ha="left",
    )


def add_weekly_candles(ax: plt.Axes, weekly: pd.DataFrame) -> None:
    width = 3.8
    for row in weekly.itertuples(index=False):
        x = mdates.date2num(pd.Timestamp(row.date))
        open_price = float(row.open)
        close_price = float(row.close)
        high_price = float(row.high)
        low_price = float(row.low)
        up = close_price >= open_price
        color = "#15803d" if up else "#b91c1c"
        fill = "#d1fae5" if up else "#fee2e2"
        ax.vlines(x, low_price, high_price, color=color, linewidth=1.0, alpha=0.9)
        body_bottom = min(open_price, close_price)
        body_height = max(abs(close_price - open_price), 85.0)
        ax.add_patch(
            Rectangle(
                (x - width / 2, body_bottom),
                width,
                body_height,
                facecolor=fill,
                edgecolor=color,
                linewidth=1.2,
                alpha=0.95,
            )
        )


def add_floor_corridors(
    ax: plt.Axes,
    plot_dates: pd.DatetimeIndex,
    floor_by_model: dict[str, np.ndarray],
) -> None:
    hard = floor_by_model["giovanni_power_law_floor"]
    e001 = floor_by_model["weekly_expectile_power_law_tau_0_0001"]
    e01 = floor_by_model["weekly_expectile_power_law_tau_0_001"]
    e1 = floor_by_model["weekly_expectile_power_law_tau_0_01"]
    corridors = [
        (hard, e001, "#d1fae5", 0.20),
        (e001, e01, "#dbeafe", 0.16),
        (e01, e1, "#fee2e2", 0.14),
    ]
    for lower, upper, color, alpha in corridors:
        ax.fill_between(plot_dates, lower, upper, color=color, alpha=alpha, linewidth=0)


def write_plot(paths: ProjectPaths) -> pd.DataFrame:
    daily = load_price_history(paths.processed_btc_csv)
    weekly = to_weekly_ohlc(daily)
    latest = daily.iloc[-1]
    latest_date = pd.Timestamp(latest["date"])
    spot_price = float(latest["price_usd"])
    phase = current_cycle_phase(latest_date)
    expected_low_date = pd.Timestamp(phase["expected_next_low_date"])
    days_to_low = int(phase["days_to_expected_next_low"])

    models = build_models(daily)
    metrics = build_metrics(models, latest_date, spot_price, expected_low_date)

    paths.report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(paths.report_dir / "floor_convergence_decision_metrics.csv", index=False)

    context_start = latest_date - pd.Timedelta(days=PRICE_CONTEXT_DAYS)
    plot_dates = pd.date_range(context_start, expected_low_date, freq="D")
    candle_frame = weekly.loc[
        weekly["date"].between(context_start, latest_date),
        ["date", "open", "high", "low", "close"],
    ].copy()
    colors = {
        "giovanni_power_law_floor": "#d95f02",
        "weekly_expectile_power_law_tau_0_0001": "#1b9e77",
        "weekly_expectile_power_law_tau_0_001": "#2c7fb8",
        "weekly_expectile_power_law_tau_0_01": "#d73027",
    }
    linewidths = {
        "giovanni_power_law_floor": 2.8,
        "weekly_expectile_power_law_tau_0_0001": 2.2,
        "weekly_expectile_power_law_tau_0_001": 1.6,
        "weekly_expectile_power_law_tau_0_01": 1.1,
    }
    alphas = {
        "giovanni_power_law_floor": 1.0,
        "weekly_expectile_power_law_tau_0_0001": 1.0,
        "weekly_expectile_power_law_tau_0_001": 0.9,
        "weekly_expectile_power_law_tau_0_01": 0.78,
    }

    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    floor_by_model = {
        model.name: model.predict_price(plot_dates, floor=True).astype(float)
        for model in models
    }
    add_floor_corridors(ax, plot_dates, floor_by_model)
    add_weekly_candles(ax, candle_frame)
    if not candle_frame.empty:
        ax.text(
            pd.Timestamp(candle_frame["date"].iloc[0]),
            57_700,
            "recent weekly candles",
            color="#6b7280",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    ax.plot(
        [latest_date, expected_low_date],
        [spot_price, spot_price],
        color="#111827",
        linewidth=2.5,
        solid_capstyle="butt",
        zorder=3,
    )
    ax.text(
        latest_date + pd.Timedelta(days=3),
        spot_price + 270,
        f"latest close {money_k(spot_price)}",
        color="#111827",
        fontsize=10,
        va="bottom",
    )

    for level_label, level in TACTICAL_LEVELS:
        color = "#9ca3af" if level_label == "recent wick low" else "#64748b"
        linestyle = ":" if level_label == "recent wick low" else "--"
        ax.axhline(level, color=color, linewidth=0.9, linestyle=linestyle)
        ax.text(
            expected_low_date + pd.Timedelta(days=6),
            level,
            f"{level_label} {money_k(level)}",
            color=color,
            fontsize=8.5,
            va="center",
        )

    for model in models:
        color = colors[model.name]
        floor = floor_by_model[model.name]
        ax.plot(
            plot_dates,
            floor,
            color=color,
            linewidth=linewidths[model.name],
            alpha=alphas[model.name],
        )
        row = metrics.loc[metrics["model"].eq(model.name)].iloc[0]
        endpoint = float(row["expected_low_floor_usd"])
        ax.scatter(
            [expected_low_date],
            [endpoint],
            color=color,
            s=32,
            marker="D",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        add_direct_label(
            ax,
            expected_low_date,
            endpoint,
            f"{model_label(model)} {money_k(endpoint)}",
            color,
        )

        cross_date = pd.Timestamp(row["floor_crosses_current_spot_date"])
        if cross_date <= expected_low_date and cross_date > latest_date:
            ax.scatter(
                [cross_date],
                [spot_price],
                color=color,
                s=34,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )
            ax.plot(
                [cross_date, cross_date],
                [spot_price - 650, spot_price + 650],
                color=color,
                linewidth=0.8,
            )
            ax.text(
                cross_date,
                spot_price - 950,
                f"{model_label(model)} catches spot\n{cross_date:%b %d}",
                color=color,
                fontsize=8.5,
                ha="center",
                va="top",
            )

    hard = metrics.loc[metrics["model"].eq("giovanni_power_law_floor")].iloc[0]
    hard_cross = pd.Timestamp(hard["floor_crosses_current_spot_date"])
    hard_expected_low = float(hard["expected_low_floor_usd"])
    ax.plot(
        [latest_date, expected_low_date],
        [hard_expected_low, hard_expected_low],
        color="#111827",
        linewidth=1.6,
        solid_capstyle="butt",
        alpha=0.9,
        zorder=3,
    )
    ax.vlines(
        latest_date,
        float(hard["latest_floor_usd"]),
        spot_price,
        color="#111827",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.text(
        latest_date + pd.Timedelta(days=10),
        hard_expected_low - 900,
        f"Hard floor does not catch current spot before expected low\n"
        f"catch-up would be {hard_cross:%Y-%m-%d}",
        color="#7c2d12",
        fontsize=9.5,
        va="top",
    )

    ax.axvline(expected_low_date, color="#991b1b", linewidth=1.2, linestyle="--")
    ax.text(
        expected_low_date - pd.Timedelta(days=3),
        88_000,
        f"expected low\n{expected_low_date:%Y-%m-%d}",
        color="#991b1b",
        fontsize=9.5,
        ha="right",
        va="top",
    )

    ax.text(
        latest_date,
        88_400,
        "Floor convergence: price now versus floors ahead",
        fontsize=18,
        color="#111827",
        ha="left",
        va="bottom",
    )
    ax.text(
        latest_date,
        87_100,
        (
            f"Latest processed UTC daily close: {latest_date:%Y-%m-%d}. "
            f"{days_to_low} days to expected bear-market low."
        ),
        fontsize=10,
        color="#4b5563",
        ha="left",
        va="bottom",
    )

    ax.set_xlim(context_start - pd.Timedelta(days=5), expected_low_date + pd.Timedelta(days=30))
    ax.set_ylim(57_000, 89_000)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(axis="both", colors="#374151", labelsize=9)
    ax.yaxis.set_major_formatter(lambda value, _: money_k(value))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    output = paths.figure_dir / "floor_convergence_decision_dashboard.png"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    result = write_plot(ProjectPaths.from_cwd())
    print("Wrote reports/figures/floor_convergence_decision_dashboard.png")
    print(result.to_string(index=False))
