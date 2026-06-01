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


RECENT_PROVISIONAL_PRICES = (
    ("2026-05-24", 76_981.13),
    ("2026-05-25", 77_279.93),
    ("2026-05-26", 75_825.73),
    ("2026-05-27", 74_344.70),
    ("2026-05-28", 73_536.56),
    ("2026-05-29", 73_283.13),
    ("2026-05-30", 73_504.69),
)

BREACH_DATE = pd.Timestamp("2026-02-05")
REJECTION_SEARCH_START = pd.Timestamp("2026-04-01")
START_DATE = pd.Timestamp("2025-07-01")


def load_chart_prices(paths: ProjectPaths) -> pd.DataFrame:
    daily = pd.read_csv(paths.processed_btc_csv, parse_dates=["date"])
    recent = pd.DataFrame(RECENT_PROVISIONAL_PRICES, columns=["date", "price_usd"])
    recent["date"] = pd.to_datetime(recent["date"])
    recent["days_since_genesis"] = (
        recent["date"] - pd.Timestamp("2009-01-03")
    ).dt.days
    recent["source"] = "provisional_public_price"
    combined = (
        pd.concat([daily, recent], ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    combined["sma50"] = combined["price_usd"].rolling(50).mean()
    combined["sma200"] = combined["price_usd"].rolling(200).mean()
    return combined


def channel_from_rejection(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    x = mdates.date2num(frame["date"])
    y = np.log(frame["price_usd"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, deg=1)
    center_log = intercept + slope * x
    residuals = y - center_log
    width = max(float(np.std(residuals, ddof=1)) * 1.65, float(np.max(np.abs(residuals))))
    center = np.exp(center_log)
    upper = np.exp(center_log + width)
    lower = np.exp(center_log - width)
    return pd.Series(center, index=frame.index), pd.Series(upper, index=frame.index), pd.Series(
        lower,
        index=frame.index,
    )


def write_plot(paths: ProjectPaths) -> Path:
    daily = load_chart_prices(paths)
    model = giovanni_power_law_floor_model()
    overlap = future_floor_overlap_daily(daily, model, horizon_months=12)
    daily = daily.merge(
        overlap[["date", "future_floor_usd", "ratio_to_future_floor"]],
        on="date",
        how="left",
    )
    daily["giovanni_floor_usd"] = model.predict_price(daily["date"], floor=True)

    post_breach = daily.loc[daily["date"] >= BREACH_DATE].copy()
    rejection_window = daily.loc[
        daily["date"] >= REJECTION_SEARCH_START,
        ["date", "price_usd", "sma200"],
    ].dropna()
    rejection_idx = (
        (rejection_window["price_usd"] / rejection_window["sma200"] - 1.0).abs().idxmin()
    )
    rejection = daily.loc[rejection_idx]
    rejection_date = pd.Timestamp(rejection["date"])

    post_rejection = daily.loc[daily["date"] >= rejection_date].copy()
    center, upper, lower = channel_from_rejection(post_rejection)
    daily.loc[post_rejection.index, "channel_center"] = center
    daily.loc[post_rejection.index, "channel_upper"] = upper
    daily.loc[post_rejection.index, "channel_lower"] = lower

    after_rejection = daily.loc[daily["date"] >= rejection_date].copy()
    sma50_loss = after_rejection.loc[
        after_rejection["price_usd"] < after_rejection["sma50"]
    ].iloc[0]
    latest = daily.iloc[-1]

    view = daily.loc[daily["date"] >= START_DATE].copy()
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.4, 0.85, 1.05])
    ax_price = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_price)
    ax_state = fig.add_subplot(gs[2])

    ax_price.plot(view["date"], view["price_usd"], color="#18212f", linewidth=1.55, label="BTC close")
    ax_price.plot(view["date"], view["sma50"], color="#2b8a5f", linewidth=1.25, label="50-day SMA")
    ax_price.plot(view["date"], view["sma200"], color="#7c3aed", linewidth=1.35, label="200-day SMA")
    ax_price.plot(
        view["date"],
        view["future_floor_usd"],
        color="#d94841",
        linewidth=1.4,
        label="Giovanni floor 12m forward",
    )
    ax_price.plot(
        view["date"],
        view["giovanni_floor_usd"],
        color="#d94841",
        linewidth=1.0,
        alpha=0.45,
        linestyle=":",
        label="Giovanni same-day floor",
    )
    channel_view = view.dropna(subset=["channel_lower", "channel_upper"])
    ax_price.fill_between(
        channel_view["date"],
        channel_view["channel_lower"],
        channel_view["channel_upper"],
        color="#f59e0b",
        alpha=0.13,
        label="post-200SMA rejection channel",
    )
    ax_price.plot(
        channel_view["date"],
        channel_view["channel_center"],
        color="#b45309",
        linewidth=1.0,
        linestyle="--",
    )

    markers = [
        (BREACH_DATE, "future-floor breach", "#d94841", "v"),
        (rejection_date, "200SMA rejection", "#7c3aed", "X"),
        (pd.Timestamp(sma50_loss["date"]), "50SMA loss", "#2b8a5f", "v"),
        (pd.Timestamp(latest["date"]), "latest/provisional", "#111827", "o"),
    ]
    for date, label, color, marker in markers:
        row = daily.loc[daily["date"].eq(date)].iloc[0]
        ax_price.scatter(
            [date],
            [row["price_usd"]],
            color=color,
            marker=marker,
            s=85,
            edgecolor="white",
            linewidth=0.9,
            zorder=6,
        )
        ax_price.annotate(
            label,
            (date, row["price_usd"]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
            color="#374151",
        )

    for y, label, color in [
        (60_000, "$60k reference", "#64748b"),
        (80_000, "$80k reclaim zone", "#7c3aed"),
    ]:
        ax_price.axhline(y, color=color, linewidth=0.9, linestyle="--", alpha=0.65)
        ax_price.text(view["date"].iloc[2], y * 1.01, label, color=color, fontsize=9)

    ax_price.set_yscale("log")
    ax_price.set_ylabel("USD, log scale")
    ax_price.set_title(
        "BTC technical context after future-floor breach: SMA rejection, 50SMA loss, channel",
        fontsize=15,
        pad=12,
    )
    ax_price.grid(True, which="both", alpha=0.20)
    ax_price.legend(loc="upper left", frameon=False, ncols=3)

    ax_ratio.plot(
        view["date"],
        view["ratio_to_future_floor"],
        color="#2563a6",
        linewidth=1.25,
    )
    ax_ratio.axhline(1.0, color="#d94841", linestyle="--", linewidth=1.0)
    ax_ratio.fill_between(
        view["date"],
        view["ratio_to_future_floor"],
        1.0,
        where=view["ratio_to_future_floor"] < 1.0,
        color="#d94841",
        alpha=0.16,
        interpolate=True,
    )
    ax_ratio.set_ylabel("Price / 12m floor")
    ax_ratio.grid(True, alpha=0.2)

    latest_price = float(latest["price_usd"])
    latest_sma50 = float(latest["sma50"])
    latest_sma200 = float(latest["sma200"])
    latest_channel_lower = float(latest["channel_lower"])
    latest_channel_upper = float(latest["channel_upper"])
    latest_future_floor = float(latest["future_floor_usd"])
    latest_same_floor = float(latest["giovanni_floor_usd"])
    rows = [
        ("Spot vs 50SMA", latest_price / latest_sma50 - 1.0, "below 50SMA" if latest_price < latest_sma50 else "above 50SMA"),
        ("Spot vs 200SMA", latest_price / latest_sma200 - 1.0, "below 200SMA" if latest_price < latest_sma200 else "above 200SMA"),
        ("Spot vs channel lower", latest_price / latest_channel_lower - 1.0, "near/below lower channel" if latest_price < latest_channel_lower * 1.03 else "inside channel"),
        ("Spot vs 12m floor", latest_price / latest_future_floor - 1.0, "below future floor" if latest_price < latest_future_floor else "above future floor"),
        ("Spot vs same-day floor", latest_price / latest_same_floor - 1.0, "above hard floor" if latest_price > latest_same_floor else "below hard floor"),
    ]
    colors = ["#b42318" if value < 0 else "#2b8a5f" for _, value, _ in rows]
    x = np.arange(len(rows))
    values = [value * 100.0 for _, value, _ in rows]
    ax_state.bar(x, values, color=colors, width=0.62)
    ax_state.axhline(0, color="#111827", linewidth=0.8)
    ax_state.set_xticks(x)
    ax_state.set_xticklabels([label for label, _, _ in rows], rotation=12, ha="right")
    ax_state.set_ylabel("Latest gap")
    ax_state.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax_state.set_title(
        f"Latest state as of {pd.Timestamp(latest['date']):%Y-%m-%d} "
        f"(last point is provisional public spot)",
        fontsize=12,
    )
    ax_state.grid(True, axis="y", alpha=0.22)
    for idx, (_, value, state) in enumerate(rows):
        pct = value * 100.0
        ax_state.text(
            idx,
            pct + (1.5 if pct >= 0 else -2.2),
            f"{pct:.1f}%\n{state}",
            ha="center",
            va="bottom" if pct >= 0 else "top",
            fontsize=9,
        )

    ax_ratio.set_xlim(START_DATE, pd.Timestamp(latest["date"]) + pd.Timedelta(days=12))
    ax_ratio.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_ratio.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax_price.get_xticklabels():
        label.set_visible(False)

    metrics = pd.DataFrame(
        [
            {
                "as_of_date": pd.Timestamp(latest["date"]),
                "spot_price_usd": latest_price,
                "sma50_usd": latest_sma50,
                "sma200_usd": latest_sma200,
                "channel_lower_usd": latest_channel_lower,
                "channel_upper_usd": latest_channel_upper,
                "future_floor_12m_usd": latest_future_floor,
                "giovanni_same_day_floor_usd": latest_same_floor,
                "future_floor_breach_date": BREACH_DATE,
                "rejection_date": rejection_date,
                "sma50_loss_date": pd.Timestamp(sma50_loss["date"]),
            }
        ]
    )
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(paths.report_dir / "sma_channel_decision_metrics.csv", index=False)
    output = paths.figure_dir / "sma_channel_decision_plot.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(write_plot(ProjectPaths.from_cwd()))
