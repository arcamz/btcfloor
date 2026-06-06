from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from btcfloor.paths import ProjectPaths


CHECKONCHAIN_CHARTS = {
    "lth_mvrv": (
        "https://charts.checkonchain.com/btconchain/unrealised/"
        "mvrv_lth/mvrv_lth_light.html"
    ),
    "sth_mvrv": (
        "https://charts.checkonchain.com/btconchain/unrealised/"
        "mvrv_sth/mvrv_sth_light.html"
    ),
    "sth_mvrv_zscore": (
        "https://charts.checkonchain.com/btconchain/unrealised/"
        "mvrv_sth_zscore/mvrv_sth_zscore_light.html"
    ),
    "lth_realised_loss": (
        "https://charts.checkonchain.com/btconchain/realised/"
        "realisedpnl_bycohort_bands_lth/realisedpnl_bycohort_bands_lth_light.html"
    ),
    "cointime_pricing": (
        "https://charts.checkonchain.com/btconchain/cointime/"
        "cointime_pricing_all_1/cointime_pricing_all_1_light.html"
    ),
}

BITBO_CVDD_URL = "https://charts.bitbo.io/api/v1/cvdd/"
LOOKNODE_CVDD_URL = "https://www.looknode.com/api/CVDD"

CYCLE_LOW_ANCHORS = {
    "2015 low": pd.Timestamp("2015-01-14"),
    "2018 low": pd.Timestamp("2018-12-15"),
    "2022 low": pd.Timestamp("2022-11-21"),
    "2026 expected low": pd.Timestamp("2026-10-19"),
}

CYCLE_COLORS = {
    "2015 low": "#8f99a8",
    "2018 low": "#5d6c7a",
    "2022 low": "#b45f4d",
    "2026 expected low": "#111111",
}


def _decode_plotly_array(value: object) -> np.ndarray:
    if isinstance(value, dict) and "bdata" in value:
        dtype = np.dtype(str(value.get("dtype", "f8")))
        if dtype.byteorder in ("=", "|"):
            dtype = dtype.newbyteorder("<")
        return np.frombuffer(base64.b64decode(str(value["bdata"])), dtype=dtype).astype(float)
    return np.asarray(value, dtype=float)


def _extract_plotly_traces(html: str) -> list[dict[str, object]]:
    marker = "Plotly.newPlot"
    start = html.find(marker)
    if start < 0:
        raise ValueError("No Plotly.newPlot payload found")
    array_start = html.find("[", start)
    traces, _ = json.JSONDecoder().raw_decode(html[array_start:])
    return traces


def fetch_checkonchain_chart(url: str) -> pd.DataFrame:
    response = requests.get(url, headers={"User-Agent": "btcfloor/0.1"}, timeout=45)
    response.raise_for_status()

    merged: pd.DataFrame | None = None
    for trace in _extract_plotly_traces(response.text):
        name = str(trace.get("name", "unnamed")).strip()
        if not name:
            continue
        dates = pd.to_datetime(trace.get("x", []), errors="coerce").tz_localize(None)
        values = _decode_plotly_array(trace.get("y", []))
        size = min(len(dates), len(values))
        if size == 0:
            continue

        frame = (
            pd.DataFrame({"date": dates[:size], name: values[:size]})
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["date", name])
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
        )
        if frame.empty:
            continue
        merged = frame if merged is None else merged.merge(frame, on="date", how="outer")

    if merged is None or merged.empty:
        raise ValueError(f"No chart series decoded from {url}")
    return merged.sort_values("date").reset_index(drop=True)


def fetch_bitbo_cvdd() -> tuple[pd.DataFrame, dict[str, object]]:
    headers = {"User-Agent": "btcfloor/0.1"}
    api_key = os.environ.get("BITBO_API_KEY")
    if not api_key:
        return pd.DataFrame(columns=["date", "CVDD"]), {
            "url": BITBO_CVDD_URL,
            "available": False,
            "reason": "missing BITBO_API_KEY; classic Woo/Bitbo CVDD is not fetched",
        }
    headers["Authorization"] = f"Bearer {api_key}"
    headers["X-API-Key"] = api_key
    response = requests.get(BITBO_CVDD_URL, headers=headers, timeout=45)
    if response.status_code == 401:
        return pd.DataFrame(columns=["date", "CVDD"]), {
            "url": BITBO_CVDD_URL,
            "available": False,
            "reason": "unauthorized; set BITBO_API_KEY to enable this optional overlay",
        }
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    data = pd.DataFrame(rows, columns=["date", "CVDD"])
    if data.empty:
        return data, {"url": BITBO_CVDD_URL, "available": False, "reason": "empty response"}
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["CVDD"] = pd.to_numeric(data["CVDD"], errors="coerce")
    data = data.dropna(subset=["date", "CVDD"]).sort_values("date").reset_index(drop=True)
    return data, {"url": BITBO_CVDD_URL, "available": True, "reason": None}


def fetch_looknode_cvdd() -> tuple[pd.DataFrame, dict[str, object]]:
    response = requests.get(
        LOOKNODE_CVDD_URL,
        headers={
            "User-Agent": "btcfloor/0.1",
            "Referer": "https://www.looknode.com/charts?chartId=CVDD",
        },
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 100 or not payload.get("data"):
        return pd.DataFrame(columns=["date", "CVDD"]), {
            "url": LOOKNODE_CVDD_URL,
            "available": False,
            "reason": f"unexpected response code {payload.get('code')}",
        }

    data = pd.DataFrame(payload["data"])
    if data.empty or not {"t", "v"}.issubset(data.columns):
        return pd.DataFrame(columns=["date", "CVDD"]), {
            "url": LOOKNODE_CVDD_URL,
            "available": False,
            "reason": "empty or incompatible response",
        }
    data["date"] = pd.to_datetime(data["t"], unit="ms", errors="coerce")
    data["CVDD"] = pd.to_numeric(data["v"], errors="coerce")
    data = (
        data.loc[:, ["date", "CVDD"]]
        .dropna(subset=["date", "CVDD"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return data, {
        "url": LOOKNODE_CVDD_URL,
        "available": True,
        "reason": None,
        "formula": (
            "Looknode describes CVDD as cumulative USD value-days destroyed "
            "divided by BTC market age times 6,000,000, matching the classic "
            "Willy Woo formula family."
        ),
    }


def load_cohort_data() -> pd.DataFrame:
    lth = fetch_checkonchain_chart(CHECKONCHAIN_CHARTS["lth_mvrv"])
    sth = fetch_checkonchain_chart(CHECKONCHAIN_CHARTS["sth_mvrv"])
    sth_z = fetch_checkonchain_chart(CHECKONCHAIN_CHARTS["sth_mvrv_zscore"])
    lth_loss = fetch_checkonchain_chart(CHECKONCHAIN_CHARTS["lth_realised_loss"])
    cointime = fetch_checkonchain_chart(CHECKONCHAIN_CHARTS["cointime_pricing"])
    cvdd, cvdd_status = fetch_bitbo_cvdd()
    looknode_cvdd_status: dict[str, object] = {
        "url": LOOKNODE_CVDD_URL,
        "available": False,
        "reason": "not needed because Bitbo CVDD was available",
    }
    if cvdd.empty:
        cvdd, looknode_cvdd_status = fetch_looknode_cvdd()
    load_cohort_data.cvdd_status = {
        "bitbo": cvdd_status,
        "looknode": looknode_cvdd_status,
    }

    lth_loss = lth_loss.rename(
        columns={
            "Price": "LTH Loss Chart Price",
            "Realised Profit": "LTH Realised Profit BTC",
            "Realised Loss": "LTH Realised Loss Raw BTC",
            "Loss +1sd": "LTH Realised Loss +1sd Raw BTC",
            "Loss +2sd": "LTH Realised Loss +2sd Raw BTC",
        }
    )
    lth_loss["LTH Realised Loss BTC"] = (
        -pd.to_numeric(lth_loss["LTH Realised Loss Raw BTC"], errors="coerce")
    ).clip(lower=0.0)
    lth_loss["LTH Realised Loss +1sd BTC"] = (
        -pd.to_numeric(lth_loss["LTH Realised Loss +1sd Raw BTC"], errors="coerce")
    ).clip(lower=0.0)
    lth_loss["LTH Realised Loss +2sd BTC"] = (
        -pd.to_numeric(lth_loss["LTH Realised Loss +2sd Raw BTC"], errors="coerce")
    ).clip(lower=0.0)
    lth_loss["LTH Realised Loss 7D EMA BTC"] = (
        lth_loss["LTH Realised Loss BTC"].ewm(span=7, adjust=False).mean()
    )
    lth_loss["LTH Realised Loss 28D EMA BTC"] = (
        lth_loss["LTH Realised Loss BTC"].ewm(span=28, adjust=False).mean()
    )

    data = lth.merge(
        sth.drop(columns=["Price"], errors="ignore"),
        on="date",
        how="outer",
    ).merge(
        sth_z.drop(columns=["Price", "STH Realised Price"], errors="ignore"),
        on="date",
        how="outer",
    ).merge(
        lth_loss.drop(
            columns=[
                "Euphoria",
                "Enthusiasm",
                "Fear",
                "Capitulation",
                "Profit +2sd",
                "Profit +1sd",
            ],
            errors="ignore",
        ),
        on="date",
        how="outer",
    ).merge(
        cointime.loc[:, ["date", "Cointime Price"]].dropna(),
        on="date",
        how="outer",
    )
    if not cvdd.empty:
        data = data.merge(cvdd, on="date", how="outer")
    return data.sort_values("date").reset_index(drop=True)


def _latest_valid(data: pd.DataFrame, column: str) -> tuple[pd.Timestamp, float]:
    valid = data.loc[data[column].notna(), ["date", column]]
    row = valid.iloc[-1]
    return pd.Timestamp(row["date"]), float(row[column])


def _usd_label(value: float) -> str:
    if abs(value) >= 1000:
        return f"${value / 1000:.1f}k"
    return f"${value:.0f}"


def _cvdd_display_name() -> str:
    status = getattr(load_cohort_data, "cvdd_status", {})
    bitbo = status.get("bitbo", {})
    looknode = status.get("looknode", {})
    if bitbo.get("available"):
        return "CVDD (Bitbo)"
    if looknode.get("available"):
        return "CVDD (Looknode fallback)"
    return "CVDD"


def plot_current_bands(data: pd.DataFrame, out_path: Path) -> None:
    latest_date, latest_price = _latest_valid(data, "Price")
    start_date = max(data["date"].min(), latest_date - pd.Timedelta(days=520))
    view = data.loc[data["date"].between(start_date, latest_date)].copy()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    bands = [
        ("STH Realised Price", "STH Realised Price", "#5d6c7a", 1.5, "-"),
        ("Price -0.5sd", "Price -0.5sd", "#e0a23b", 1.2, "--"),
        ("Price -1.0sd", "Price -1.0sd", "#c96f52", 1.2, "--"),
        ("Price -1.5sd", "Price -1.5sd", "#9f5b79", 1.2, "--"),
        ("Price -2.0sd", "Price -2.0sd", "#6f5d9f", 1.2, "--"),
        ("Cointime Price", "Cointime Price", "#0891b2", 1.25, "-."),
        ("CVDD", _cvdd_display_name(), "#9333ea", 1.25, "-."),
        ("LTH Realised Price", "LTH Realised Price", "#777777", 1.2, ":"),
    ]
    ax.plot(view["date"], view["Price"], color="#111111", linewidth=2.0, label="BTC price")
    for column, _, color, width, style in bands:
        if column in view:
            ax.plot(view["date"], view[column], color=color, linewidth=width, linestyle=style)

    ax.scatter([latest_date], [latest_price], color="#111111", s=30, zorder=5)
    right_labels: list[tuple[str, float, str, bool]] = []
    for column, display_name, color, _, _ in bands:
        if column not in data:
            continue
        _, value = _latest_valid(data, column)
        right_labels.append((display_name.replace("Price ", ""), value, color, False))
    right_labels.append(("Price", latest_price, "#111111", True))
    label_y = float("-inf")
    min_gap = 1_650.0
    for text, value, color, bold in sorted(right_labels, key=lambda item: item[1]):
        label_y = max(value, label_y + min_gap)
        ax.plot(
            [latest_date + pd.Timedelta(days=3), latest_date + pd.Timedelta(days=7)],
            [value, label_y],
            color=color,
            linewidth=0.6,
            alpha=0.55,
        )
        ax.text(
            latest_date + pd.Timedelta(days=8),
            label_y,
            f"{text}: {_usd_label(value)}",
            color=color,
            fontsize=9,
            va="center",
            fontweight="bold" if bold else "normal",
        )

    _, sth_z = _latest_valid(data, "STH-MVRV Z-Score")
    _, sth_mvrv = _latest_valid(data, "STH-MVRV")
    _, lth_mvrv = _latest_valid(data, "LTH-MVRV")
    ax.set_title(
        "Checkonchain cohort cost-basis stress",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.96,
        (
            f"Latest {latest_date:%Y-%m-%d}: STH-MVRV {sth_mvrv:.2f}, "
            f"STH Z {sth_z:.2f}, LTH-MVRV {lth_mvrv:.2f}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        color="#333333",
    )
    ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(lambda value, _: _usd_label(value))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, axis="y", color="#e6e8ec", linewidth=0.8)
    ax.grid(True, axis="x", color="#f1f2f4", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(view["date"].min(), latest_date + pd.Timedelta(days=125))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _weekly_window(
    data: pd.DataFrame,
    column: str,
    anchor: pd.Timestamp,
    min_weeks: int = -56,
    max_weeks: int = 36,
) -> pd.DataFrame:
    frame = data.loc[:, ["date", column]].dropna().copy()
    frame = frame.loc[
        frame["date"].between(anchor + pd.Timedelta(weeks=min_weeks), anchor + pd.Timedelta(weeks=max_weeks))
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


def plot_cycle_low_panels(data: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("STH-MVRV", "Recent buyers underwater below 1.0", [1.0], (0.5, 1.55)),
        (
            "STH-MVRV Z-Score",
            "Short-holder stress, standardized",
            [0.0, -1.0, -1.5],
            (-2.6, 1.6),
        ),
        ("LTH-MVRV", "Long-holder cushion, capitulation near/below 1.0", [1.0], (0.5, 5.0)),
        ("LTH-SOPR", "Long-holder spent coins at profit/loss", [1.0], (0.45, 2.25)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    axes = axes.ravel()
    fig.patch.set_facecolor("white")

    for ax, (metric, subtitle, refs, ylim) in zip(axes, metrics):
        for label, anchor in CYCLE_LOW_ANCHORS.items():
            window = _weekly_window(data, metric, anchor)
            if window.empty:
                continue
            ax.plot(
                window["weeks_from_low"],
                window[metric],
                color=CYCLE_COLORS[label],
                linewidth=2.6 if label.startswith("2026") else 1.6,
                alpha=1.0 if label.startswith("2026") else 0.75,
                label=label,
            )
        for ref in refs:
            ax.axhline(ref, color="#c9ced6", linewidth=0.9, linestyle="--")
        ax.axvline(0, color="#222222", linewidth=0.8, alpha=0.55)
        ax.set_title(metric, loc="left", fontsize=12, fontweight="bold")
        ax.text(0.01, 0.91, subtitle, transform=ax.transAxes, fontsize=9, color="#555555")
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", color="#e8eaee", linewidth=0.8)
        ax.grid(True, axis="x", color="#f2f3f5", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-56, 36)

    axes[2].set_xlabel("Weeks from actual/expected cycle low")
    axes[3].set_xlabel("Weeks from actual/expected cycle low")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, frameon=False, bbox_to_anchor=(0.62, 0.965))
    fig.suptitle(
        "Checkonchain Cohort Low Windows",
        x=0.05,
        y=0.965,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.05,
        0.925,
        "Y-axes are clipped to the low-tracking range so capitulation zones remain visible.",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_lth_realised_loss_cycle(data: pd.DataFrame, out_path: Path) -> None:
    metric = "LTH Realised Loss 7D EMA BTC"
    fig, ax = plt.subplots(figsize=(15, 7.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for label, anchor in CYCLE_LOW_ANCHORS.items():
        window = _weekly_window(data, metric, anchor, min_weeks=-60, max_weeks=36)
        if window.empty:
            continue
        color = "#d62728" if label.startswith("2026") else CYCLE_COLORS[label]
        ax.plot(
            window["weeks_from_low"],
            window[metric],
            color=color,
            linewidth=2.8 if label.startswith("2026") else 1.8,
            alpha=1.0 if label.startswith("2026") else 0.85,
            label=label,
        )

    ax.axvline(0, color="#2563a6", linewidth=1.0, linestyle="--")
    ax.set_title(
        "LTH Realised Loss Cycle View, reanchored to Oct 2026 expected low",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Weeks from actual / expected cycle low")
    ax.set_ylabel("7D EMA realised loss, BTC")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(-60, 36)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_low_signal_compare(data: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("LTH-MVRV", "Long-holder cost-basis stress", [1.0], (0.5, 5.0)),
        ("LTH-SOPR", "Long-holder spent coins at profit/loss", [1.0], (0.45, 2.25)),
        (
            "LTH Realised Loss 7D EMA BTC",
            "Long-holder realised loss impulse",
            [],
            None,
        ),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14.5, 10.5), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, (metric, subtitle, refs, ylim) in zip(axes, metrics):
        for label, anchor in CYCLE_LOW_ANCHORS.items():
            window = _weekly_window(data, metric, anchor, min_weeks=-60, max_weeks=36)
            if window.empty:
                continue
            color = "#d62728" if label.startswith("2026") else CYCLE_COLORS[label]
            ax.plot(
                window["weeks_from_low"],
                window[metric],
                color=color,
                linewidth=2.8 if label.startswith("2026") else 1.65,
                alpha=1.0 if label.startswith("2026") else 0.75,
                label=label,
            )
        for ref in refs:
            ax.axhline(ref, color="#c9ced6", linewidth=0.9, linestyle="--")
        ax.axvline(0, color="#222222", linewidth=0.8, alpha=0.55)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(metric, loc="left", fontsize=12, fontweight="bold")
        ax.text(0.01, 0.87, subtitle, transform=ax.transAxes, fontsize=9, color="#555555")
        ax.grid(True, axis="y", color="#e8eaee", linewidth=0.8)
        ax.grid(True, axis="x", color="#f2f3f5", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-60, 36)

    axes[-1].set_xlabel("Weeks from actual/expected cycle low")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, frameon=False, bbox_to_anchor=(0.58, 0.975))
    fig.suptitle(
        "LTH Low-Tracking Signal Comparison",
        x=0.05,
        y=0.975,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(data: pd.DataFrame, out_path: Path) -> None:
    fields = [
        "Price",
        "STH Realised Price",
        "STH-MVRV",
        "STH-MVRV Z-Score",
        "Price -1.0sd",
        "Price -1.5sd",
        "Price -2.0sd",
        "LTH Realised Price",
        "LTH True Realised Price",
        "Cointime Price",
        "CVDD",
        "LTH-MVRV",
        "LTH-SOPR",
        "LTH Loss Chart Price",
        "LTH Realised Loss BTC",
        "LTH Realised Loss 7D EMA BTC",
        "LTH Realised Loss 28D EMA BTC",
        "LTH Realised Loss +1sd BTC",
        "LTH Realised Loss +2sd BTC",
    ]
    payload = {
        "source": {
            **CHECKONCHAIN_CHARTS,
            "cvdd": getattr(
                load_cohort_data,
                "cvdd_status",
                {
                    "bitbo": {
                        "url": BITBO_CVDD_URL,
                        "available": False,
                        "reason": "not loaded",
                    },
                    "looknode": {
                        "url": LOOKNODE_CVDD_URL,
                        "available": False,
                        "reason": "not loaded",
                    },
                },
            ),
        },
        "latest": {},
    }
    for field in fields:
        if field not in data:
            continue
        date, value = _latest_valid(data, field)
        payload["latest"][field] = {"date": f"{date:%Y-%m-%d}", "value": value}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    paths = ProjectPaths.from_cwd()
    paths.ensure_dirs()
    data = load_cohort_data()
    data.to_csv(paths.processed_dir / "checkonchain_cohort_metrics.csv", index=False)
    plot_current_bands(data, paths.figure_dir / "checkonchain_cohort_current_bands.png")
    plot_cycle_low_panels(data, paths.figure_dir / "checkonchain_cohort_cycle_lows.png")
    plot_lth_realised_loss_cycle(data, paths.figure_dir / "checkonchain_lth_realised_loss_cycle.png")
    plot_low_signal_compare(data, paths.figure_dir / "checkonchain_low_signal_compare.png")
    write_summary(data, paths.report_dir / "checkonchain_cohort_summary.json")


if __name__ == "__main__":
    main()
