from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import requests

from btcfloor.paths import ProjectPaths
from plot_checkonchain_cohorts import CYCLE_LOW_ANCHORS, load_cohort_data, write_summary


BITSTAMP_OHLC = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
FIRST_BITSTAMP_DAY = pd.Timestamp("2011-08-18")
OUTPUT_NAME = "btc_daily_market_dashboard.html"
ASSETS = Path(__file__).resolve().parents[1] / "resources" / "daily_market"
OHLC = ("open", "high", "low", "close")


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    group: str
    pane: str
    color: str
    source: str
    column: str = ""
    default: bool = False
    dashed: bool = False


# These are the daily source columns used by the existing weekly dashboard and
# LTH study, plus the display indicators calculated from Bitstamp candles.
METRICS = (
    Metric("sma40", "SMA 40d", "Technical", "price", "orange", "bitstamp", default=True),
    Metric("sma200", "SMA 200d", "Technical", "price", "cyan", "bitstamp"),
    Metric("sma50w", "SMA 50w", "Technical", "price", "pink", "bitstamp"),
    Metric("bands", "Z-score bands · EMA27 ± 1.7σ", "Technical", "price", "violet", "bitstamp", default=True, dashed=True),
    Metric("cointime", "Cointime Price", "Price references", "price", "cyan", "cointime_pricing", "Cointime Price"),
    Metric("cvdd", "CVDD", "Price references", "price", "violet", "cvdd", "CVDD", dashed=True),
    Metric("sth_price", "STH Realised Price", "Short-term holders", "price", "blue", "sth_mvrv", "STH Realised Price", True),
    Metric("sth_10", "STH price −1.0σ", "Short-term holders", "price", "gold", "sth_mvrv_zscore", "Price -1.0sd", dashed=True),
    Metric("sth_15", "STH price −1.5σ", "Short-term holders", "price", "coral", "sth_mvrv_zscore", "Price -1.5sd", dashed=True),
    Metric("sth_20", "STH price −2.0σ", "Short-term holders", "price", "pink", "sth_mvrv_zscore", "Price -2.0sd", dashed=True),
    Metric("sth_mvrv", "STH MVRV", "Short-term holders", "ratio", "blue", "sth_mvrv", "STH-MVRV"),
    Metric("sth_z", "STH MVRV Z-score", "Short-term holders", "z", "violet", "sth_mvrv_zscore", "STH-MVRV Z-Score"),
    Metric("lth_price", "LTH Realised Price", "Long-term holders", "price", "green", "lth_mvrv", "LTH Realised Price", True),
    Metric("lth_true", "LTH True Realised Price", "Long-term holders", "price", "teal", "lth_mvrv", "LTH True Realised Price", dashed=True),
    Metric("lth_mvrv", "LTH MVRV", "Long-term holders", "ratio", "green", "lth_mvrv", "LTH-MVRV"),
    Metric("lth_sopr", "LTH SOPR", "Long-term holders", "ratio", "orange", "lth_mvrv", "LTH-SOPR"),
    Metric("lth_aviv", "LTH AVIV", "Long-term holders", "ratio", "teal", "lth_mvrv", "LTH-AVIV"),
    Metric("lth_loss", "LTH Realised Loss", "Long-term holders", "btc", "coral", "lth_realised_loss", "LTH Realised Loss BTC"),
    Metric("lth_loss7", "LTH Realised Loss · 7d EMA", "Long-term holders", "btc", "pink", "lth_realised_loss", "LTH Realised Loss 7D EMA BTC"),
    Metric("lth_loss28", "LTH Realised Loss · 28d EMA", "Long-term holders", "btc", "violet", "lth_realised_loss", "LTH Realised Loss 28D EMA BTC"),
)


def fetch_candles(through: pd.Timestamp) -> pd.DataFrame:
    pages = []
    cursor = FIRST_BITSTAMP_DAY
    with requests.Session() as session:
        while cursor <= through:
            end = min(cursor + pd.Timedelta(days=999), through)
            response = session.get(
                BITSTAMP_OHLC,
                params={
                    "step": 86400,
                    "limit": (end - cursor).days + 1,
                    "start": int(cursor.tz_localize("UTC").timestamp()),
                    "end": int(end.tz_localize("UTC").timestamp()),
                    "exclude_current_candle": "true",
                },
                timeout=45,
            )
            response.raise_for_status()
            rows = pd.DataFrame(response.json()["data"]["ohlc"])
            if rows.empty:
                raise ValueError(f"Bitstamp returned no candles for {cursor.date()} to {end.date()}")
            dates = pd.to_datetime(rows["timestamp"].astype("int64"), unit="s")
            # Bitstamp may return an extra row when querying a one-day range.
            rows = rows.assign(date=dates).loc[dates.between(cursor, end)]
            pages.append(rows[["date", *OHLC]])
            cursor = end + pd.Timedelta(days=1)
    return validate_candles(pd.concat(pages, ignore_index=True))


def validate_candles(candles: pd.DataFrame) -> pd.DataFrame:
    candles = candles[["date", *OHLC]].copy()
    candles["date"] = pd.to_datetime(candles["date"])
    prices = candles[list(OHLC)].apply(pd.to_numeric)
    if (
        candles.empty
        or candles["date"].isna().any()
        or candles["date"].duplicated().any()
        or not candles["date"].eq(candles["date"].dt.normalize()).all()
        or not np.isfinite(prices).all().all()
        or (prices <= 0).any().any()
        or (prices["low"] > prices[["open", "close"]].min(axis=1)).any()
        or (prices["high"] < prices[["open", "close"]].max(axis=1)).any()
    ):
        raise ValueError("Invalid Bitstamp daily OHLC")
    return candles.assign(**prices).sort_values("date").reset_index(drop=True)


def technical_indicators(candles: pd.DataFrame) -> dict[str, pd.Series]:
    close = candles["close"]
    basis = close.ewm(span=27, adjust=False).mean()
    deviation = close.rolling(27).std(ddof=0)
    week = candles["date"].dt.to_period("W-SUN")
    weekly = close.groupby(week).agg(["last", "size"])
    # Reindex so a missing week also breaks the 50-completed-week run.
    weekly = weekly.reindex(pd.period_range(weekly.index[0], weekly.index[-1], freq="W-SUN"))
    completed = weekly["last"].where(weekly["size"].eq(7)).rolling(50).mean()
    last_complete = weekly["size"].eq(7)
    if not last_complete.iloc[-1]:
        completed = completed.iloc[:-1]
    completed.index = completed.index.end_time.normalize()
    completed = completed.reindex(pd.DatetimeIndex(candles["date"]), method="ffill")
    return {
        "sma40": close.rolling(40).mean(),
        "sma200": close.rolling(200).mean(),
        "sma50w": pd.Series(completed.to_numpy(), index=close.index),
        "basis": basis,
        "upper": basis + 1.7 * deviation,
        "lower": basis - 1.7 * deviation,
        "zscore": ((close - basis) / deviation).mask(deviation.eq(0), 0),
    }


def payload_for(candles: pd.DataFrame, cohort: pd.DataFrame, summary: dict) -> dict:
    candles = validate_candles(candles)
    dates = pd.DatetimeIndex(candles["date"])
    cohort = cohort.copy()
    cohort["date"] = pd.to_datetime(cohort["date"]).dt.normalize()
    aligned = cohort.set_index("date")
    if not aligned.index.is_unique:
        raise ValueError("Duplicate Checkonchain UTC dates")
    aligned = aligned.reindex(dates)
    indicators = technical_indicators(candles)
    metrics = []
    sources = summary.get("source", {})
    cvdd_sources = sources.get("cvdd", {})
    for metric in METRICS:
        label = metric.label
        source = "Bitstamp BTC/USD" if metric.source == "bitstamp" else "Checkonchain"
        if metric.key == "bands":
            series = {name: indicators[name] for name in ("basis", "upper", "lower", "zscore")}
        elif metric.column:
            if metric.key == "cvdd":
                if cvdd_sources.get("bitbo", {}).get("available"):
                    label, source = "CVDD (Bitbo)", "Bitbo"
                elif cvdd_sources.get("looknode", {}).get("available"):
                    label, source = "CVDD (Looknode fallback)", "Looknode"
                else:
                    source = "CVDD provider unavailable"
            series = {metric.key: aligned.get(metric.column)}
        else:
            series = {metric.key: indicators[metric.key]}
        values = {
            name: (
                [float(x) if np.isfinite(x) else None for x in pd.to_numeric(column, errors="coerce")]
                if column is not None else [None] * len(dates)
            )
            for name, column in series.items()
        }
        observed = [i for i, row in enumerate(zip(*values.values())) if any(x is not None for x in row)]
        metrics.append({
            "key": metric.key, "label": label, "group": metric.group, "pane": metric.pane,
            "color": metric.color, "source": source, "default": metric.default,
            "dashed": metric.dashed, "values": values,
            "first": dates[observed[0]].date().isoformat() if observed else None,
            "last": dates[observed[-1]].date().isoformat() if observed else None,
        })
    return {
        "dates": dates.strftime("%Y-%m-%d").tolist(),
        "candles": candles[list(OHLC)].to_numpy().tolist(),
        "metrics": metrics,
        "cycle_lows": [{"label": name, "date": day.date().isoformat()} for name, day in CYCLE_LOW_ANCHORS.items()],
    }


def build_dashboard(paths: ProjectPaths, *, refresh_cohorts: bool = False) -> Path:
    paths.ensure_dirs()
    cohort_path = paths.processed_dir / "checkonchain_cohort_metrics.csv"
    summary_path = paths.report_dir / "checkonchain_cohort_summary.json"
    if refresh_cohorts or not cohort_path.exists():
        cohort = load_cohort_data()
        cohort.to_csv(cohort_path, index=False)
        write_summary(cohort, summary_path)
    else:
        cohort = pd.read_csv(cohort_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    through = pd.Timestamp(datetime.now(UTC).date()) - pd.Timedelta(days=1)
    payload = payload_for(fetch_candles(through), cohort, summary)
    html = Template((ASSETS / "dashboard.html").read_text(encoding="utf-8")).substitute(
        styles=(ASSETS / "dashboard.css").read_text(encoding="utf-8"),
        vendor=(ASSETS / "lightweight-charts.js").read_text(encoding="utf-8"),
        license_text=(ASSETS / "LIGHTWEIGHT-CHARTS-LICENSE.txt").read_text(encoding="utf-8"),
        script=(ASSETS / "dashboard.js").read_text(encoding="utf-8"),
        payload=json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c"),
    )
    output = paths.interactive_dir / OUTPUT_NAME
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the standalone BTC daily market dashboard.")
    parser.add_argument("--refresh-cohorts", action="store_true", help="Fetch the project's on-chain inputs again.")
    args = parser.parse_args()
    print(build_dashboard(ProjectPaths.from_cwd(), refresh_cohorts=args.refresh_cohorts))


if __name__ == "__main__":
    main()
