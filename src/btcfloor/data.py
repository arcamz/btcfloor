from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from btcfloor.paths import ProjectPaths


COINMETRICS_BTC_CSV_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
COINGECKO_BTC_MARKET_CHART_RANGE_URL = (
    "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
)
GENESIS_DATE = pd.Timestamp("2009-01-03")
PRICE_FIX_COLUMNS = ("date", "action", "price_usd", "reason")


@dataclass(frozen=True)
class PriceQualityReport:
    source_rows: int
    valid_price_rows: int
    first_raw_date: pd.Timestamp
    last_raw_date: pd.Timestamp
    first_valid_price_date: pd.Timestamp
    last_valid_price_date: pd.Timestamp
    zero_or_negative_rows: int
    configured_price_fix_rows: int
    applied_manual_price_rows: int
    processed_source_counts: dict[str, int]
    missing_calendar_days: int
    suspicious_return_days: int
    suspicious_return_sample: pd.DataFrame

    def to_markdown(self) -> str:
        sample = self.suspicious_return_sample.copy()
        if not sample.empty:
            sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
            sample["log_return"] = sample["log_return"].map(lambda x: f"{x:.4f}")
            sample["price_usd"] = sample["price_usd"].map(lambda x: f"{x:,.6f}")
            sample_md = sample.to_markdown(index=False)
        else:
            sample_md = "No daily absolute log returns above the diagnostic threshold."
        source_counts_md = "\n".join(
            f"  - {source}: {count:,}"
            for source, count in sorted(self.processed_source_counts.items())
        )

        return "\n".join(
            [
                "# BTC Price Data Quality",
                "",
                f"- Source rows: {self.source_rows:,}",
                f"- Valid positive daily price rows: {self.valid_price_rows:,}",
                f"- Raw date range: {self.first_raw_date:%Y-%m-%d} to {self.last_raw_date:%Y-%m-%d}",
                (
                    "- First valid positive PriceUSD date: "
                    f"{self.first_valid_price_date:%Y-%m-%d}"
                ),
                (
                    "- Last valid positive PriceUSD date: "
                    f"{self.last_valid_price_date:%Y-%m-%d}"
                ),
                f"- Zero or negative PriceUSD rows: {self.zero_or_negative_rows:,}",
                f"- Configured manual price fix rows: {self.configured_price_fix_rows:,}",
                f"- Applied manual replacement/insert rows: {self.applied_manual_price_rows:,}",
                "- Processed source rows:",
                source_counts_md,
                f"- Missing calendar days inside valid range: {self.missing_calendar_days:,}",
                (
                    "- Suspicious daily return rows "
                    f"(|log return| > 1.0): {self.suspicious_return_days:,}"
                ),
                "",
                "## Suspicious Return Sample",
                "",
                sample_md,
                "",
            ]
        )


def fetch_coinmetrics_btc_csv(
    raw_path: Path,
    url: str = COINMETRICS_BTC_CSV_URL,
    timeout_seconds: int = 90,
) -> Path:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    raw_path.write_bytes(response.content)
    return raw_path


def _date_to_utc_epoch_seconds(date: pd.Timestamp) -> int:
    normalized = pd.Timestamp(date).normalize()
    return int(normalized.tz_localize("UTC").timestamp())


def fetch_coingecko_recent_btc_prices(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp | None = None,
    url: str = COINGECKO_BTC_MARKET_CHART_RANGE_URL,
    timeout_seconds: int = 45,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = (
        pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        if end_date is None
        else pd.Timestamp(end_date).normalize()
    )
    if start > end:
        return pd.DataFrame(columns=["date", "days_since_genesis", "price_usd", "source"])

    params = {
        "vs_currency": "usd",
        "from": _date_to_utc_epoch_seconds(start),
        "to": _date_to_utc_epoch_seconds(end + pd.Timedelta(days=1)),
    }
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": "btcfloor/0.1"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return normalize_coingecko_price_points(response.json().get("prices", []), start, end)


def normalize_coingecko_price_points(
    prices: list[list[float]] | tuple[tuple[float, float], ...],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if not prices:
        return pd.DataFrame(columns=["date", "days_since_genesis", "price_usd", "source"])

    ticks = pd.DataFrame(prices, columns=["timestamp_ms", "price_usd"])
    ticks["timestamp"] = pd.to_datetime(
        pd.to_numeric(ticks["timestamp_ms"], errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    )
    ticks["price_usd"] = pd.to_numeric(ticks["price_usd"], errors="coerce")
    ticks = ticks.dropna(subset=["timestamp", "price_usd"])
    ticks = ticks.loc[ticks["price_usd"] > 0].sort_values("timestamp")
    if ticks.empty:
        return pd.DataFrame(columns=["date", "days_since_genesis", "price_usd", "source"])

    ticks["date"] = ticks["timestamp"].dt.tz_convert(None).dt.normalize()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    ticks = ticks.loc[ticks["date"].between(start, end)].copy()
    if ticks.empty:
        return pd.DataFrame(columns=["date", "days_since_genesis", "price_usd", "source"])

    daily = (
        ticks.groupby("date", as_index=False)
        .agg(price_usd=("price_usd", "last"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["days_since_genesis"] = (daily["date"] - GENESIS_DATE).dt.days
    daily = daily.loc[daily["days_since_genesis"] > 0].copy()
    daily["source"] = "coingecko_market_chart_range"
    return daily.loc[:, ["date", "days_since_genesis", "price_usd", "source"]]


def append_missing_recent_market_prices(
    daily: pd.DataFrame,
    recent: pd.DataFrame,
) -> pd.DataFrame:
    if recent.empty:
        return daily.sort_values("date").reset_index(drop=True)

    base = daily.copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    latest_base_date = base["date"].max()
    additions = recent.loc[pd.to_datetime(recent["date"]).dt.normalize() > latest_base_date]
    if additions.empty:
        return base.sort_values("date").reset_index(drop=True)

    combined = pd.concat([base, additions], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined["price_usd"] = pd.to_numeric(combined["price_usd"], errors="coerce")
    combined = combined.dropna(subset=["date", "price_usd"])
    combined = combined.loc[combined["price_usd"] > 0].copy()
    combined["days_since_genesis"] = (combined["date"] - GENESIS_DATE).dt.days
    combined = combined.loc[combined["days_since_genesis"] > 0].copy()
    return combined.loc[:, ["date", "days_since_genesis", "price_usd", "source"]].sort_values(
        "date"
    ).reset_index(drop=True)


def extend_processed_history_to_today(
    processed_path: Path,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    daily = load_price_history(processed_path)
    latest_date = pd.Timestamp(daily["date"].max()).normalize()
    target_date = (
        pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        if end_date is None
        else pd.Timestamp(end_date).normalize()
    )
    start_date = latest_date + pd.Timedelta(days=1)
    if start_date > target_date:
        return daily

    recent = fetch_coingecko_recent_btc_prices(start_date, target_date)
    extended = append_missing_recent_market_prices(daily, recent)
    extended.to_csv(processed_path, index=False)
    return extended


def ensure_price_fixes_file(price_fixes_path: Path) -> Path:
    if not price_fixes_path.exists():
        price_fixes_path.parent.mkdir(parents=True, exist_ok=True)
        price_fixes_path.write_text(",".join(PRICE_FIX_COLUMNS) + "\n", encoding="utf-8")
    return price_fixes_path


def load_price_fixes(price_fixes_path: Path) -> pd.DataFrame:
    if not price_fixes_path.exists():
        return pd.DataFrame(columns=PRICE_FIX_COLUMNS)

    fixes = pd.read_csv(price_fixes_path)
    missing = set(PRICE_FIX_COLUMNS).difference(fixes.columns)
    if missing:
        raise ValueError(f"Price fixes file missing columns: {sorted(missing)}")
    fixes = fixes.loc[:, PRICE_FIX_COLUMNS].dropna(how="all")
    if fixes.empty:
        return fixes

    fixes["date"] = pd.to_datetime(fixes["date"], errors="coerce")
    fixes["action"] = fixes["action"].astype(str).str.strip().str.lower()
    fixes["price_usd"] = pd.to_numeric(fixes["price_usd"], errors="coerce")
    fixes["reason"] = fixes["reason"].fillna("").astype(str)

    if fixes["date"].isna().any():
        raise ValueError("All configured price fixes must include a valid date")
    invalid_actions = sorted(set(fixes["action"]).difference({"replace", "drop"}))
    if invalid_actions:
        raise ValueError(f"Unsupported price fix actions: {invalid_actions}")
    replacement = fixes["action"].eq("replace")
    if fixes.loc[replacement, "price_usd"].isna().any():
        raise ValueError("Replacement price fixes must include price_usd")
    if fixes.loc[replacement, "price_usd"].le(0).any():
        raise ValueError("Replacement price_usd values must be positive")
    return fixes


def apply_price_fixes(df: pd.DataFrame, fixes: pd.DataFrame) -> pd.DataFrame:
    if fixes.empty:
        return df

    fixed = df.copy()
    for fix in fixes.itertuples(index=False):
        date = pd.Timestamp(fix.date).normalize()
        action = str(fix.action)
        same_date = fixed["date"].dt.normalize().eq(date)

        if action == "drop":
            fixed = fixed.loc[~same_date].copy()
            continue

        price = float(fix.price_usd)
        if same_date.any():
            fixed.loc[same_date, "price_usd"] = price
            fixed.loc[same_date, "source"] = "coinmetrics_community_csv_manual_replace"
        else:
            fixed = pd.concat(
                [
                    fixed,
                    pd.DataFrame(
                        {
                            "date": [date],
                            "price_usd": [price],
                            "source": ["manual_price_insert"],
                        }
                    ),
                ],
                ignore_index=True,
            )

    fixed["days_since_genesis"] = (fixed["date"] - GENESIS_DATE).dt.days
    fixed = fixed.loc[fixed["days_since_genesis"] > 0].copy()
    fixed = fixed.sort_values("date").reset_index(drop=True)
    return fixed


def normalize_coinmetrics_btc(
    raw_path: Path,
    processed_path: Path,
    price_fixes_path: Path | None = None,
) -> Path:
    raw = pd.read_csv(raw_path)
    if "time" not in raw.columns or "PriceUSD" not in raw.columns:
        raise ValueError("Coin Metrics BTC CSV must contain time and PriceUSD columns")

    df = raw.loc[:, ["time", "PriceUSD"]].rename(
        columns={"time": "date", "PriceUSD": "price_usd"}
    )
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df[df["price_usd"] > 0].copy()
    df["days_since_genesis"] = (df["date"] - GENESIS_DATE).dt.days
    df = df[df["days_since_genesis"] > 0].copy()
    df["source"] = "coinmetrics_community_csv"

    if price_fixes_path is not None:
        ensure_price_fixes_file(price_fixes_path)
        df = apply_price_fixes(df, load_price_fixes(price_fixes_path))

    df = df.loc[:, ["date", "days_since_genesis", "price_usd", "source"]]

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    return processed_path


def load_price_history(processed_path: Path) -> pd.DataFrame:
    df = pd.read_csv(processed_path, parse_dates=["date"])
    required = {"date", "days_since_genesis", "price_usd"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Processed price history missing columns: {sorted(missing)}")
    return df.sort_values("date").reset_index(drop=True)


def download_and_prepare(paths: ProjectPaths, force_download: bool = False) -> pd.DataFrame:
    paths.ensure_dirs()
    ensure_price_fixes_file(paths.price_fixes_csv)
    if force_download or not paths.raw_btc_csv.exists():
        fetch_coinmetrics_btc_csv(paths.raw_btc_csv)
    if force_download or not paths.processed_btc_csv.exists():
        normalize_coinmetrics_btc(
            paths.raw_btc_csv,
            paths.processed_btc_csv,
            price_fixes_path=paths.price_fixes_csv,
        )
    return extend_processed_history_to_today(paths.processed_btc_csv)


def validate_price_history(
    raw_path: Path,
    processed: pd.DataFrame,
    price_fixes_path: Path | None = None,
) -> PriceQualityReport:
    raw = pd.read_csv(raw_path)
    raw["date"] = pd.to_datetime(raw["time"], utc=True, errors="coerce").dt.tz_localize(None)
    raw["price_usd"] = pd.to_numeric(raw["PriceUSD"], errors="coerce")
    raw_dates = raw["date"].dropna()
    zero_or_negative = raw["price_usd"].le(0).fillna(False)

    valid = processed.copy()
    configured_price_fix_rows = (
        0 if price_fixes_path is None else len(load_price_fixes(price_fixes_path))
    )
    applied_manual_price_rows = int(
        valid.get("source", pd.Series(dtype=str))
        .astype(str)
        .str.contains("manual_", regex=False)
        .sum()
    )
    processed_source_counts = (
        valid.get("source", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    )
    full_index = pd.date_range(valid["date"].min(), valid["date"].max(), freq="D")
    missing_calendar_days = len(full_index.difference(pd.DatetimeIndex(valid["date"])))

    valid["log_return"] = np.log(valid["price_usd"]).diff()
    suspicious = valid.loc[
        valid["log_return"].abs() > 1.0,
        ["date", "price_usd", "log_return"],
    ].copy()

    return PriceQualityReport(
        source_rows=len(raw),
        valid_price_rows=len(valid),
        first_raw_date=raw_dates.min(),
        last_raw_date=raw_dates.max(),
        first_valid_price_date=valid["date"].min(),
        last_valid_price_date=valid["date"].max(),
        zero_or_negative_rows=int(zero_or_negative.sum()),
        configured_price_fix_rows=configured_price_fix_rows,
        applied_manual_price_rows=applied_manual_price_rows,
        processed_source_counts={str(k): int(v) for k, v in processed_source_counts.items()},
        missing_calendar_days=missing_calendar_days,
        suspicious_return_days=len(suspicious),
        suspicious_return_sample=suspicious.head(20),
    )


def write_data_quality_report(paths: ProjectPaths, report: PriceQualityReport) -> Path:
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    paths.data_quality_report.write_text(report.to_markdown(), encoding="utf-8")
    return paths.data_quality_report


def to_weekly_close(daily: pd.DataFrame, week_ending: str = "W-SUN") -> pd.DataFrame:
    weekly = daily.sort_values("date").groupby(pd.Grouper(key="date", freq=week_ending))
    weekly = weekly.agg(date=("date", "max"), price_usd=("price_usd", "last")).dropna()
    weekly["days_since_genesis"] = (weekly["date"] - GENESIS_DATE).dt.days
    weekly["source"] = "coinmetrics_community_csv_weekly_close"
    return weekly.loc[:, ["date", "days_since_genesis", "price_usd", "source"]]


def to_weekly_ohlc(daily: pd.DataFrame, week_ending: str = "W-SUN") -> pd.DataFrame:
    weekly = daily.sort_values("date").groupby(pd.Grouper(key="date", freq=week_ending))
    ohlc = weekly.agg(
        date=("date", "max"),
        open=("price_usd", "first"),
        high=("price_usd", "max"),
        low=("price_usd", "min"),
        close=("price_usd", "last"),
    ).dropna()
    ohlc["days_since_genesis"] = (ohlc["date"] - GENESIS_DATE).dt.days
    ohlc["source"] = "coinmetrics_community_csv_weekly_ohlc"
    return ohlc.loc[
        :,
        ["date", "days_since_genesis", "open", "high", "low", "close", "source"],
    ].reset_index(drop=True)


def plot_price_diagnostics(
    daily: pd.DataFrame,
    figure_dir: Path,
    early_end: str = "2013-12-31",
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily["date"], daily["price_usd"], linewidth=1.2)
    ax.set_yscale("log")
    ax.set_title("BTC PriceUSD, processed daily series")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD, log scale")
    ax.grid(True, which="both", alpha=0.25)
    fig.autofmt_xdate()
    out = figure_dir / "btc_price_log.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    outputs.append(out)

    early = daily[daily["date"] <= pd.Timestamp(early_end)]
    if not early.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(early["date"], early["price_usd"], linewidth=1.4)
        ax.set_yscale("log")
        ax.set_title(f"BTC early PriceUSD through {early_end}")
        ax.set_xlabel("Date")
        ax.set_ylabel("USD, log scale")
        ax.grid(True, which="both", alpha=0.25)
        fig.autofmt_xdate()
        out = figure_dir / "btc_price_early_log.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)

    return outputs
