from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from btcfloor.paths import ProjectPaths


COINMETRICS_BTC_CSV_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
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
    return load_price_history(paths.processed_btc_csv)


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
    ax.set_title("BTC PriceUSD, Coin Metrics daily series")
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
