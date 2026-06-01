from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from btcfloor.data import GENESIS_DATE
from btcfloor.powerlaw import PowerLawModel


DEFAULT_OVERLAP_HORIZON_MONTHS = 12


def floor_target_crossing_date(
    model: PowerLawModel,
    target_floor_usd: float,
    start_date: pd.Timestamp | str = GENESIS_DATE + pd.Timedelta(days=1),
    max_years: int = 80,
) -> pd.Timestamp:
    """Return the first daily date where a model floor is at least target_floor_usd."""
    if target_floor_usd <= 0.0:
        raise ValueError("target_floor_usd must be positive")
    if max_years <= 0:
        raise ValueError("max_years must be positive")

    start = pd.Timestamp(start_date).normalize()
    if float(model.predict_price(start, floor=True)[0]) >= target_floor_usd:
        return start

    end = start + pd.DateOffset(years=max_years)
    if float(model.predict_price(end, floor=True)[0]) < target_floor_usd:
        raise ValueError(
            f"{model.name} floor does not reach {target_floor_usd:g} "
            f"within {max_years} years from {start:%Y-%m-%d}"
        )

    low = 0
    high = int((end - start).days)
    while low < high:
        mid = (low + high) // 2
        date = start + pd.Timedelta(days=mid)
        floor_usd = float(model.predict_price(date, floor=True)[0])
        if floor_usd >= target_floor_usd:
            high = mid
        else:
            low = mid + 1

    return start + pd.Timedelta(days=low)


def floor_threshold_signal_dates(
    model: PowerLawModel,
    target_floor_usd: float,
    horizons_months: Iterable[int] = (0, 1, 3, 6, 9, 12),
) -> pd.DataFrame:
    crossing_date = floor_target_crossing_date(model, target_floor_usd)
    rows = []
    for months in horizons_months:
        months = int(months)
        as_of = crossing_date - pd.DateOffset(months=months)
        target = as_of + pd.DateOffset(months=months)
        rows.append(
            {
                "model": model.name,
                "target_floor_usd": float(target_floor_usd),
                "horizon_months": months,
                "as_of_date": as_of.normalize(),
                "target_date": target.normalize(),
                "floor_usd": float(model.predict_price(target, floor=True)[0]),
            }
        )
    return pd.DataFrame(rows)


def future_floor_overlap_daily(
    daily: pd.DataFrame,
    model: PowerLawModel,
    horizon_months: int = DEFAULT_OVERLAP_HORIZON_MONTHS,
) -> pd.DataFrame:
    required = {"date", "price_usd"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"daily frame missing columns: {sorted(missing)}")

    frame = daily.loc[:, ["date", "price_usd"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["target_date"] = frame["date"].map(
        lambda date: date + pd.DateOffset(months=int(horizon_months))
    )
    frame["model"] = model.name
    frame["horizon_months"] = int(horizon_months)
    frame["current_floor_usd"] = model.predict_price(frame["date"], floor=True)
    frame["future_floor_usd"] = model.predict_price(frame["target_date"], floor=True)
    frame["ratio_to_current_floor"] = (
        frame["price_usd"].to_numpy(dtype=float)
        / frame["current_floor_usd"].to_numpy(dtype=float)
    )
    frame["ratio_to_future_floor"] = (
        frame["price_usd"].to_numpy(dtype=float)
        / frame["future_floor_usd"].to_numpy(dtype=float)
    )
    frame["below_current_floor"] = frame["ratio_to_current_floor"] < 1.0
    frame["below_future_floor"] = frame["ratio_to_future_floor"] < 1.0
    return frame


def _empty_overlap_episode_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model",
            "horizon_months",
            "start_date",
            "end_date",
            "calendar_days",
            "breach_days",
            "breach_coverage_pct",
            "start_price_usd",
            "end_price_usd",
            "min_ratio_date",
            "target_date_at_min",
            "min_price_usd",
            "current_floor_at_min_usd",
            "future_floor_at_min_usd",
            "min_ratio_to_current_floor",
            "min_ratio_to_future_floor",
            "below_current_floor_at_min",
        ]
    )


def group_future_floor_overlap_episodes(
    overlap: pd.DataFrame,
    max_gap_days: int = 30,
) -> pd.DataFrame:
    if max_gap_days < 0:
        raise ValueError("max_gap_days must be non-negative")
    if overlap.empty:
        return _empty_overlap_episode_frame()
    required = {
        "date",
        "price_usd",
        "target_date",
        "model",
        "horizon_months",
        "current_floor_usd",
        "future_floor_usd",
        "ratio_to_current_floor",
        "ratio_to_future_floor",
        "below_future_floor",
    }
    missing = required.difference(overlap.columns)
    if missing:
        raise ValueError(f"overlap frame missing columns: {sorted(missing)}")

    below = overlap.loc[overlap["below_future_floor"].astype(bool)].copy()
    if below.empty:
        return _empty_overlap_episode_frame()

    below["date"] = pd.to_datetime(below["date"])
    below = below.sort_values("date").reset_index(drop=True)
    gap_days = below["date"].diff().dt.days.fillna(0).astype(int)
    below["episode_id"] = (gap_days > max_gap_days).cumsum()

    rows = []
    for _, group in below.groupby("episode_id", sort=False):
        start = pd.Timestamp(group["date"].iloc[0])
        end = pd.Timestamp(group["date"].iloc[-1])
        min_idx = group["ratio_to_future_floor"].astype(float).idxmin()
        min_row = group.loc[min_idx]
        calendar_days = int((end - start).days) + 1
        breach_days = int(len(group))
        rows.append(
            {
                "model": str(group["model"].iloc[0]),
                "horizon_months": int(group["horizon_months"].iloc[0]),
                "start_date": start,
                "end_date": end,
                "calendar_days": calendar_days,
                "breach_days": breach_days,
                "breach_coverage_pct": breach_days / calendar_days,
                "start_price_usd": float(group["price_usd"].iloc[0]),
                "end_price_usd": float(group["price_usd"].iloc[-1]),
                "min_ratio_date": pd.Timestamp(min_row["date"]),
                "target_date_at_min": pd.Timestamp(min_row["target_date"]),
                "min_price_usd": float(min_row["price_usd"]),
                "current_floor_at_min_usd": float(min_row["current_floor_usd"]),
                "future_floor_at_min_usd": float(min_row["future_floor_usd"]),
                "min_ratio_to_current_floor": float(min_row["ratio_to_current_floor"]),
                "min_ratio_to_future_floor": float(min_row["ratio_to_future_floor"]),
                "below_current_floor_at_min": bool(
                    float(min_row["ratio_to_current_floor"]) < 1.0
                ),
            }
        )

    return pd.DataFrame(rows)


def add_nearest_cycle_low_context(
    episodes: pd.DataFrame,
    cycle_lows: Iterable[tuple[str, pd.Timestamp]],
    reference_date_column: str = "min_ratio_date",
    near_low_window_days: int = 365,
) -> pd.DataFrame:
    if near_low_window_days <= 0:
        raise ValueError("near_low_window_days must be positive")
    if episodes.empty:
        return episodes.copy()

    lows = [(name, pd.Timestamp(date)) for name, date in cycle_lows]
    if not lows:
        raise ValueError("cycle_lows must not be empty")

    rows = []
    for row in episodes.to_dict(orient="records"):
        reference_date = pd.Timestamp(row[reference_date_column])
        distances = np.array(
            [abs((reference_date - low_date).days) for _, low_date in lows],
            dtype=int,
        )
        nearest_idx = int(np.argmin(distances))
        low_name, low_date = lows[nearest_idx]
        days_from_low = int((reference_date - low_date).days)
        row.update(
            {
                "nearest_cycle_low": low_name,
                "nearest_cycle_low_date": low_date,
                "days_from_nearest_cycle_low": days_from_low,
                "abs_days_from_nearest_cycle_low": abs(days_from_low),
                "near_cycle_low_window": abs(days_from_low) <= near_low_window_days,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
