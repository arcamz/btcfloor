from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from btcfloor.powerlaw import PowerLawModel


@dataclass(frozen=True)
class CycleLow:
    name: str
    date: pd.Timestamp


DEFAULT_CYCLE_LOWS = (
    CycleLow("2011_low", pd.Timestamp("2011-11-18")),
    CycleLow("2015_low", pd.Timestamp("2015-01-14")),
    CycleLow("2018_low", pd.Timestamp("2018-12-15")),
    CycleLow("2022_low", pd.Timestamp("2022-11-21")),
)

FLOOR_RATIO_TOLERANCE = 1e-10


@dataclass(frozen=True)
class WalkForwardFitSpec:
    name: str
    fit_model: Callable[[pd.DataFrame], PowerLawModel]


def _nearest_daily_row(daily: pd.DataFrame, target_date: pd.Timestamp) -> pd.Series:
    if daily.empty:
        raise ValueError("daily price frame is empty")
    distances = (daily["date"] - pd.Timestamp(target_date)).abs()
    return daily.loc[distances.idxmin()]


def _is_below_floor(ratio: float | np.ndarray) -> bool | np.ndarray:
    return ratio < (1.0 - FLOOR_RATIO_TOLERANCE)


def evaluate_cycle_low_windows(
    daily: pd.DataFrame,
    models: Iterable[PowerLawModel],
    lows: Iterable[CycleLow] = DEFAULT_CYCLE_LOWS,
    window_days: int = 90,
) -> pd.DataFrame:
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    rows = []
    daily = daily.sort_values("date").reset_index(drop=True)
    models = list(models)
    for low in lows:
        low_date = pd.Timestamp(low.date)
        window_start = low_date - pd.Timedelta(days=window_days)
        window_end = low_date + pd.Timedelta(days=window_days)
        window = daily.loc[
            daily["date"].between(window_start, window_end),
            ["date", "price_usd"],
        ].copy()
        if window.empty:
            continue

        low_row = _nearest_daily_row(window, low_date)
        for model in models:
            low_floor = float(model.predict_price(pd.Timestamp(low_row["date"]), floor=True)[0])
            low_ratio = float(low_row["price_usd"] / low_floor)

            floor = model.predict_price(window["date"], floor=True)
            ratio = window["price_usd"].to_numpy(dtype=float) / floor
            min_idx = int(np.argmin(ratio))
            min_row = window.iloc[min_idx]

            rows.append(
                {
                    "model": model.name,
                    "cycle_low": low.name,
                    "anchor_low_date": low_date,
                    "observed_low_date": pd.Timestamp(low_row["date"]),
                    "observed_low_price_usd": float(low_row["price_usd"]),
                    "floor_at_observed_low_usd": low_floor,
                    "ratio_at_observed_low": low_ratio,
                    "below_floor_at_observed_low": _is_below_floor(low_ratio),
                    "window_days": window_days,
                    "window_rows": len(window),
                    "min_ratio_date": pd.Timestamp(min_row["date"]),
                    "min_ratio_price_usd": float(min_row["price_usd"]),
                    "min_ratio_floor_usd": float(floor[min_idx]),
                    "min_ratio_to_floor": float(ratio[min_idx]),
                    "days_from_anchor_low_to_min_ratio": int(
                        (pd.Timestamp(min_row["date"]) - low_date).days
                    ),
                    "breach_days_in_window": int(np.sum(_is_below_floor(ratio))),
                }
            )

    return pd.DataFrame(rows)


def _evaluate_model_on_low_window(
    window: pd.DataFrame,
    low: CycleLow,
    model: PowerLawModel,
    window_days: int,
) -> dict[str, object]:
    low_date = pd.Timestamp(low.date)
    low_row = _nearest_daily_row(window, low_date)
    low_floor = float(model.predict_price(pd.Timestamp(low_row["date"]), floor=True)[0])
    low_ratio = float(low_row["price_usd"] / low_floor)

    floor = model.predict_price(window["date"], floor=True)
    ratio = window["price_usd"].to_numpy(dtype=float) / floor
    min_idx = int(np.argmin(ratio))
    min_row = window.iloc[min_idx]

    return {
        "model": model.name,
        "cycle_low": low.name,
        "anchor_low_date": low_date,
        "observed_low_date": pd.Timestamp(low_row["date"]),
        "observed_low_price_usd": float(low_row["price_usd"]),
        "floor_at_observed_low_usd": low_floor,
        "ratio_at_observed_low": low_ratio,
        "below_floor_at_observed_low": _is_below_floor(low_ratio),
        "window_days": window_days,
        "window_rows": len(window),
        "min_ratio_date": pd.Timestamp(min_row["date"]),
        "min_ratio_price_usd": float(min_row["price_usd"]),
        "min_ratio_floor_usd": float(floor[min_idx]),
        "min_ratio_to_floor": float(ratio[min_idx]),
        "days_from_anchor_low_to_min_ratio": int(
            (pd.Timestamp(min_row["date"]) - low_date).days
        ),
        "breach_days_in_window": int(np.sum(_is_below_floor(ratio))),
    }


def evaluate_walk_forward_cycle_lows(
    daily: pd.DataFrame,
    fit_specs: Iterable[WalkForwardFitSpec],
    lows: Iterable[CycleLow] = DEFAULT_CYCLE_LOWS,
    window_days: int = 90,
    min_train_rows: int = 365,
) -> pd.DataFrame:
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if min_train_rows < 3:
        raise ValueError("min_train_rows must be at least 3")

    rows = []
    daily = daily.sort_values("date").reset_index(drop=True)
    for low in lows:
        low_date = pd.Timestamp(low.date)
        window_start = low_date - pd.Timedelta(days=window_days)
        window_end = low_date + pd.Timedelta(days=window_days)
        train_end = window_start - pd.Timedelta(days=1)
        train = daily.loc[daily["date"] <= train_end].copy()
        window = daily.loc[
            daily["date"].between(window_start, window_end),
            ["date", "price_usd"],
        ].copy()
        if window.empty or len(train) < min_train_rows:
            continue

        for spec in fit_specs:
            model = spec.fit_model(train)
            row = _evaluate_model_on_low_window(
                window=window,
                low=low,
                model=model,
                window_days=window_days,
            )
            row.update(
                {
                    "fit_spec": spec.name,
                    "train_start_date": train["date"].min(),
                    "train_end_date": train["date"].max(),
                    "train_rows": len(train),
                    "model_fit_rows": model.n_obs,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_cycle_low_validation(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "cycles_evaluated",
                "median_ratio_at_observed_low",
                "median_abs_log_error_at_low",
                "worst_abs_log_error_at_low",
                "cycles_below_floor_at_low",
                "cycles_with_window_breach",
                "total_breach_days_in_windows",
                "mean_abs_days_from_low_to_min_ratio",
                "floor_quality_score",
            ]
        )

    rows = []
    for model, group in validation.groupby("model", sort=False):
        abs_log_error = np.abs(np.log(group["ratio_at_observed_low"].to_numpy(dtype=float)))
        cycles_below = int(group["below_floor_at_observed_low"].sum())
        cycles_with_breach = int(group["breach_days_in_window"].gt(0).sum())
        total_breach_days = int(group["breach_days_in_window"].sum())
        mean_abs_timing = float(
            group["days_from_anchor_low_to_min_ratio"].abs().mean()
        )

        closeness_component = max(
            0.0,
            100.0 * (1.0 - float(np.median(abs_log_error)) / np.log(2.0)),
        )
        breach_penalty = min(40.0, total_breach_days / max(1, len(group)) * 2.0)
        timing_penalty = min(20.0, mean_abs_timing / 90.0 * 20.0)
        score = max(0.0, closeness_component - breach_penalty - timing_penalty)

        rows.append(
            {
                "model": model,
                "cycles_evaluated": len(group),
                "median_ratio_at_observed_low": float(
                    group["ratio_at_observed_low"].median()
                ),
                "median_abs_log_error_at_low": float(np.median(abs_log_error)),
                "worst_abs_log_error_at_low": float(np.max(abs_log_error)),
                "cycles_below_floor_at_low": cycles_below,
                "cycles_with_window_breach": cycles_with_breach,
                "total_breach_days_in_windows": total_breach_days,
                "mean_abs_days_from_low_to_min_ratio": mean_abs_timing,
                "floor_quality_score": score,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["floor_quality_score", "median_abs_log_error_at_low"],
        ascending=[False, True],
    )
