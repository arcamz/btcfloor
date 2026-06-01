from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
import pandas as pd

from btcfloor.powerlaw import PowerLawModel


DEFAULT_HORIZON_MONTHS = (1, 3, 6, 9, 12)


def floor_distance_by_horizon(
    model: PowerLawModel,
    as_of_date: pd.Timestamp,
    spot_price_usd: float,
    horizons_months: Iterable[int] = DEFAULT_HORIZON_MONTHS,
    holdings_btc: float = 1.0,
) -> pd.DataFrame:
    rows = []
    as_of = pd.Timestamp(as_of_date)
    for months in horizons_months:
        target = as_of + pd.DateOffset(months=int(months))
        floor = float(model.predict_price(target, floor=True)[0])
        pct_above_floor = spot_price_usd / floor - 1.0
        downside_to_floor_pct = max(0.0, 1.0 - floor / spot_price_usd)
        rows.append(
            {
                "model": model.name,
                "horizon_months": int(months),
                "target_date": target.normalize(),
                "current_price_usd": spot_price_usd,
                "future_floor_usd": floor,
                "pct_above_future_floor": pct_above_floor,
                "downside_to_future_floor_pct": downside_to_floor_pct,
                "usd_at_risk_for_holdings": max(
                    0.0, (spot_price_usd - floor) * holdings_btc
                ),
                "below_future_floor": spot_price_usd < floor,
            }
        )
    return pd.DataFrame(rows)


def floor_proximity_score(
    spot_price_usd: float,
    floor_usd: float,
    neutral_multiple: float = 2.0,
) -> float:
    if spot_price_usd <= 0.0:
        raise ValueError("spot_price_usd must be positive")
    if floor_usd <= 0.0:
        raise ValueError("floor_usd must be positive")
    if neutral_multiple <= 1.0:
        raise ValueError("neutral_multiple must be greater than 1")

    ratio = spot_price_usd / floor_usd
    if ratio <= 1.0:
        return 100.0

    log_room = math.log(ratio) / math.log(neutral_multiple)
    return float(max(0.0, min(100.0, (1.0 - log_room) * 100.0)))


def cycle_time_pressure_score(
    target_date: pd.Timestamp,
    expected_low_date: pd.Timestamp,
    full_window_days: int = 365,
) -> float:
    if full_window_days <= 0:
        raise ValueError("full_window_days must be positive")
    days_from_low = abs((pd.Timestamp(expected_low_date) - pd.Timestamp(target_date)).days)
    return float(max(0.0, min(100.0, (1.0 - days_from_low / full_window_days) * 100.0)))


def ensemble_floor_risk_by_horizon(
    models: Iterable[PowerLawModel],
    as_of_date: pd.Timestamp,
    spot_price_usd: float,
    expected_low_date: pd.Timestamp | None,
    horizons_months: Iterable[int] = DEFAULT_HORIZON_MONTHS,
    holdings_btc: float = 1.0,
    neutral_multiple: float = 2.0,
    time_weight: float = 0.30,
    time_window_days: int = 365,
) -> pd.DataFrame:
    models = list(models)
    if not models:
        raise ValueError("At least one model is required")
    if not 0.0 <= time_weight <= 1.0:
        raise ValueError("time_weight must be between 0 and 1")

    rows = []
    as_of = pd.Timestamp(as_of_date)
    for months in horizons_months:
        target = as_of + pd.DateOffset(months=int(months))
        floors = np.array(
            [float(model.predict_price(target, floor=True)[0]) for model in models],
            dtype=float,
        )
        floor_min = float(np.min(floors))
        floor_median = float(np.median(floors))
        floor_max = float(np.max(floors))
        pct_above_median_floor = spot_price_usd / floor_median - 1.0
        downside_to_median_floor_pct = max(0.0, 1.0 - floor_median / spot_price_usd)
        proximity_score = floor_proximity_score(
            spot_price_usd,
            floor_median,
            neutral_multiple=neutral_multiple,
        )

        if expected_low_date is None:
            days_to_expected_low = None
            time_score = 0.0
            bottom_pressure_score = proximity_score
        else:
            days_to_expected_low = int((pd.Timestamp(expected_low_date) - target).days)
            time_score = cycle_time_pressure_score(
                target,
                pd.Timestamp(expected_low_date),
                full_window_days=time_window_days,
            )
            bottom_pressure_score = (
                (1.0 - time_weight) * proximity_score + time_weight * time_score
            )

        rows.append(
            {
                "horizon_months": int(months),
                "target_date": target.normalize(),
                "current_price_usd": spot_price_usd,
                "floor_min_usd": floor_min,
                "floor_median_usd": floor_median,
                "floor_max_usd": floor_max,
                "model_spread_pct_of_median": (floor_max - floor_min) / floor_median,
                "pct_above_median_floor": pct_above_median_floor,
                "downside_to_median_floor_pct": downside_to_median_floor_pct,
                "usd_at_risk_to_median_floor": max(
                    0.0, (spot_price_usd - floor_median) * holdings_btc
                ),
                "below_median_future_floor": spot_price_usd < floor_median,
                "days_to_expected_cycle_low_at_target": days_to_expected_low,
                "floor_proximity_score": proximity_score,
                "cycle_time_pressure_score": time_score,
                "bottom_pressure_score": float(bottom_pressure_score),
            }
        )
    return pd.DataFrame(rows)


def role_based_floor_risk_by_horizon(
    hard_floor_model: PowerLawModel,
    warning_floor_model: PowerLawModel,
    as_of_date: pd.Timestamp,
    spot_price_usd: float,
    expected_low_date: pd.Timestamp | None,
    horizons_months: Iterable[int] = DEFAULT_HORIZON_MONTHS,
    holdings_btc: float = 1.0,
    neutral_multiple: float = 2.0,
    hard_floor_weight: float = 0.45,
    warning_floor_weight: float = 0.35,
    time_weight: float = 0.20,
    time_window_days: int = 365,
) -> pd.DataFrame:
    if spot_price_usd <= 0.0:
        raise ValueError("spot_price_usd must be positive")
    if holdings_btc < 0.0:
        raise ValueError("holdings_btc must be non-negative")
    weights = (hard_floor_weight, warning_floor_weight, time_weight)
    if any(weight < 0.0 for weight in weights):
        raise ValueError("risk weights must be non-negative")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("risk weights must sum to 1")

    rows = []
    as_of = pd.Timestamp(as_of_date)
    for months in horizons_months:
        target = as_of + pd.DateOffset(months=int(months))
        hard_floor = float(hard_floor_model.predict_price(target, floor=True)[0])
        warning_floor = float(warning_floor_model.predict_price(target, floor=True)[0])

        hard_gap = spot_price_usd - hard_floor
        warning_gap = spot_price_usd - warning_floor
        pct_above_hard = spot_price_usd / hard_floor - 1.0
        pct_above_warning = spot_price_usd / warning_floor - 1.0
        hard_downside_pct = max(0.0, 1.0 - hard_floor / spot_price_usd)
        warning_downside_pct = max(0.0, 1.0 - warning_floor / spot_price_usd)
        hard_score = floor_proximity_score(
            spot_price_usd,
            hard_floor,
            neutral_multiple=neutral_multiple,
        )
        warning_score = floor_proximity_score(
            spot_price_usd,
            warning_floor,
            neutral_multiple=neutral_multiple,
        )

        if expected_low_date is None:
            days_to_expected_low = None
            time_score = 0.0
        else:
            expected_low = pd.Timestamp(expected_low_date)
            days_to_expected_low = int((expected_low - target).days)
            time_score = cycle_time_pressure_score(
                target,
                expected_low,
                full_window_days=time_window_days,
            )

        bottom_pressure_score = float(
            hard_floor_weight * hard_score
            + warning_floor_weight * warning_score
            + time_weight * time_score
        )
        if spot_price_usd < hard_floor:
            risk_state = "below_fixed_floor"
        elif spot_price_usd < warning_floor:
            risk_state = "below_adaptive_warning"
        elif bottom_pressure_score >= 80.0:
            risk_state = "high_bottom_pressure"
        elif bottom_pressure_score >= 60.0:
            risk_state = "elevated_bottom_pressure"
        else:
            risk_state = "normal"

        rows.append(
            {
                "horizon_months": int(months),
                "target_date": target.normalize(),
                "current_price_usd": spot_price_usd,
                "hard_floor_model": hard_floor_model.name,
                "warning_floor_model": warning_floor_model.name,
                "hard_floor_usd": hard_floor,
                "warning_floor_usd": warning_floor,
                "pct_above_hard_floor": pct_above_hard,
                "pct_above_warning_floor": pct_above_warning,
                "hard_floor_gap_usd": hard_gap,
                "warning_floor_gap_usd": warning_gap,
                "downside_to_hard_floor_pct": hard_downside_pct,
                "downside_to_warning_floor_pct": warning_downside_pct,
                "usd_at_risk_to_hard_floor": max(0.0, hard_gap * holdings_btc),
                "usd_at_risk_to_warning_floor": max(0.0, warning_gap * holdings_btc),
                "below_hard_future_floor": spot_price_usd < hard_floor,
                "below_warning_future_floor": spot_price_usd < warning_floor,
                "days_to_expected_cycle_low_at_target": days_to_expected_low,
                "hard_floor_proximity_score": hard_score,
                "warning_floor_proximity_score": warning_score,
                "cycle_time_pressure_score": time_score,
                "bottom_pressure_score": bottom_pressure_score,
                "risk_state": risk_state,
            }
        )
    return pd.DataFrame(rows)


def composite_floor_estimate_by_horizon(
    hard_floor_model: PowerLawModel,
    warning_floor_model: PowerLawModel,
    as_of_date: pd.Timestamp,
    spot_price_usd: float,
    expected_low_date: pd.Timestamp | None,
    horizons_months: Iterable[int] = DEFAULT_HORIZON_MONTHS,
    holdings_btc: float = 1.0,
    blend_min: float = 0.20,
    blend_max: float = 0.80,
    time_window_days: int = 365,
) -> pd.DataFrame:
    if spot_price_usd <= 0.0:
        raise ValueError("spot_price_usd must be positive")
    if holdings_btc < 0.0:
        raise ValueError("holdings_btc must be non-negative")
    if not 0.0 <= blend_min <= 1.0:
        raise ValueError("blend_min must be between 0 and 1")
    if not 0.0 <= blend_max <= 1.0:
        raise ValueError("blend_max must be between 0 and 1")
    if blend_min > blend_max:
        raise ValueError("blend_min must be less than or equal to blend_max")

    rows = []
    as_of = pd.Timestamp(as_of_date)
    for months in horizons_months:
        target = as_of + pd.DateOffset(months=int(months))
        hard_floor = float(hard_floor_model.predict_price(target, floor=True)[0])
        warning_floor = float(warning_floor_model.predict_price(target, floor=True)[0])
        lower_floor = min(hard_floor, warning_floor)
        upper_floor = max(hard_floor, warning_floor)

        if expected_low_date is None:
            days_to_expected_low = None
            time_score = 0.0
        else:
            expected_low = pd.Timestamp(expected_low_date)
            days_to_expected_low = int((expected_low - target).days)
            time_score = cycle_time_pressure_score(
                target,
                expected_low,
                full_window_days=time_window_days,
            )

        blend_weight = blend_min + (blend_max - blend_min) * (time_score / 100.0)
        composite_floor = lower_floor * (1.0 - blend_weight) + upper_floor * blend_weight
        pct_above_composite = spot_price_usd / composite_floor - 1.0
        downside_to_composite_pct = max(0.0, 1.0 - composite_floor / spot_price_usd)
        usd_at_risk_to_composite = max(
            0.0, (spot_price_usd - composite_floor) * holdings_btc
        )

        if spot_price_usd < lower_floor:
            composite_state = "below_floor_band"
        elif spot_price_usd < composite_floor:
            composite_state = "below_composite_floor"
        elif spot_price_usd < upper_floor:
            composite_state = "inside_floor_band"
        else:
            composite_state = "above_floor_band"

        rows.append(
            {
                "horizon_months": int(months),
                "target_date": target.normalize(),
                "current_price_usd": spot_price_usd,
                "hard_floor_model": hard_floor_model.name,
                "warning_floor_model": warning_floor_model.name,
                "hard_floor_usd": hard_floor,
                "warning_floor_usd": warning_floor,
                "lower_floor_usd": lower_floor,
                "upper_floor_usd": upper_floor,
                "time_pressure_score": time_score,
                "blend_weight": blend_weight,
                "composite_floor_usd": composite_floor,
                "pct_above_composite_floor": pct_above_composite,
                "downside_to_composite_floor_pct": downside_to_composite_pct,
                "usd_at_risk_to_composite_floor": usd_at_risk_to_composite,
                "below_composite_floor": spot_price_usd < composite_floor,
                "days_to_expected_cycle_low_at_target": days_to_expected_low,
                "composite_state": composite_state,
            }
        )
    return pd.DataFrame(rows)
