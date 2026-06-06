from __future__ import annotations

import numpy as np
import pandas as pd

from btcfloor.analysis import _build_current_bottom_summary, _build_model_evidence_summary
from btcfloor.data import (
    GENESIS_DATE,
    append_missing_recent_market_prices,
    apply_price_fixes,
    normalize_coingecko_price_points,
    to_weekly_close,
    to_weekly_ohlc,
)
from btcfloor.expectile import (
    evaluate_bottom_expectile_sensitivity,
    expectile_model_name,
    fit_expectile_power_law,
)
from btcfloor.forward_floor import (
    floor_target_crossing_date,
    floor_threshold_signal_dates,
    future_floor_overlap_daily,
    group_future_floor_overlap_episodes,
)
from btcfloor.powerlaw import (
    BURGER_2019_END_DATE,
    GIOVANNI_FLOOR_MULTIPLIER,
    GIOVANNI_POWER_LAW_EXPONENT,
    GIOVANNI_POWER_LAW_MULTIPLIER,
    fit_burger_2019_ransac_floor,
    fit_power_law_ols,
    giovanni_power_law_floor_model,
)
from btcfloor.risk import (
    cycle_time_pressure_score,
    ensemble_floor_risk_by_horizon,
    floor_distance_by_horizon,
    floor_proximity_score,
    role_based_floor_risk_by_horizon,
)
from btcfloor.validation import (
    CycleLow,
    evaluate_cycle_low_windows,
    evaluate_walk_forward_cycle_lows,
    summarize_cycle_low_validation,
    WalkForwardFitSpec,
)


def synthetic_power_law_frame(
    intercept: float = -3.0,
    slope: float = 2.0,
    start_day: int = 500,
    n: int = 200,
) -> pd.DataFrame:
    days = np.arange(start_day, start_day + n)
    dates = GENESIS_DATE + pd.to_timedelta(days, unit="D")
    price = np.power(10.0, intercept + slope * np.log10(days))
    return pd.DataFrame(
        {
            "date": dates,
            "days_since_genesis": days,
            "price_usd": price,
            "source": "synthetic",
        }
    )


def test_ols_power_law_recovers_exact_synthetic_series() -> None:
    df = synthetic_power_law_frame(intercept=-4.25, slope=2.75)

    model = fit_power_law_ols(df)

    assert model.intercept == pytest_approx(-4.25)
    assert model.slope == pytest_approx(2.75)
    assert model.floor_offset_log10 < 1e-10


def test_expectile_power_law_recovers_exact_synthetic_series() -> None:
    df = synthetic_power_law_frame(intercept=-2.0, slope=1.5)

    model = fit_expectile_power_law(df, tau=0.0001)

    assert model.intercept == pytest_approx(-2.0)
    assert model.slope == pytest_approx(1.5)


def test_expectile_model_name_is_stable_for_report_keys() -> None:
    assert expectile_model_name(0.0001) == "weekly_expectile_power_law_tau_0_0001"
    assert expectile_model_name(0.01) == "weekly_expectile_power_law_tau_0_01"


def test_giovanni_floor_uses_specified_formula_and_multiplier() -> None:
    model = giovanni_power_law_floor_model()
    date = GENESIS_DATE + pd.Timedelta(days=6400)
    expected_trend = GIOVANNI_POWER_LAW_MULTIPLIER * 6400 ** GIOVANNI_POWER_LAW_EXPONENT

    trend = model.predict_price(date, floor=False)[0]
    floor = model.predict_price(date, floor=True)[0]

    assert trend == pytest_approx(expected_trend)
    assert floor / trend == pytest_approx(GIOVANNI_FLOOR_MULTIPLIER)


def test_burger_2019_ransac_floor_ignores_post_vintage_rows() -> None:
    df = synthetic_power_law_frame(intercept=-4.25, slope=2.75, start_day=560, n=3600)
    vintage_model = fit_burger_2019_ransac_floor(df)

    future = df.copy()
    future.loc[len(future)] = {
        "date": pd.Timestamp("2026-06-03"),
        "days_since_genesis": (pd.Timestamp("2026-06-03") - GENESIS_DATE).days,
        "price_usd": 1e12,
        "source": "synthetic",
    }
    future_model = fit_burger_2019_ransac_floor(future)

    assert vintage_model.fitted_to <= BURGER_2019_END_DATE
    assert vintage_model.n_obs == future_model.n_obs
    assert future_model.intercept == pytest_approx(vintage_model.intercept)
    assert future_model.slope == pytest_approx(vintage_model.slope)
    assert future_model.floor_offset_log10 == pytest_approx(
        vintage_model.floor_offset_log10
    )


def test_giovanni_floor_crosses_60k_in_july_2026() -> None:
    model = giovanni_power_law_floor_model()

    crossing = floor_target_crossing_date(model, 60_000.0)

    assert crossing == pd.Timestamp("2026-07-08")


def test_forward_threshold_signal_dates_shift_by_horizon() -> None:
    model = giovanni_power_law_floor_model()

    thresholds = floor_threshold_signal_dates(model, 60_000.0, horizons_months=(0, 6, 12))

    assert list(thresholds["horizon_months"]) == [0, 6, 12]
    assert thresholds.loc[0, "as_of_date"] == pd.Timestamp("2026-07-08")
    assert thresholds.loc[1, "as_of_date"] == pd.Timestamp("2026-01-08")
    assert thresholds.loc[2, "as_of_date"] == pd.Timestamp("2025-07-08")


def test_future_floor_overlap_groups_contiguous_breach_windows() -> None:
    model = fit_power_law_ols(
        synthetic_power_law_frame(intercept=2.0, slope=0.0, start_day=500, n=20),
        name="constant_floor",
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05"]
            ),
            "price_usd": [120.0, 80.0, 70.0, 90.0],
        }
    )

    overlap = future_floor_overlap_daily(daily, model, horizon_months=12)
    episodes = group_future_floor_overlap_episodes(overlap, max_gap_days=1)

    assert len(episodes) == 2
    assert episodes.loc[0, "start_date"] == pd.Timestamp("2026-01-02")
    assert episodes.loc[0, "end_date"] == pd.Timestamp("2026-01-03")
    assert episodes.loc[0, "breach_days"] == 2
    assert episodes.loc[0, "min_ratio_to_future_floor"] == pytest_approx(0.7)
    assert episodes.loc[1, "start_date"] == pd.Timestamp("2026-01-05")


def test_floor_distance_by_horizon_flags_future_floor_breach() -> None:
    model = giovanni_power_law_floor_model()
    as_of = pd.Timestamp("2026-05-23")
    low_spot = 1.0

    risk = floor_distance_by_horizon(model, as_of, low_spot, horizons_months=(12,))

    assert bool(risk.loc[0, "below_future_floor"])
    assert risk.loc[0, "usd_at_risk_for_holdings"] == 0.0


def test_floor_proximity_score_runs_from_floor_to_neutral_multiple() -> None:
    assert floor_proximity_score(100.0, 100.0) == pytest_approx(100.0)
    assert floor_proximity_score(150.0, 100.0) == pytest_approx(41.5037499278844)
    assert floor_proximity_score(200.0, 100.0) == pytest_approx(0.0)
    assert floor_proximity_score(300.0, 100.0) == pytest_approx(0.0)


def test_cycle_time_pressure_score_is_high_near_expected_low() -> None:
    low = pd.Timestamp("2026-10-19")

    assert cycle_time_pressure_score(low, low) == pytest_approx(100.0)
    assert cycle_time_pressure_score(low + pd.Timedelta(days=365), low) == pytest_approx(
        0.0
    )


def test_ensemble_floor_risk_uses_median_floor_and_time_pressure() -> None:
    df = synthetic_power_law_frame(intercept=-4.25, slope=2.75)
    ols = fit_power_law_ols(df)
    giovanni = giovanni_power_law_floor_model()
    expectile = fit_expectile_power_law(df, tau=0.0001)
    as_of = pd.Timestamp("2026-05-23")

    risk = ensemble_floor_risk_by_horizon(
        [ols, giovanni, expectile],
        as_of_date=as_of,
        spot_price_usd=100_000.0,
        expected_low_date=as_of + pd.DateOffset(months=6),
        horizons_months=(6,),
    )

    floors = sorted(
        [
            float(model.predict_price(as_of + pd.DateOffset(months=6), floor=True)[0])
            for model in (ols, giovanni, expectile)
        ]
    )
    assert risk.loc[0, "floor_median_usd"] == pytest_approx(floors[1])
    assert risk.loc[0, "cycle_time_pressure_score"] == pytest_approx(100.0)
    assert 0.0 <= risk.loc[0, "bottom_pressure_score"] <= 100.0


def test_role_based_floor_risk_uses_hard_warning_and_time_scores() -> None:
    hard = fit_power_law_ols(
        synthetic_power_law_frame(intercept=0.0, slope=0.0, start_day=500, n=20),
        name="hard_floor",
    )
    warning = fit_power_law_ols(
        synthetic_power_law_frame(intercept=np.log10(1.5), slope=0.0, start_day=500, n=20),
        name="warning_floor",
    )
    as_of = pd.Timestamp("2026-05-23")

    risk = role_based_floor_risk_by_horizon(
        hard,
        warning,
        as_of_date=as_of,
        spot_price_usd=1.25,
        expected_low_date=as_of + pd.DateOffset(months=1),
        horizons_months=(1,),
        hard_floor_weight=0.4,
        warning_floor_weight=0.4,
        time_weight=0.2,
    )

    assert risk.loc[0, "hard_floor_usd"] == pytest_approx(1.0)
    assert risk.loc[0, "warning_floor_usd"] == pytest_approx(1.5)
    assert risk.loc[0, "usd_at_risk_to_hard_floor"] == pytest_approx(0.25)
    assert risk.loc[0, "usd_at_risk_to_warning_floor"] == pytest_approx(0.0)
    assert bool(risk.loc[0, "below_warning_future_floor"])
    assert risk.loc[0, "cycle_time_pressure_score"] == pytest_approx(100.0)
    assert risk.loc[0, "risk_state"] == "below_adaptive_warning"


def test_role_based_floor_risk_requires_weights_to_sum_to_one() -> None:
    hard = giovanni_power_law_floor_model()
    warning = giovanni_power_law_floor_model()

    import pytest

    with pytest.raises(ValueError, match="weights must sum"):
        role_based_floor_risk_by_horizon(
            hard,
            warning,
            as_of_date=pd.Timestamp("2026-05-23"),
            spot_price_usd=100_000.0,
            expected_low_date=None,
            hard_floor_weight=0.5,
            warning_floor_weight=0.5,
            time_weight=0.5,
        )


def test_bottom_expectile_sensitivity_contains_requested_bottom_tau() -> None:
    df = synthetic_power_law_frame(intercept=-4.25, slope=2.75, n=500)
    weekly = to_weekly_close(df)

    sensitivity = evaluate_bottom_expectile_sensitivity(
        weekly,
        df,
        as_of_date=df["date"].iloc[-1],
        spot_price_usd=float(df["price_usd"].iloc[-1]),
        taus=(0.0001, 0.01),
    )

    assert list(sensitivity["tau"]) == [0.0001, 0.01]
    assert sensitivity.loc[0, "tau_percent"] == pytest_approx(0.01)
    assert sensitivity.loc[0, "model"] == "weekly_expectile_power_law_tau_0_0001"


def test_cycle_low_validation_records_low_window_ratios() -> None:
    df = synthetic_power_law_frame(intercept=-3.0, slope=2.0, start_day=500, n=100)
    model = fit_power_law_ols(df)
    low = CycleLow("synthetic_low", df["date"].iloc[50])

    validation = evaluate_cycle_low_windows(df, [model], lows=(low,), window_days=10)

    assert len(validation) == 1
    assert validation.loc[0, "cycle_low"] == "synthetic_low"
    assert validation.loc[0, "ratio_at_observed_low"] == pytest_approx(1.0)
    assert validation.loc[0, "min_ratio_to_floor"] == pytest_approx(1.0)
    assert validation.loc[0, "breach_days_in_window"] == 0


def test_cycle_low_validation_summary_scores_breachy_model_lower() -> None:
    validation = pd.DataFrame(
        {
            "model": ["good", "good", "bad", "bad"],
            "ratio_at_observed_low": [1.0, 1.05, 0.8, 3.0],
            "below_floor_at_observed_low": [False, False, True, False],
            "breach_days_in_window": [0, 0, 10, 0],
            "days_from_anchor_low_to_min_ratio": [0, 5, 40, 80],
        }
    )

    summary = summarize_cycle_low_validation(validation)

    assert summary.iloc[0]["model"] == "good"
    assert summary.iloc[0]["floor_quality_score"] > summary.iloc[-1]["floor_quality_score"]


def test_walk_forward_cycle_low_validation_trains_before_window() -> None:
    df = synthetic_power_law_frame(intercept=-3.0, slope=2.0, start_day=500, n=700)
    low = CycleLow("synthetic_low", df["date"].iloc[500])
    spec = WalkForwardFitSpec("ols", lambda frame: fit_power_law_ols(frame))

    validation = evaluate_walk_forward_cycle_lows(
        df,
        fit_specs=(spec,),
        lows=(low,),
        window_days=30,
        min_train_rows=365,
    )

    expected_train_end = low.date - pd.Timedelta(days=31)
    assert len(validation) == 1
    assert validation.loc[0, "train_end_date"] == expected_train_end
    assert validation.loc[0, "observed_low_date"] == low.date
    assert validation.loc[0, "ratio_at_observed_low"] == pytest_approx(1.0)


def test_walk_forward_cycle_low_validation_skips_insufficient_training() -> None:
    df = synthetic_power_law_frame(intercept=-3.0, slope=2.0, start_day=500, n=100)
    low = CycleLow("synthetic_low", df["date"].iloc[50])
    spec = WalkForwardFitSpec("ols", lambda frame: fit_power_law_ols(frame))

    validation = evaluate_walk_forward_cycle_lows(
        df,
        fit_specs=(spec,),
        lows=(low,),
        window_days=10,
        min_train_rows=365,
    )

    assert validation.empty


def test_walk_forward_cycle_low_validation_accepts_fixed_floor_model() -> None:
    df = synthetic_power_law_frame(intercept=-3.0, slope=2.0, start_day=500, n=700)
    low = CycleLow("synthetic_low", df["date"].iloc[500])
    spec = WalkForwardFitSpec(
        "giovanni_power_law_floor",
        lambda frame: giovanni_power_law_floor_model(),
    )

    validation = evaluate_walk_forward_cycle_lows(
        df,
        fit_specs=(spec,),
        lows=(low,),
        window_days=30,
        min_train_rows=365,
    )

    assert len(validation) == 1
    assert validation.loc[0, "model"] == "giovanni_power_law_floor"
    assert validation.loc[0, "fit_spec"] == "giovanni_power_law_floor"
    assert validation.loc[0, "model_fit_rows"] == 0
    assert validation.loc[0, "train_end_date"] == low.date - pd.Timedelta(days=31)


def test_model_evidence_summary_distinguishes_fixed_and_adaptive_roles() -> None:
    snapshot = pd.DataFrame(
        {
            "model": [
                "giovanni_power_law_floor",
                "weekly_expectile_power_law_tau_0_0001",
            ],
            "floor_usd": [10.0, 12.0],
            "pct_above_floor": [0.2, 0.1],
        }
    )
    full = pd.DataFrame(
        {
            "model": snapshot["model"],
            "median_ratio_at_observed_low": [1.1, 1.0],
            "total_breach_days_in_windows": [0, 0],
        }
    )
    walk = pd.DataFrame(
        {
            "model": snapshot["model"],
            "median_ratio_at_observed_low": [1.2, 0.8],
            "total_breach_days_in_windows": [0, 30],
            "cycles_below_floor_at_low": [0, 2],
        }
    )
    stability = pd.DataFrame(
        {
            "model": ["weekly_expectile_power_law_tau_0_0001"],
            "exclude_last_cycles": [1],
            "floor_breach_days": [10],
            "latest_ratio_to_floor": [1.1],
        }
    )

    summary = _build_model_evidence_summary(snapshot, full, walk, stability)

    assert summary.iloc[0]["model"] == "giovanni_power_law_floor"
    assert summary.iloc[0]["role"] == "fixed lower rail"
    expectile = summary.loc[
        summary["model"].eq("weekly_expectile_power_law_tau_0_0001")
    ].iloc[0]
    assert expectile["role"] == "adaptive bottom-pressure signal"
    assert "chase recent lows" in expectile["interpretation"]


def test_current_bottom_summary_selects_pressure_and_crossing_horizons() -> None:
    snapshot = pd.DataFrame(
        {
            "model": [
                "giovanni_power_law_floor",
                "weekly_expectile_power_law_tau_0_0001",
            ],
            "as_of_date": [pd.Timestamp("2026-05-23"), pd.Timestamp("2026-05-23")],
            "spot_price_usd": [100.0, 100.0],
            "floor_usd": [70.0, 85.0],
            "pct_above_floor": [100.0 / 70.0 - 1.0, 100.0 / 85.0 - 1.0],
        }
    )
    current_risk = pd.DataFrame(
        {
            "bottom_pressure_score": [55.0],
            "risk_state": ["normal"],
        }
    )
    forward_risk = pd.DataFrame(
        {
            "horizon_months": [1, 3, 6],
            "target_date": pd.to_datetime(["2026-06-23", "2026-08-23", "2026-11-23"]),
            "bottom_pressure_score": [60.0, 90.0, 80.0],
            "risk_state": ["normal", "below_adaptive_warning", "below_fixed_floor"],
            "below_warning_future_floor": [False, True, True],
            "below_hard_future_floor": [False, False, True],
        }
    )
    phase = {
        "expected_next_low_date": pd.Timestamp("2026-10-19"),
        "days_to_expected_next_low": 149,
    }

    summary = _build_current_bottom_summary(snapshot, current_risk, forward_risk, phase)

    row = summary.iloc[0]
    assert row["hard_floor_usd"] == 70.0
    assert row["warning_floor_usd"] == 85.0
    assert row["highest_pressure_horizon_months"] == 3
    assert row["first_warning_floor_cross_horizon_months"] == 3
    assert row["first_hard_floor_cross_horizon_months"] == 6
    assert row["current_risk_state"] == "normal"


def test_weekly_close_uses_actual_last_observed_date_for_partial_week() -> None:
    df = synthetic_power_law_frame(start_day=500, n=9)

    weekly = to_weekly_close(df)

    assert weekly["date"].iloc[-1] == df["date"].iloc[-1]
    assert weekly["days_since_genesis"].iloc[-1] == df["days_since_genesis"].iloc[-1]


def test_weekly_ohlc_uses_actual_dates_and_price_extremes() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]),
            "price_usd": [10.0, 15.0, 8.0],
        }
    )
    df["days_since_genesis"] = (df["date"] - GENESIS_DATE).dt.days
    df["source"] = "synthetic"

    weekly = to_weekly_ohlc(df)

    assert weekly.loc[0, "date"] == pd.Timestamp("2024-01-05")
    assert weekly.loc[0, "open"] == 10.0
    assert weekly.loc[0, "high"] == 15.0
    assert weekly.loc[0, "low"] == 8.0
    assert weekly.loc[0, "close"] == 8.0


def test_apply_price_fixes_replaces_and_drops_rows() -> None:
    df = synthetic_power_law_frame(start_day=500, n=3)
    fixes = pd.DataFrame(
        {
            "date": [df["date"].iloc[0], df["date"].iloc[1]],
            "action": ["replace", "drop"],
            "price_usd": [123.0, np.nan],
            "reason": ["bad print", "bad row"],
        }
    )

    fixed = apply_price_fixes(df, fixes)

    assert len(fixed) == 2
    assert fixed.loc[fixed["date"].eq(df["date"].iloc[0]), "price_usd"].iloc[0] == 123.0
    assert df["date"].iloc[1] not in set(fixed["date"])


def test_normalize_coingecko_price_points_uses_last_tick_per_utc_day() -> None:
    ticks = [
        [pd.Timestamp("2026-06-01T00:05:00Z").timestamp() * 1000, 70_000.0],
        [pd.Timestamp("2026-06-01T23:55:00Z").timestamp() * 1000, 69_000.0],
        [pd.Timestamp("2026-06-02T12:00:00Z").timestamp() * 1000, 68_000.0],
    ]

    daily = normalize_coingecko_price_points(
        ticks,
        start_date=pd.Timestamp("2026-06-01"),
        end_date=pd.Timestamp("2026-06-02"),
    )

    assert list(daily["date"]) == [
        pd.Timestamp("2026-06-01"),
        pd.Timestamp("2026-06-02"),
    ]
    assert list(daily["price_usd"]) == [69_000.0, 68_000.0]
    assert set(daily["source"]) == {"coingecko_market_chart_range"}


def test_append_missing_recent_market_prices_only_adds_after_base_end() -> None:
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-30", "2026-05-31"]),
            "days_since_genesis": [6356, 6357],
            "price_usd": [73_000.0, 72_000.0],
            "source": ["coinmetrics_community_csv", "coinmetrics_community_csv"],
        }
    )
    recent = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-31", "2026-06-01"]),
            "days_since_genesis": [6357, 6358],
            "price_usd": [71_500.0, 70_000.0],
            "source": ["coingecko_market_chart_range", "coingecko_market_chart_range"],
        }
    )

    combined = append_missing_recent_market_prices(base, recent)

    assert list(combined["date"]) == [
        pd.Timestamp("2026-05-30"),
        pd.Timestamp("2026-05-31"),
        pd.Timestamp("2026-06-01"),
    ]
    assert list(combined["price_usd"]) == [73_000.0, 72_000.0, 70_000.0]
    assert combined.iloc[-1]["source"] == "coingecko_market_chart_range"


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=1e-8, rel=1e-8)
