from __future__ import annotations

from pathlib import Path

import pandas as pd

from btcfloor.cycle import current_cycle_phase, cycle_timing_table
from btcfloor.data import (
    download_and_prepare,
    plot_price_diagnostics,
    to_weekly_close,
    validate_price_history,
    write_data_quality_report,
)
from btcfloor.expectile import (
    evaluate_bottom_expectile_sensitivity,
    expectile_model_name,
    fit_expectile_power_law,
)
from btcfloor.forward_floor import (
    add_nearest_cycle_low_context,
    floor_threshold_signal_dates,
    future_floor_overlap_daily,
    group_future_floor_overlap_episodes,
)
from btcfloor.interactive import write_interactive_weekly_floor_chart
from btcfloor.paths import ProjectPaths
from btcfloor.powerlaw import (
    fit_burger_2019_ransac_floor,
    fit_power_law_ols,
    giovanni_power_law_floor_model,
)
from btcfloor.risk import (
    ensemble_floor_risk_by_horizon,
    floor_distance_by_horizon,
    role_based_floor_risk_by_horizon,
)
from btcfloor.stability import collect_floor_breach_details, evaluate_model_stability
from btcfloor.validation import (
    DEFAULT_CYCLE_LOWS,
    evaluate_cycle_low_windows,
    summarize_cycle_low_validation,
)
from btcfloor.validation import evaluate_walk_forward_cycle_lows, WalkForwardFitSpec


INTERACTIVE_EXPECTILE_TAUS = (0.0001, 0.0005, 0.001, 0.005, 0.01)
DECISION_FLOOR_THRESHOLD_USD = 60_000.0
FORWARD_OVERLAP_HORIZON_MONTHS = 12


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def _format_report_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if "date" in col:
            formatted[col] = pd.to_datetime(formatted[col]).dt.strftime("%Y-%m-%d")
    money_cols = [c for c in formatted.columns if c.endswith("_usd") or "floor_usd" in c]
    money_cols.extend([c for c in formatted.columns if c.startswith("usd_at_risk")])
    for col in money_cols:
        formatted[col] = formatted[col].map(_money)
    pct_cols = [c for c in formatted.columns if c.endswith("_pct") or c.startswith("pct_")]
    for col in pct_cols:
        formatted[col] = formatted[col].map(_pct)
    return formatted.to_markdown(index=False)


def _format_ensemble_risk_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    formatted["target_date"] = pd.to_datetime(formatted["target_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    for col in (
        "current_price_usd",
        "floor_min_usd",
        "floor_median_usd",
        "floor_max_usd",
        "usd_at_risk_to_median_floor",
    ):
        formatted[col] = formatted[col].map(_money)
    for col in (
        "model_spread_pct_of_median",
        "pct_above_median_floor",
        "downside_to_median_floor_pct",
    ):
        formatted[col] = formatted[col].map(_pct)
    for col in (
        "floor_proximity_score",
        "cycle_time_pressure_score",
        "bottom_pressure_score",
    ):
        formatted[col] = formatted[col].map(lambda x: f"{x:,.1f}")
    return formatted.to_markdown(index=False)


def _format_role_based_risk_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    formatted["target_date"] = pd.to_datetime(formatted["target_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    for col in (
        "current_price_usd",
        "hard_floor_usd",
        "warning_floor_usd",
        "hard_floor_gap_usd",
        "warning_floor_gap_usd",
        "usd_at_risk_to_hard_floor",
        "usd_at_risk_to_warning_floor",
    ):
        formatted[col] = formatted[col].map(_money)
    for col in (
        "pct_above_hard_floor",
        "pct_above_warning_floor",
        "downside_to_hard_floor_pct",
        "downside_to_warning_floor_pct",
    ):
        formatted[col] = formatted[col].map(_pct)
    for col in (
        "hard_floor_proximity_score",
        "warning_floor_proximity_score",
        "cycle_time_pressure_score",
        "bottom_pressure_score",
    ):
        formatted[col] = formatted[col].map(lambda x: f"{x:,.1f}")
    view_cols = [
        "horizon_months",
        "target_date",
        "current_price_usd",
        "hard_floor_usd",
        "warning_floor_usd",
        "pct_above_hard_floor",
        "pct_above_warning_floor",
        "usd_at_risk_to_hard_floor",
        "usd_at_risk_to_warning_floor",
        "days_to_expected_cycle_low_at_target",
        "hard_floor_proximity_score",
        "warning_floor_proximity_score",
        "cycle_time_pressure_score",
        "bottom_pressure_score",
        "risk_state",
    ]
    return formatted[view_cols].to_markdown(index=False)


def _format_current_bottom_summary_table(df: pd.DataFrame) -> str:
    row = df.iloc[0]

    def optional_date(value: object) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{pd.Timestamp(value):%Y-%m-%d}"

    def optional_month(value: object) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{int(value)}m"

    rows = [
        ("as of date", f"{pd.Timestamp(row['as_of_date']):%Y-%m-%d}"),
        ("spot price", _money(float(row["spot_price_usd"]))),
        ("Giovanni hard floor", _money(float(row["hard_floor_usd"]))),
        ("0.01% expectile warning floor", _money(float(row["warning_floor_usd"]))),
        ("spot above hard floor", _pct(float(row["pct_above_hard_floor"]))),
        ("spot above warning floor", _pct(float(row["pct_above_warning_floor"]))),
        ("1 BTC risk to hard floor", _money(float(row["usd_at_risk_to_hard_floor_1btc"]))),
        (
            "1 BTC risk to warning floor",
            _money(float(row["usd_at_risk_to_warning_floor_1btc"])),
        ),
        ("current bottom pressure", f"{float(row['current_bottom_pressure_score']):,.1f}"),
        ("current risk state", str(row["current_risk_state"])),
        ("expected next low", optional_date(row["expected_next_low_date"])),
        ("days to expected next low", f"{int(row['days_to_expected_next_low']):,}"),
        ("highest pressure horizon", optional_month(row["highest_pressure_horizon_months"])),
        ("highest pressure target", optional_date(row["highest_pressure_target_date"])),
        ("highest pressure score", f"{float(row['highest_pressure_score']):,.1f}"),
        ("highest pressure state", str(row["highest_pressure_state"])),
        (
            "first warning-floor cross",
            f"{optional_month(row['first_warning_floor_cross_horizon_months'])} "
            f"({optional_date(row['first_warning_floor_cross_date'])})",
        ),
        (
            "first hard-floor cross",
            f"{optional_month(row['first_hard_floor_cross_horizon_months'])} "
            f"({optional_date(row['first_hard_floor_cross_date'])})",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"]).to_markdown(index=False)


def _format_cycle_low_validation_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for col in (
        "median_ratio_at_observed_low",
        "median_abs_log_error_at_low",
        "worst_abs_log_error_at_low",
    ):
        formatted[col] = formatted[col].map(lambda x: f"{x:,.3f}")
    formatted["mean_abs_days_from_low_to_min_ratio"] = formatted[
        "mean_abs_days_from_low_to_min_ratio"
    ].map(lambda x: f"{x:,.1f}")
    formatted["floor_quality_score"] = formatted["floor_quality_score"].map(
        lambda x: f"{x:,.1f}"
    )
    return formatted.to_markdown(index=False)


def _format_walk_forward_summary_table(df: pd.DataFrame) -> str:
    return _format_cycle_low_validation_table(df)


def _format_model_evidence_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    formatted["current_floor_usd"] = formatted["current_floor_usd"].map(_money)
    formatted["pct_above_current_floor"] = formatted["pct_above_current_floor"].map(_pct)
    for col in (
        "full_sample_median_ratio_at_low",
        "walk_forward_median_ratio_at_low",
        "stability_exclude_last_1_latest_ratio_to_floor",
        "stability_exclude_last_2_latest_ratio_to_floor",
    ):
        formatted[col] = formatted[col].map(
            lambda x: "n/a" if pd.isna(x) else f"{float(x):,.3f}x"
        )
    int_cols = [
        "full_sample_breach_days",
        "walk_forward_breach_days",
        "stability_exclude_last_1_breach_days",
        "stability_exclude_last_2_breach_days",
    ]
    for col in int_cols:
        formatted[col] = formatted[col].map(
            lambda x: "n/a" if pd.isna(x) else f"{int(x):,}"
        )
    return formatted.to_markdown(index=False)


def _format_expectile_sensitivity_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    formatted["as_of_date"] = pd.to_datetime(formatted["as_of_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    for col in ("spot_price_usd", "floor_usd"):
        formatted[col] = formatted[col].map(_money)
    formatted["pct_above_floor"] = formatted["pct_above_floor"].map(_pct)
    formatted["tau_percent"] = formatted["tau_percent"].map(lambda x: f"{x:g}%")
    for col in (
        "median_ratio_at_observed_low",
        "slope",
        "intercept",
        "floor_quality_score",
    ):
        formatted[col] = formatted[col].map(lambda x: f"{x:,.3f}")
    return formatted.to_markdown(index=False)


def _format_forward_threshold_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for col in ("as_of_date", "target_date"):
        formatted[col] = pd.to_datetime(formatted[col]).dt.strftime("%Y-%m-%d")
    for col in ("target_floor_usd", "floor_usd"):
        formatted[col] = formatted[col].map(_money)
    return formatted[
        ["horizon_months", "as_of_date", "target_date", "target_floor_usd", "floor_usd"]
    ].to_markdown(index=False)


def _format_forward_overlap_episode_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No forward-floor overlap episodes found._"
    formatted = df.copy()
    for col in (
        "start_date",
        "end_date",
        "min_ratio_date",
        "target_date_at_min",
        "nearest_cycle_low_date",
    ):
        formatted[col] = pd.to_datetime(formatted[col]).dt.strftime("%Y-%m-%d")
    for col in ("min_price_usd", "future_floor_at_min_usd"):
        formatted[col] = formatted[col].map(_money)
    formatted["breach_coverage_pct"] = formatted["breach_coverage_pct"].map(_pct)
    for col in ("min_ratio_to_future_floor", "min_ratio_to_current_floor"):
        formatted[col] = formatted[col].map(lambda x: f"{float(x):,.3f}x")
    view_cols = [
        "start_date",
        "end_date",
        "breach_days",
        "calendar_days",
        "min_ratio_date",
        "min_price_usd",
        "future_floor_at_min_usd",
        "min_ratio_to_future_floor",
        "nearest_cycle_low",
        "days_from_nearest_cycle_low",
        "near_cycle_low_window",
    ]
    return formatted[view_cols].to_markdown(index=False)


def _plot_floor_models(
    daily: pd.DataFrame,
    models: list,
    figure_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        daily["date"],
        daily["price_usd"],
        color="black",
        linewidth=1.1,
        alpha=0.8,
        label="BTC PriceUSD",
    )
    for model in models:
        floor = model.predict_price(daily["date"], floor=True)
        ax.plot(daily["date"], floor, linewidth=1.15, label=model.name)

    ax.set_yscale("log")
    ax.set_title("BTC floor model comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD, log scale")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    out = figure_dir / "btc_floor_models_log.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _model_snapshot(models: list, latest: pd.Series) -> pd.DataFrame:
    rows = []
    as_of = pd.Timestamp(latest["date"])
    price = float(latest["price_usd"])
    for model in models:
        trend = float(model.predict_price(as_of, floor=False)[0])
        floor = float(model.predict_price(as_of, floor=True)[0])
        rows.append(
            {
                "model": model.name,
                "method": model.method,
                "as_of_date": as_of,
                "spot_price_usd": price,
                "trend_usd": trend,
                "floor_usd": floor,
                "pct_above_floor": price / floor - 1.0,
                "slope": model.slope,
                "intercept": model.intercept,
                "floor_offset_log10": model.floor_offset_log10,
                "fit_rows": model.n_obs,
            }
        )
    return pd.DataFrame(rows)


def _lookup_model_row(df: pd.DataFrame, model: str, **filters: object) -> pd.Series | None:
    if df.empty or "model" not in df.columns:
        return None
    matched = df["model"].eq(model)
    for column, value in filters.items():
        if column not in df.columns:
            return None
        matched &= df[column].eq(value)
    subset = df.loc[matched]
    if subset.empty:
        return None
    return subset.iloc[0]


def _optional_float(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row or pd.isna(row[column]):
        return float("nan")
    return float(row[column])


def _optional_int(row: pd.Series | None, column: str) -> int | None:
    if row is None or column not in row or pd.isna(row[column]):
        return None
    return int(row[column])


def _model_role(model: str, walk_forward_row: pd.Series | None) -> tuple[int, str, str]:
    breach_days = _optional_int(walk_forward_row, "total_breach_days_in_windows") or 0
    cycles_below = _optional_int(walk_forward_row, "cycles_below_floor_at_low") or 0

    if model == "giovanni_power_law_floor":
        if breach_days == 0 and cycles_below == 0:
            return (
                1,
                "fixed lower rail",
                "Fixed formula; 2022 closeness is not created by refitting.",
            )
        return (
            4,
            "fixed formula under review",
            "Fixed formula still needs review because walk-forward breaches appeared.",
        )
    if model.startswith("burger_2019_ransac"):
        return (
            2,
            "pre-2022 RANSAC support reference",
            "Fit only on the 2010-2019 Burger window; useful provenance check, not a live refit.",
        )
    if "expectile" in model:
        if breach_days > 0 or cycles_below > 0:
            return (
                3,
                "adaptive bottom-pressure signal",
                "Useful for proximity, but walk-forward breaches show it can chase recent lows.",
            )
        return (
            3,
            "adaptive floor candidate",
            "Adaptive fit has no walk-forward low-window breaches in this sample.",
        )
    if "ols" in model:
        return (
            4,
            "conservative stress floor",
            "Historically lower than cycle lows; useful as a deep-stress bound.",
        )
    return (9, "secondary reference", "Needs model-specific interpretation.")


def _build_model_evidence_summary(
    model_snapshot: pd.DataFrame,
    cycle_low_summary: pd.DataFrame,
    walk_forward_low_summary: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for snapshot_row in model_snapshot.itertuples(index=False):
        model = str(snapshot_row.model)
        full = _lookup_model_row(cycle_low_summary, model)
        walk = _lookup_model_row(walk_forward_low_summary, model)
        stability_1 = _lookup_model_row(stability, model, exclude_last_cycles=1)
        stability_2 = _lookup_model_row(stability, model, exclude_last_cycles=2)
        rank, role, note = _model_role(model, walk)

        rows.append(
            {
                "model": model,
                "role": role,
                "current_floor_usd": float(snapshot_row.floor_usd),
                "pct_above_current_floor": float(snapshot_row.pct_above_floor),
                "full_sample_median_ratio_at_low": _optional_float(
                    full, "median_ratio_at_observed_low"
                ),
                "full_sample_breach_days": _optional_int(
                    full, "total_breach_days_in_windows"
                ),
                "walk_forward_median_ratio_at_low": _optional_float(
                    walk, "median_ratio_at_observed_low"
                ),
                "walk_forward_breach_days": _optional_int(
                    walk, "total_breach_days_in_windows"
                ),
                "stability_exclude_last_1_breach_days": _optional_int(
                    stability_1, "floor_breach_days"
                ),
                "stability_exclude_last_1_latest_ratio_to_floor": _optional_float(
                    stability_1, "latest_ratio_to_floor"
                ),
                "stability_exclude_last_2_breach_days": _optional_int(
                    stability_2, "floor_breach_days"
                ),
                "stability_exclude_last_2_latest_ratio_to_floor": _optional_float(
                    stability_2, "latest_ratio_to_floor"
                ),
                "interpretation": note,
                "rank": rank,
            }
        )
    return pd.DataFrame(rows).sort_values(["rank", "model"]).drop(columns=["rank"])


def _first_crossing(
    risk: pd.DataFrame,
    column: str,
) -> tuple[int | None, pd.Timestamp | None]:
    crossing = risk.loc[risk[column].astype(bool)].sort_values("horizon_months")
    if crossing.empty:
        return None, None
    row = crossing.iloc[0]
    return int(row["horizon_months"]), pd.Timestamp(row["target_date"])


def _build_current_bottom_summary(
    model_snapshot: pd.DataFrame,
    current_role_based_risk: pd.DataFrame,
    forward_role_based_risk: pd.DataFrame,
    phase: dict[str, object],
) -> pd.DataFrame:
    hard_snapshot = _lookup_model_row(model_snapshot, "giovanni_power_law_floor")
    warning_snapshot = _lookup_model_row(
        model_snapshot,
        "weekly_expectile_power_law_tau_0_0001",
    )
    if hard_snapshot is None:
        raise ValueError("model_snapshot is missing giovanni_power_law_floor")
    if warning_snapshot is None:
        raise ValueError("model_snapshot is missing 0.01% expectile floor")
    if current_role_based_risk.empty:
        raise ValueError("current_role_based_risk is empty")
    if forward_role_based_risk.empty:
        raise ValueError("forward_role_based_risk is empty")

    current = current_role_based_risk.iloc[0]
    highest = forward_role_based_risk.sort_values(
        ["bottom_pressure_score", "horizon_months"],
        ascending=[False, True],
    ).iloc[0]
    warning_cross_months, warning_cross_date = _first_crossing(
        forward_role_based_risk,
        "below_warning_future_floor",
    )
    hard_cross_months, hard_cross_date = _first_crossing(
        forward_role_based_risk,
        "below_hard_future_floor",
    )

    return pd.DataFrame(
        [
            {
                "as_of_date": pd.Timestamp(hard_snapshot["as_of_date"]),
                "spot_price_usd": float(hard_snapshot["spot_price_usd"]),
                "hard_floor_model": str(hard_snapshot["model"]),
                "warning_floor_model": str(warning_snapshot["model"]),
                "hard_floor_usd": float(hard_snapshot["floor_usd"]),
                "warning_floor_usd": float(warning_snapshot["floor_usd"]),
                "pct_above_hard_floor": float(hard_snapshot["pct_above_floor"]),
                "pct_above_warning_floor": float(warning_snapshot["pct_above_floor"]),
                "usd_at_risk_to_hard_floor_1btc": max(
                    0.0,
                    float(hard_snapshot["spot_price_usd"])
                    - float(hard_snapshot["floor_usd"]),
                ),
                "usd_at_risk_to_warning_floor_1btc": max(
                    0.0,
                    float(warning_snapshot["spot_price_usd"])
                    - float(warning_snapshot["floor_usd"]),
                ),
                "current_bottom_pressure_score": float(
                    current["bottom_pressure_score"]
                ),
                "current_risk_state": str(current["risk_state"]),
                "expected_next_low_date": pd.Timestamp(phase["expected_next_low_date"]),
                "days_to_expected_next_low": int(phase["days_to_expected_next_low"]),
                "highest_pressure_horizon_months": int(highest["horizon_months"]),
                "highest_pressure_target_date": pd.Timestamp(highest["target_date"]),
                "highest_pressure_score": float(highest["bottom_pressure_score"]),
                "highest_pressure_state": str(highest["risk_state"]),
                "first_warning_floor_cross_horizon_months": warning_cross_months,
                "first_warning_floor_cross_date": warning_cross_date,
                "first_hard_floor_cross_horizon_months": hard_cross_months,
                "first_hard_floor_cross_date": hard_cross_date,
            }
        ]
    )


def _write_initial_report(
    paths: ProjectPaths,
    daily: pd.DataFrame,
    figures: list[Path],
    interactive_chart_path: Path,
    model_snapshot: pd.DataFrame,
    current_bottom_summary: pd.DataFrame,
    current_bottom_summary_path: Path,
    risk_table: pd.DataFrame,
    role_based_risk_table: pd.DataFrame,
    role_based_risk_path: Path,
    ensemble_risk_table: pd.DataFrame,
    forward_floor_thresholds: pd.DataFrame,
    forward_floor_thresholds_path: Path,
    forward_overlap_episodes: pd.DataFrame,
    forward_overlap_episodes_path: Path,
    expectile_sensitivity: pd.DataFrame,
    cycle_low_summary: pd.DataFrame,
    cycle_low_detail_path: Path,
    walk_forward_low_summary: pd.DataFrame,
    walk_forward_low_detail_path: Path,
    model_evidence_summary: pd.DataFrame,
    model_evidence_summary_path: Path,
    stability: pd.DataFrame,
    breach_details_path: Path,
    cycle_table: pd.DataFrame,
    phase: dict[str, object],
) -> Path:
    latest = daily.iloc[-1]
    figure_lines = [f"- `{path.as_posix()}`" for path in figures]
    phase_lines = [
        f"- Anchor cycle: {phase['anchor_cycle']}",
        f"- Anchor low date: {phase['anchor_low_date']:%Y-%m-%d}",
        f"- As-of date: {phase['as_of_date']:%Y-%m-%d}",
        f"- Days since anchor low: {phase['days_since_low']:,}",
        f"- Expected peak date: {phase['expected_peak_date']:%Y-%m-%d}",
        f"- Days to expected peak: {phase['days_to_expected_peak']:,}",
        f"- Expected next low date: {phase['expected_next_low_date']:%Y-%m-%d}",
        f"- Days to expected next low: {phase['days_to_expected_next_low']:,}",
    ]
    relevant_forward_overlap_episodes = forward_overlap_episodes.loc[
        pd.to_datetime(forward_overlap_episodes["start_date"]) >= pd.Timestamp("2013-01-01")
    ].copy()

    stability_view = stability.copy()
    for col in ("min_ratio_to_floor", "median_ratio_to_floor", "latest_ratio_to_floor"):
        stability_view[col] = stability_view[col].map(lambda x: f"{x:,.3f}")
    for col in ("intercept", "slope", "floor_offset_log10"):
        stability_view[col] = stability_view[col].map(lambda x: f"{x:,.6f}")
    for col in (
        "train_end_date",
        "first_breach_date",
        "last_breach_date",
        "min_ratio_date",
    ):
        stability_view[col] = pd.to_datetime(stability_view[col]).dt.strftime("%Y-%m-%d")
        stability_view[col] = stability_view[col].fillna("n/a")

    cycle_view = cycle_table.copy()
    for col in cycle_view.columns:
        if "date" in col:
            cycle_view[col] = pd.to_datetime(cycle_view[col]).dt.strftime("%Y-%m-%d")
            cycle_view[col] = cycle_view[col].fillna("n/a")
    for col in ("peak_timing_error_days", "next_low_timing_error_days"):
        cycle_view[col] = cycle_view[col].map(
            lambda x: "n/a" if pd.isna(x) else f"{int(x):,}"
        )

    report = "\n".join(
        [
            "# Initial BTC Floor Analysis",
            "",
            "## Latest Data Point",
            "",
            f"- Date: {latest['date']:%Y-%m-%d}",
            f"- BTC PriceUSD: {_money(float(latest['price_usd']))}",
            f"- Valid daily rows: {len(daily):,}",
            "",
            "## Current Bottom Summary",
            "",
            _format_current_bottom_summary_table(current_bottom_summary),
            "",
            f"- `{current_bottom_summary_path.as_posix()}`",
            "",
            "## Diagnostic Figures",
            "",
            *figure_lines,
            "",
            "## Interactive Chart",
            "",
            f"- `{interactive_chart_path.as_posix()}`",
            "- Wide layout: weekly candles, Giovanni fixed floor, expectile 0.01%/0.05%/0.1%/0.5%/1% variants, price-to-floor distance, cycle timing bands, and cycle-aligned Giovanni distance.",
            "- Baseline OLS is intentionally excluded from the interactive chart.",
            "",
            "## Model Snapshot",
            "",
            _format_report_table(model_snapshot),
            "",
            "## Forward Floor-Distance Risk",
            "",
            _format_report_table(risk_table),
            "",
            "## Role-Based Forward Bottom Risk",
            "",
            _format_role_based_risk_table(role_based_risk_table),
            "",
            "- Hard floor: Giovanni fixed floor.",
            "- Warning floor: 0.01% weekly expectile floor.",
            "- USD at risk is per 1 BTC and is capped at zero when the projected floor is already above current spot.",
            "- Bottom-pressure score = 45% hard-floor proximity + 35% warning-floor proximity + 20% cycle-time pressure.",
            f"- `{role_based_risk_path.as_posix()}`",
            "",
            "## Giovanni Forward-Floor Overlap Signal",
            "",
            f"- Decision threshold shown here: {_money(DECISION_FLOOR_THRESHOLD_USD)}.",
            f"- Overlap definition: daily close is below the Giovanni {FORWARD_OVERLAP_HORIZON_MONTHS}m-forward floor.",
            "- This is different from touching the same-day floor: it asks whether today's price overlaps a floor value that the model expects inside the forward window.",
            "",
            "Threshold timing:",
            "",
            _format_forward_threshold_table(forward_floor_thresholds),
            "",
            f"- `{forward_floor_thresholds_path.as_posix()}`",
            "",
            "Historical overlap episodes since 2013:",
            "",
            _format_forward_overlap_episode_table(relevant_forward_overlap_episodes),
            "",
            f"- Full episode table: `{forward_overlap_episodes_path.as_posix()}`",
            "",
            "## Ensemble And Time-Adjusted Risk",
            "",
            _format_ensemble_risk_table(ensemble_risk_table),
            "",
            "## Bottom Expectile Sensitivity",
            "",
            _format_expectile_sensitivity_table(expectile_sensitivity),
            "",
            "## Full-Sample Historical Cycle Low Validation",
            "",
            _format_cycle_low_validation_table(cycle_low_summary),
            "",
            "Detailed cycle-low rows:",
            "",
            f"- `{cycle_low_detail_path.as_posix()}`",
            "",
            "## Walk-Forward Cycle Low Validation",
            "",
            _format_walk_forward_summary_table(walk_forward_low_summary),
            "",
            "Detailed walk-forward rows:",
            "",
            f"- `{walk_forward_low_detail_path.as_posix()}`",
            "",
            "## Model Evidence Summary",
            "",
            _format_model_evidence_table(model_evidence_summary),
            "",
            f"- `{model_evidence_summary_path.as_posix()}`",
            "",
            "## Stability By Excluding Recent Cycles",
            "",
            stability_view.to_markdown(index=False),
            "",
            "Detailed breach rows:",
            "",
            f"- `{breach_details_path.as_posix()}`",
            "",
            "## 1428-Day Cycle Timing",
            "",
            cycle_view.to_markdown(index=False),
            "",
            "## Current Cycle Phase",
            "",
            *phase_lines,
            "",
            "## Notes",
            "",
            "- Giovanni's fixed power-law floor uses 1.0117e-17 * days^5.82 * 0.42.",
            "- Burger 2019 RANSAC support uses the Sep 2019 vintage window, trims to the closest 50% of log-log observations, and plots a -1.5 inlier-sigma lower band.",
            "- The OLS floor is fit on daily Coin Metrics prices with a -2 sigma log10 floor.",
            "- The expectile floor uses weekly closes and tau 0.0001, which corresponds to 0.01%.",
            "- The interactive chart excludes OLS to focus on Giovanni, Burger RANSAC, and expectile variants; OLS remains in tabular evidence as a conservative stress reference.",
            "- Stability rows with excluded cycles fit only on data through the selected cycle low cutoff and evaluate afterward.",
            "",
        ]
    )
    paths.initial_analysis_report.write_text(report, encoding="utf-8")
    return paths.initial_analysis_report


def run_initial_analysis(force_download: bool = False) -> dict[str, Path]:
    paths = ProjectPaths.from_cwd()
    daily = download_and_prepare(paths, force_download=force_download)
    quality = validate_price_history(
        paths.raw_btc_csv,
        daily,
        price_fixes_path=paths.price_fixes_csv,
    )
    quality_path = write_data_quality_report(paths, quality)
    figures = plot_price_diagnostics(daily, paths.figure_dir)

    weekly = to_weekly_close(daily)
    ols = fit_power_law_ols(daily)
    giovanni = giovanni_power_law_floor_model()
    burger_ransac = fit_burger_2019_ransac_floor(daily)
    expectile = fit_expectile_power_law(
        weekly,
        tau=0.0001,
        name=expectile_model_name(0.0001),
    )
    models = [ols, giovanni, burger_ransac, expectile]
    interactive_models = [
        giovanni,
        burger_ransac,
        expectile,
        *[
            fit_expectile_power_law(
                weekly,
                tau=tau,
                name=expectile_model_name(tau),
            )
            for tau in INTERACTIVE_EXPECTILE_TAUS
            if tau != 0.0001
        ],
    ]
    figures.append(_plot_floor_models(daily, models, paths.figure_dir))
    interactive_chart_path = write_interactive_weekly_floor_chart(
        daily,
        interactive_models,
        paths.interactive_dir / "btc_floor_weekly.html",
    )

    latest = daily.iloc[-1]
    cycle_table = cycle_timing_table()
    phase = current_cycle_phase(pd.Timestamp(latest["date"]))
    snapshot = _model_snapshot(models, latest)
    risk = pd.concat(
        [
            floor_distance_by_horizon(
                model,
                as_of_date=latest["date"],
                spot_price_usd=float(latest["price_usd"]),
            )
            for model in models
        ],
        ignore_index=True,
    )
    role_based_risk = role_based_floor_risk_by_horizon(
        hard_floor_model=giovanni,
        warning_floor_model=expectile,
        as_of_date=latest["date"],
        spot_price_usd=float(latest["price_usd"]),
        expected_low_date=phase["expected_next_low_date"],
    )
    current_role_based_risk = role_based_floor_risk_by_horizon(
        hard_floor_model=giovanni,
        warning_floor_model=expectile,
        as_of_date=latest["date"],
        spot_price_usd=float(latest["price_usd"]),
        expected_low_date=phase["expected_next_low_date"],
        horizons_months=(0,),
    )
    ensemble_risk = ensemble_floor_risk_by_horizon(
        models,
        as_of_date=latest["date"],
        spot_price_usd=float(latest["price_usd"]),
        expected_low_date=phase["expected_next_low_date"],
    )
    forward_floor_thresholds = floor_threshold_signal_dates(
        giovanni,
        target_floor_usd=DECISION_FLOOR_THRESHOLD_USD,
    )
    forward_overlap_daily = future_floor_overlap_daily(
        daily,
        giovanni,
        horizon_months=FORWARD_OVERLAP_HORIZON_MONTHS,
    )
    cycle_low_context = [
        *[(low.name, low.date) for low in DEFAULT_CYCLE_LOWS],
        ("2026_expected_low", pd.Timestamp(phase["expected_next_low_date"])),
    ]
    forward_overlap_episodes = add_nearest_cycle_low_context(
        group_future_floor_overlap_episodes(forward_overlap_daily),
        cycle_lows=cycle_low_context,
    )
    expectile_sensitivity = evaluate_bottom_expectile_sensitivity(
        weekly,
        daily,
        as_of_date=pd.Timestamp(latest["date"]),
        spot_price_usd=float(latest["price_usd"]),
    )
    cycle_low_validation = evaluate_cycle_low_windows(daily, models, window_days=90)
    cycle_low_summary = summarize_cycle_low_validation(cycle_low_validation)
    walk_forward_low_validation = evaluate_walk_forward_cycle_lows(
        daily,
        fit_specs=(
            WalkForwardFitSpec(
                name="ols_power_law_2sigma_floor",
                fit_model=lambda frame: fit_power_law_ols(frame),
            ),
            WalkForwardFitSpec(
                name="giovanni_power_law_floor",
                fit_model=lambda frame: giovanni_power_law_floor_model(),
            ),
            WalkForwardFitSpec(
                name="burger_2019_ransac_1_5sigma_floor",
                fit_model=lambda frame: fit_burger_2019_ransac_floor(frame),
            ),
            WalkForwardFitSpec(
                name=expectile_model_name(0.0001),
                fit_model=lambda frame: fit_expectile_power_law(
                    to_weekly_close(frame),
                    tau=0.0001,
                    name=expectile_model_name(0.0001),
                ),
            ),
        ),
        window_days=90,
    )
    walk_forward_low_summary = summarize_cycle_low_validation(walk_forward_low_validation)

    stability = pd.concat(
        [
            evaluate_model_stability(daily, lambda frame: fit_power_law_ols(frame)),
            evaluate_model_stability(
                daily,
                lambda frame: fit_expectile_power_law(
                    to_weekly_close(frame),
                    tau=0.0001,
                    name=expectile_model_name(0.0001),
                ),
            ),
        ],
        ignore_index=True,
    )
    breach_details = pd.concat(
        [
            collect_floor_breach_details(daily, lambda frame: fit_power_law_ols(frame)),
            collect_floor_breach_details(
                daily,
                lambda frame: fit_expectile_power_law(
                    to_weekly_close(frame),
                    tau=0.0001,
                    name=expectile_model_name(0.0001),
                ),
            ),
        ],
        ignore_index=True,
    )
    model_evidence_summary = _build_model_evidence_summary(
        model_snapshot=snapshot,
        cycle_low_summary=cycle_low_summary,
        walk_forward_low_summary=walk_forward_low_summary,
        stability=stability,
    )
    current_bottom_summary = _build_current_bottom_summary(
        model_snapshot=snapshot,
        current_role_based_risk=current_role_based_risk,
        forward_role_based_risk=role_based_risk,
        phase=phase,
    )
    snapshot.to_csv(paths.report_dir / "model_snapshot.csv", index=False)
    current_bottom_summary_path = paths.report_dir / "current_bottom_summary.csv"
    current_bottom_summary.to_csv(current_bottom_summary_path, index=False)
    risk.to_csv(paths.report_dir / "risk_horizons.csv", index=False)
    role_based_risk_path = paths.report_dir / "risk_role_based.csv"
    role_based_risk.to_csv(role_based_risk_path, index=False)
    ensemble_risk.to_csv(paths.report_dir / "risk_ensemble.csv", index=False)
    forward_floor_thresholds_path = paths.report_dir / "forward_floor_thresholds.csv"
    forward_floor_thresholds.to_csv(forward_floor_thresholds_path, index=False)
    forward_overlap_daily.to_csv(
        paths.report_dir / "forward_floor_overlap_daily.csv",
        index=False,
    )
    forward_overlap_episodes_path = paths.report_dir / "forward_floor_overlap_episodes.csv"
    forward_overlap_episodes.to_csv(forward_overlap_episodes_path, index=False)
    expectile_sensitivity.to_csv(paths.report_dir / "expectile_sensitivity.csv", index=False)
    cycle_low_summary.to_csv(paths.report_dir / "cycle_low_validation_summary.csv", index=False)
    cycle_low_detail_path = paths.report_dir / "cycle_low_validation.csv"
    cycle_low_validation.to_csv(cycle_low_detail_path, index=False)
    walk_forward_low_summary.to_csv(
        paths.report_dir / "walk_forward_cycle_low_summary.csv",
        index=False,
    )
    walk_forward_low_detail_path = paths.report_dir / "walk_forward_cycle_low_validation.csv"
    walk_forward_low_validation.to_csv(walk_forward_low_detail_path, index=False)
    model_evidence_summary_path = paths.report_dir / "model_evidence_summary.csv"
    model_evidence_summary.to_csv(model_evidence_summary_path, index=False)
    stability.to_csv(paths.report_dir / "stability.csv", index=False)
    breach_details_path = paths.report_dir / "stability_breaches.csv"
    breach_details.to_csv(breach_details_path, index=False)
    cycle_table.to_csv(paths.report_dir / "cycle_timing.csv", index=False)

    analysis_path = _write_initial_report(
        paths=paths,
        daily=daily,
        figures=figures,
        interactive_chart_path=interactive_chart_path,
        model_snapshot=snapshot,
        current_bottom_summary=current_bottom_summary,
        current_bottom_summary_path=current_bottom_summary_path,
        risk_table=risk,
        role_based_risk_table=role_based_risk,
        role_based_risk_path=role_based_risk_path,
        ensemble_risk_table=ensemble_risk,
        forward_floor_thresholds=forward_floor_thresholds,
        forward_floor_thresholds_path=forward_floor_thresholds_path,
        forward_overlap_episodes=forward_overlap_episodes,
        forward_overlap_episodes_path=forward_overlap_episodes_path,
        expectile_sensitivity=expectile_sensitivity,
        cycle_low_summary=cycle_low_summary,
        cycle_low_detail_path=cycle_low_detail_path,
        walk_forward_low_summary=walk_forward_low_summary,
        walk_forward_low_detail_path=walk_forward_low_detail_path,
        model_evidence_summary=model_evidence_summary,
        model_evidence_summary_path=model_evidence_summary_path,
        stability=stability,
        breach_details_path=breach_details_path,
        cycle_table=cycle_table,
        phase=phase,
    )
    return {
        "processed_data": paths.processed_btc_csv,
        "data_quality_report": quality_path,
        "initial_analysis_report": analysis_path,
        "current_bottom_summary": current_bottom_summary_path,
        "role_based_risk": role_based_risk_path,
        "forward_floor_thresholds": forward_floor_thresholds_path,
        "forward_floor_overlap_episodes": forward_overlap_episodes_path,
        "interactive_chart": interactive_chart_path,
    }
