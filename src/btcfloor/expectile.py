from __future__ import annotations

import numpy as np
import pandas as pd

from btcfloor.powerlaw import PowerLawModel, _valid_loglog_frame
from btcfloor.validation import evaluate_cycle_low_windows, summarize_cycle_low_validation


DEFAULT_BOTTOM_EXPECTILE_TAUS = (0.0001, 0.0005, 0.001, 0.005, 0.01)


def fit_expectile_power_law(
    df: pd.DataFrame,
    tau: float = 0.0001,
    max_iter: int = 500,
    tolerance: float = 1e-10,
    name: str | None = None,
) -> PowerLawModel:
    if not 0.0 < tau < 1.0:
        raise ValueError("Expectile tau must be between 0 and 1")

    valid = _valid_loglog_frame(df)
    x = np.log10(valid["days_since_genesis"].to_numpy(dtype=float))
    y = np.log10(valid["price_usd"].to_numpy(dtype=float))
    design = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]

    for _ in range(max_iter):
        residual = y - design @ beta
        weights = np.where(residual >= 0.0, tau, 1.0 - tau)
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        next_beta = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        if np.max(np.abs(next_beta - beta)) < tolerance:
            beta = next_beta
            break
        beta = next_beta

    residuals = y - design @ beta
    sigma = float(np.std(residuals, ddof=2))
    label = name or f"expectile_power_law_tau_{tau:g}"
    return PowerLawModel(
        name=label,
        intercept=float(beta[0]),
        slope=float(beta[1]),
        residual_sigma_log10=sigma,
        floor_offset_log10=0.0,
        fitted_from=valid["date"].min(),
        fitted_to=valid["date"].max(),
        n_obs=len(valid),
        method=f"asymmetric least squares expectile tau={tau:g}",
    )


def expectile_model_name(tau: float) -> str:
    return f"weekly_expectile_power_law_tau_{tau:g}".replace(".", "_")


def evaluate_bottom_expectile_sensitivity(
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    as_of_date: pd.Timestamp,
    spot_price_usd: float,
    taus: tuple[float, ...] = DEFAULT_BOTTOM_EXPECTILE_TAUS,
) -> pd.DataFrame:
    rows = []
    for tau in taus:
        model = fit_expectile_power_law(
            weekly,
            tau=tau,
            name=expectile_model_name(tau),
        )
        floor = float(model.predict_price(as_of_date, floor=True)[0])
        validation = evaluate_cycle_low_windows(daily, [model], window_days=90)
        summary = summarize_cycle_low_validation(validation).iloc[0]
        rows.append(
            {
                "tau": tau,
                "tau_percent": tau * 100.0,
                "model": model.name,
                "as_of_date": pd.Timestamp(as_of_date),
                "spot_price_usd": spot_price_usd,
                "floor_usd": floor,
                "pct_above_floor": spot_price_usd / floor - 1.0,
                "slope": model.slope,
                "intercept": model.intercept,
                "fit_rows": model.n_obs,
                "cycles_evaluated": int(summary["cycles_evaluated"]),
                "median_ratio_at_observed_low": float(
                    summary["median_ratio_at_observed_low"]
                ),
                "cycles_below_floor_at_low": int(summary["cycles_below_floor_at_low"]),
                "cycles_with_window_breach": int(summary["cycles_with_window_breach"]),
                "total_breach_days_in_windows": int(
                    summary["total_breach_days_in_windows"]
                ),
                "floor_quality_score": float(summary["floor_quality_score"]),
            }
        )
    return pd.DataFrame(rows)
