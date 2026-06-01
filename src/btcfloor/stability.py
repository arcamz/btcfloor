from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from btcfloor.powerlaw import PowerLawModel


CYCLE_LOW_CUTOFFS = (
    pd.Timestamp("2015-01-14"),
    pd.Timestamp("2018-12-15"),
    pd.Timestamp("2022-11-21"),
)


def cutoff_for_excluding_last_cycles(exclude_last_cycles: int) -> pd.Timestamp | None:
    if exclude_last_cycles == 0:
        return None
    if exclude_last_cycles < 0 or exclude_last_cycles > len(CYCLE_LOW_CUTOFFS):
        raise ValueError(
            f"exclude_last_cycles must be between 0 and {len(CYCLE_LOW_CUTOFFS)}"
        )
    return CYCLE_LOW_CUTOFFS[-exclude_last_cycles]


def evaluate_model_stability(
    daily: pd.DataFrame,
    fit_model: Callable[[pd.DataFrame], PowerLawModel],
    exclude_last_cycles_values: Iterable[int] = (0, 1, 2, 3),
) -> pd.DataFrame:
    rows = []
    for excluded in exclude_last_cycles_values:
        cutoff = cutoff_for_excluding_last_cycles(int(excluded))
        if cutoff is None:
            train = daily.copy()
            evaluation = daily.copy()
        else:
            train = daily[daily["date"] <= cutoff].copy()
            evaluation = daily[daily["date"] > cutoff].copy()

        model = fit_model(train)
        floor = model.predict_price(evaluation["date"], floor=True)
        ratio_to_floor = evaluation["price_usd"].to_numpy(dtype=float) / floor
        breaches = ratio_to_floor < 1.0
        min_ratio_index = int(np.argmin(ratio_to_floor))
        breach_dates = evaluation.loc[breaches, "date"]
        rows.append(
            {
                "model": model.name,
                "exclude_last_cycles": int(excluded),
                "train_end_date": model.fitted_to,
                "train_rows": model.n_obs,
                "evaluation_rows": len(evaluation),
                "floor_breach_days": int(np.sum(breaches)),
                "first_breach_date": (
                    pd.NaT if breach_dates.empty else breach_dates.iloc[0]
                ),
                "last_breach_date": (
                    pd.NaT if breach_dates.empty else breach_dates.iloc[-1]
                ),
                "min_ratio_date": evaluation["date"].iloc[min_ratio_index],
                "min_ratio_to_floor": float(np.min(ratio_to_floor)),
                "median_ratio_to_floor": float(np.median(ratio_to_floor)),
                "latest_ratio_to_floor": float(ratio_to_floor[-1]),
                "intercept": model.intercept,
                "slope": model.slope,
                "floor_offset_log10": model.floor_offset_log10,
            }
        )
    return pd.DataFrame(rows)


def collect_floor_breach_details(
    daily: pd.DataFrame,
    fit_model: Callable[[pd.DataFrame], PowerLawModel],
    exclude_last_cycles_values: Iterable[int] = (0, 1, 2, 3),
) -> pd.DataFrame:
    rows = []
    for excluded in exclude_last_cycles_values:
        cutoff = cutoff_for_excluding_last_cycles(int(excluded))
        if cutoff is None:
            train = daily.copy()
            evaluation = daily.copy()
        else:
            train = daily[daily["date"] <= cutoff].copy()
            evaluation = daily[daily["date"] > cutoff].copy()

        model = fit_model(train)
        floor = model.predict_price(evaluation["date"], floor=True)
        ratio_to_floor = evaluation["price_usd"].to_numpy(dtype=float) / floor
        breach_frame = evaluation.loc[ratio_to_floor < 1.0, ["date", "price_usd"]].copy()
        if breach_frame.empty:
            continue

        breach_frame["model"] = model.name
        breach_frame["exclude_last_cycles"] = int(excluded)
        breach_frame["floor_usd"] = floor[ratio_to_floor < 1.0]
        breach_frame["ratio_to_floor"] = ratio_to_floor[ratio_to_floor < 1.0]
        rows.append(
            breach_frame.loc[
                :,
                [
                    "model",
                    "exclude_last_cycles",
                    "date",
                    "price_usd",
                    "floor_usd",
                    "ratio_to_floor",
                ],
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "exclude_last_cycles",
                "date",
                "price_usd",
                "floor_usd",
                "ratio_to_floor",
            ]
        )
    return pd.concat(rows, ignore_index=True)
