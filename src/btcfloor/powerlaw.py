from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from btcfloor.data import GENESIS_DATE


GIOVANNI_POWER_LAW_MULTIPLIER = 1.0117e-17
GIOVANNI_POWER_LAW_EXPONENT = 5.82
GIOVANNI_FLOOR_MULTIPLIER = 0.42
BURGER_2019_START_DATE = pd.Timestamp("2010-07-17")
BURGER_2019_END_DATE = pd.Timestamp("2019-09-03")


@dataclass(frozen=True)
class PowerLawModel:
    name: str
    intercept: float
    slope: float
    residual_sigma_log10: float
    floor_offset_log10: float
    fitted_from: pd.Timestamp | None
    fitted_to: pd.Timestamp | None
    n_obs: int
    method: str

    def trend_log10_from_days(self, days_since_genesis: np.ndarray) -> np.ndarray:
        days = np.asarray(days_since_genesis, dtype=float)
        return self.intercept + self.slope * np.log10(days)

    def floor_log10_from_days(self, days_since_genesis: np.ndarray) -> np.ndarray:
        return self.trend_log10_from_days(days_since_genesis) - self.floor_offset_log10

    def predict_price(
        self,
        dates: pd.Series | Iterable[pd.Timestamp] | pd.Timestamp,
        floor: bool = True,
    ) -> np.ndarray:
        date_values = pd.to_datetime(dates)
        if isinstance(date_values, pd.Timestamp):
            date_index = pd.DatetimeIndex([date_values])
        else:
            date_index = pd.DatetimeIndex(date_values)
        days = (date_index - GENESIS_DATE).days.to_numpy()
        log_values = (
            self.floor_log10_from_days(days)
            if floor
            else self.trend_log10_from_days(days)
        )
        return np.power(10.0, log_values)

    def predict_frame(self, dates: pd.Series, floor: bool = True) -> pd.DataFrame:
        prices = self.predict_price(dates, floor=floor)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "model": self.name,
                "price_usd": prices,
                "line": "floor" if floor else "trend",
            }
        )


def _valid_loglog_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "days_since_genesis", "price_usd"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Price frame missing columns: {sorted(missing)}")

    valid = df.loc[
        (df["days_since_genesis"] > 0) & (df["price_usd"] > 0),
        ["date", "days_since_genesis", "price_usd"],
    ].copy()
    if len(valid) < 3:
        raise ValueError("Need at least 3 valid observations for log-log fit")
    return valid


def _fit_loglog_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def _burger_ransac_inlier_mask(
    x: np.ndarray,
    y: np.ndarray,
    keep_fraction: float,
) -> np.ndarray:
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be between 0 and 1")

    keep = np.ones(len(x), dtype=bool)
    target_keep = max(3, int(np.floor(len(x) * keep_fraction)))
    intercept, slope = _fit_loglog_line(x, y)

    while int(keep.sum()) > target_keep:
        kept_indices = np.flatnonzero(keep)
        residuals = y[keep] - (intercept + slope * x[keep])
        worst_index = kept_indices[int(np.argmax(np.abs(residuals)))]
        keep[worst_index] = False
        intercept, slope = _fit_loglog_line(x[keep], y[keep])

    return keep


def fit_power_law_ols(
    df: pd.DataFrame,
    floor_sigma: float = 2.0,
    name: str = "ols_power_law_2sigma_floor",
) -> PowerLawModel:
    valid = _valid_loglog_frame(df)
    x = np.log10(valid["days_since_genesis"].to_numpy(dtype=float))
    y = np.log10(valid["price_usd"].to_numpy(dtype=float))
    intercept, slope = _fit_loglog_line(x, y)
    residuals = y - (intercept + slope * x)
    sigma = float(np.std(residuals, ddof=2))
    return PowerLawModel(
        name=name,
        intercept=float(intercept),
        slope=float(slope),
        residual_sigma_log10=sigma,
        floor_offset_log10=float(floor_sigma * sigma),
        fitted_from=valid["date"].min(),
        fitted_to=valid["date"].max(),
        n_obs=len(valid),
        method=f"ordinary least squares floor at -{floor_sigma:g} sigma",
    )


def fit_burger_2019_ransac_floor(
    df: pd.DataFrame,
    floor_sigma: float = 1.5,
    keep_fraction: float = 0.5,
    start_date: pd.Timestamp = BURGER_2019_START_DATE,
    end_date: pd.Timestamp = BURGER_2019_END_DATE,
    name: str | None = None,
) -> PowerLawModel:
    if floor_sigma < 0.0:
        raise ValueError("floor_sigma must be non-negative")

    valid = _valid_loglog_frame(df)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    vintage = valid.loc[valid["date"].between(start, end)].copy()
    if len(vintage) < 6:
        raise ValueError("Need at least 6 vintage observations for Burger RANSAC fit")

    x = np.log10(vintage["days_since_genesis"].to_numpy(dtype=float))
    y = np.log10(vintage["price_usd"].to_numpy(dtype=float))
    inliers = _burger_ransac_inlier_mask(x, y, keep_fraction=keep_fraction)
    intercept, slope = _fit_loglog_line(x[inliers], y[inliers])
    residuals = y[inliers] - (intercept + slope * x[inliers])
    sigma = float(np.std(residuals, ddof=2))
    label = name or f"burger_2019_ransac_{floor_sigma:g}sigma_floor".replace(".", "_")

    return PowerLawModel(
        name=label,
        intercept=intercept,
        slope=slope,
        residual_sigma_log10=sigma,
        floor_offset_log10=float(floor_sigma * sigma),
        fitted_from=vintage.loc[inliers, "date"].min(),
        fitted_to=vintage.loc[inliers, "date"].max(),
        n_obs=int(np.sum(inliers)),
        method=(
            "Burger 2019 vintage RANSAC-style log-log fit; "
            f"source window {vintage['date'].min():%Y-%m-%d} to "
            f"{vintage['date'].max():%Y-%m-%d}; "
            f"kept {keep_fraction:g} of observations; "
            f"floor at -{floor_sigma:g} inlier sigma"
        ),
    )


def giovanni_power_law_floor_model() -> PowerLawModel:
    return PowerLawModel(
        name="giovanni_power_law_floor",
        intercept=float(np.log10(GIOVANNI_POWER_LAW_MULTIPLIER)),
        slope=GIOVANNI_POWER_LAW_EXPONENT,
        residual_sigma_log10=0.0,
        floor_offset_log10=float(-np.log10(GIOVANNI_FLOOR_MULTIPLIER)),
        fitted_from=None,
        fitted_to=None,
        n_obs=0,
        method=(
            "Giovanni power law: 1.0117e-17 * days^5.82; "
            "floor = trend x 0.42"
        ),
    )


def santostasi_perrenod_floor_model() -> PowerLawModel:
    return giovanni_power_law_floor_model()
