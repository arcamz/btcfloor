from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from btcfloor.data import GENESIS_DATE


GIOVANNI_POWER_LAW_MULTIPLIER = 1.0117e-17
GIOVANNI_POWER_LAW_EXPONENT = 5.82
GIOVANNI_FLOOR_MULTIPLIER = 0.42


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


def fit_power_law_ols(
    df: pd.DataFrame,
    floor_sigma: float = 2.0,
    name: str = "ols_power_law_2sigma_floor",
) -> PowerLawModel:
    valid = _valid_loglog_frame(df)
    x = np.log10(valid["days_since_genesis"].to_numpy(dtype=float))
    y = np.log10(valid["price_usd"].to_numpy(dtype=float))
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
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
