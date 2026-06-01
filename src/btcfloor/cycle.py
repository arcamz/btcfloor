from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


FOURCHAN_LOW_TO_PEAK_DAYS = 1064
FOURCHAN_PEAK_TO_LOW_DAYS = 364
FOURCHAN_FULL_LOW_TO_LOW_DAYS = FOURCHAN_LOW_TO_PEAK_DAYS + FOURCHAN_PEAK_TO_LOW_DAYS


@dataclass(frozen=True)
class CycleTimingAnchor:
    name: str
    low_date: pd.Timestamp
    observed_peak_date: pd.Timestamp | None = None
    observed_next_low_date: pd.Timestamp | None = None


DEFAULT_CYCLE_ANCHORS = (
    CycleTimingAnchor(
        name="2015_cycle",
        low_date=pd.Timestamp("2015-01-14"),
        observed_peak_date=pd.Timestamp("2017-12-17"),
        observed_next_low_date=pd.Timestamp("2018-12-15"),
    ),
    CycleTimingAnchor(
        name="2018_cycle",
        low_date=pd.Timestamp("2018-12-15"),
        observed_peak_date=pd.Timestamp("2021-11-10"),
        observed_next_low_date=pd.Timestamp("2022-11-21"),
    ),
    CycleTimingAnchor(
        name="2022_cycle",
        low_date=pd.Timestamp("2022-11-21"),
        observed_peak_date=None,
        observed_next_low_date=None,
    ),
)


def cycle_timing_table(
    anchors: tuple[CycleTimingAnchor, ...] = DEFAULT_CYCLE_ANCHORS,
) -> pd.DataFrame:
    rows = []
    for anchor in anchors:
        expected_peak = anchor.low_date + pd.Timedelta(days=FOURCHAN_LOW_TO_PEAK_DAYS)
        expected_next_low = anchor.low_date + pd.Timedelta(days=FOURCHAN_FULL_LOW_TO_LOW_DAYS)
        rows.append(
            {
                "cycle": anchor.name,
                "low_date": anchor.low_date,
                "expected_peak_date": expected_peak,
                "observed_peak_date": anchor.observed_peak_date,
                "peak_timing_error_days": (
                    None
                    if anchor.observed_peak_date is None
                    else int((anchor.observed_peak_date - expected_peak).days)
                ),
                "expected_next_low_date": expected_next_low,
                "observed_next_low_date": anchor.observed_next_low_date,
                "next_low_timing_error_days": (
                    None
                    if anchor.observed_next_low_date is None
                    else int((anchor.observed_next_low_date - expected_next_low).days)
                ),
            }
        )
    return pd.DataFrame(rows)


def current_cycle_phase(
    as_of_date: pd.Timestamp,
    anchor: CycleTimingAnchor = DEFAULT_CYCLE_ANCHORS[-1],
) -> dict[str, object]:
    as_of = pd.Timestamp(as_of_date)
    days_since_low = int((as_of - anchor.low_date).days)
    expected_peak = anchor.low_date + pd.Timedelta(days=FOURCHAN_LOW_TO_PEAK_DAYS)
    expected_next_low = anchor.low_date + pd.Timedelta(days=FOURCHAN_FULL_LOW_TO_LOW_DAYS)
    return {
        "anchor_cycle": anchor.name,
        "anchor_low_date": anchor.low_date,
        "as_of_date": as_of,
        "days_since_low": days_since_low,
        "expected_peak_date": expected_peak,
        "days_to_expected_peak": int((expected_peak - as_of).days),
        "expected_next_low_date": expected_next_low,
        "days_to_expected_next_low": int((expected_next_low - as_of).days),
    }
