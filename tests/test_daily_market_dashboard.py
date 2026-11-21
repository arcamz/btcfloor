from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def dashboard(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("build_daily_market_dashboard")


def _candles(dates, closes):
    return pd.DataFrame({
        "date": dates, "open": closes, "high": closes, "low": closes, "close": closes,
    })


def test_weekly_sma_updates_only_on_completed_sunday(dashboard):
    dates = pd.date_range("2025-01-06", periods=357, freq="D")
    candles = _candles(dates, range(1, 358))
    average = dashboard.technical_indicators(candles)["sma50w"]

    assert pd.isna(average.iloc[348])
    assert average.iloc[349] == pytest.approx((7 + 350) / 2)
    assert list(average.iloc[350:356]) == [average.iloc[349]] * 6
    assert average.iloc[356] == pytest.approx((14 + 357) / 2)


def test_holder_metrics_follow_candle_dates_and_keep_gaps(dashboard):
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    cohort = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-03", "2025-01-04"],
        "STH Realised Price": [95.0, 96.0, 99.0],
        "CVDD": [70.0, 71.0, 72.0],
    })
    summary = {"source": {"cvdd": {"looknode": {"available": True}}}}
    payload = dashboard.payload_for(_candles(dates, [100, 105, 110]), cohort, summary)
    metrics = {metric["key"]: metric for metric in payload["metrics"]}

    assert payload["dates"] == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert metrics["sth_price"]["values"]["sth_price"] == [95.0, None, 96.0]
    assert metrics["cvdd"]["label"] == "CVDD (Looknode fallback)"
    assert metrics["cvdd"]["last"] == "2025-01-03"
    assert metrics["lth_sopr"]["last"] is None


def test_bitstamp_preserves_ohlc_and_excludes_following_candle(dashboard, monkeypatch):
    day = pd.Timestamp("2025-01-01")
    midnight = int(day.tz_localize("UTC").timestamp())
    monkeypatch.setattr(dashboard, "FIRST_BITSTAMP_DAY", day)

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"ohlc": [
                {"timestamp": str(midnight), "open": "100", "high": "115", "low": "95", "close": "108"},
                {"timestamp": str(midnight + 86400), "open": "108", "high": "123", "low": "101", "close": "102"},
            ]}}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def get(self, url, *, params, timeout):
            assert params["step"] == 86400
            assert params["exclude_current_candle"] == "true"
            assert params["start"] == params["end"] == midnight
            return Response()

    monkeypatch.setattr(dashboard.requests, "Session", Session)
    candles = dashboard.fetch_candles(day)

    assert candles.to_dict("records") == [
        {"date": day, "open": 100, "high": 115, "low": 95, "close": 108},
    ]
