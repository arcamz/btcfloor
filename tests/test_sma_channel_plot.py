from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_sma_channel_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_sma_channel_decision.py"
    spec = importlib.util.spec_from_file_location("plot_sma_channel_decision", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sma_channel = _load_sma_channel_module()


def test_first_sma50_loss_after_rejection_returns_none_when_no_loss() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]),
            "price_usd": [101.0, 102.0, 103.0],
            "sma50": [100.0, 100.0, 100.0],
        }
    )

    result = sma_channel.first_sma50_loss_after_rejection(daily, pd.Timestamp("2026-04-01"))

    assert result is None


def test_first_sma50_loss_after_rejection_returns_first_loss() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]),
            "price_usd": [101.0, 99.0, 98.0],
            "sma50": [100.0, 100.0, 100.0],
        }
    )

    result = sma_channel.first_sma50_loss_after_rejection(daily, pd.Timestamp("2026-04-01"))

    assert result is not None
    assert result["date"] == pd.Timestamp("2026-04-02")
