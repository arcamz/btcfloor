from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_btc_gold_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    module_path = scripts_dir / "build_btc_gold_dashboard.py"
    spec = importlib.util.spec_from_file_location("build_btc_gold_dashboard", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


btc_gold = _load_btc_gold_module()


def test_build_ratio_frame_uses_latest_shared_dates_only() -> None:
    btc = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "price_usd": [100_000.0, 110_000.0, 120_000.0],
        }
    )
    btc["days_since_genesis"] = [6200, 6201, 6202]
    btc["source"] = "synthetic"
    gold = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "price": [5_000.0, 5_500.0],
        }
    )

    frame = btc_gold._build_ratio_frame(btc, gold)

    assert list(frame["date"]) == list(pd.to_datetime(["2026-01-01", "2026-01-02"]))
    assert list(frame["btc_xau"]) == [20.0, 20.0]
