from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_health_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_pipeline_health_dashboard.py"
    )
    spec = importlib.util.spec_from_file_location("build_pipeline_health_dashboard", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


health = _load_health_module()


def test_status_for_lag_days_classifies_freshness() -> None:
    assert health._status_for_lag_days(None, ok_days=1, warn_days=3) == "missing"
    assert health._status_for_lag_days(1, ok_days=1, warn_days=3) == "ok"
    assert health._status_for_lag_days(3, ok_days=1, warn_days=3) == "warn"
    assert health._status_for_lag_days(4, ok_days=1, warn_days=3) == "stale"


def test_overall_status_escalates_warnings_and_stale_items() -> None:
    assert health._overall_status([{"status": "ok"}]) == "ok"
    assert health._overall_status([{"status": "ok"}, {"status": "warn"}]) == "warn"
    assert health._overall_status([{"status": "warn"}, {"status": "stale"}]) == "stale"
    assert health._overall_status([{"status": "missing"}]) == "stale"
