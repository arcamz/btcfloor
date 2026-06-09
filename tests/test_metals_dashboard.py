from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from btcfloor.paths import ProjectPaths


def _load_metals_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_metals_dashboard.py"
    spec = importlib.util.spec_from_file_location("build_metals_dashboard", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metals = _load_metals_module()


def _write_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("date,price\n2026-01-29,100\n2026-06-08,120\n", encoding="utf-8")


def test_legacy_lbma_uses_bundled_snapshot_without_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = ProjectPaths.from_cwd(tmp_path)
    snapshot = tmp_path / "resources" / "legacy" / "lbma_gold_pm.csv"
    _write_snapshot(snapshot)

    def fail_fetch(url: str) -> object:
        raise AssertionError(f"unexpected LBMA fetch: {url}")

    monkeypatch.setattr(metals, "_fetch_json", fail_fetch)
    monkeypatch.delenv(metals.LBMA_REFRESH_ENV, raising=False)

    series = metals._load_legacy_lbma_series(
        paths=paths,
        name="Gold PM fix",
        url=metals.GOLD_PM_URL,
        filename="lbma_gold_pm.csv",
    )

    assert len(series.data) == 2
    assert series.status["source"] == "bundled_snapshot"
    assert series.status["latest_date"] == "2026-06-08"
    assert series.status["refreshed"] is False


def test_legacy_lbma_manual_refresh_updates_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = ProjectPaths.from_cwd(tmp_path)
    snapshot = tmp_path / "resources" / "legacy" / "lbma_gold_pm.csv"
    _write_snapshot(snapshot)

    monkeypatch.setenv(metals.LBMA_REFRESH_ENV, "1")
    monkeypatch.setattr(
        metals,
        "_fetch_json",
        lambda url: [
            {"d": "2026-01-29", "v": [101]},
            {"d": "2026-06-09", "v": [125]},
        ],
    )

    series = metals._load_legacy_lbma_series(
        paths=paths,
        name="Gold PM fix",
        url=metals.GOLD_PM_URL,
        filename="lbma_gold_pm.csv",
    )

    assert series.status["source"] == "api_manual_refresh"
    assert series.status["latest_date"] == "2026-06-09"
    assert series.status["refreshed"] is True
    assert "2026-06-09" in snapshot.read_text(encoding="utf-8")


def test_legacy_lbma_failed_manual_refresh_falls_back_to_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ProjectPaths.from_cwd(tmp_path)
    snapshot = tmp_path / "resources" / "legacy" / "lbma_gold_pm.csv"
    _write_snapshot(snapshot)

    monkeypatch.setenv(metals.LBMA_REFRESH_ENV, "true")

    def fail_fetch(url: str) -> object:
        raise RuntimeError("temporary LBMA outage")

    monkeypatch.setattr(metals, "_fetch_json", fail_fetch)

    series = metals._load_legacy_lbma_series(
        paths=paths,
        name="Gold PM fix",
        url=metals.GOLD_PM_URL,
        filename="lbma_gold_pm.csv",
    )

    assert len(series.data) == 2
    assert series.status["source"] == "bundled_snapshot_after_refresh_failure"
    assert "temporary LBMA outage" in str(series.status["detail"])


def test_metals_analog_charts_use_live_futures_lines_and_dashed_2026_lbma() -> None:
    root = Path(__file__).resolve().parents[1]
    gold = metals._load_lbma_snapshot(root / "resources" / "legacy" / "lbma_gold_pm.csv")
    silver = metals._load_lbma_snapshot(root / "resources" / "legacy" / "lbma_silver.csv")
    live_gold = gold[gold["date"].between("2026-01-29", "2026-06-05")].copy()
    live_gold["price"] = live_gold["price"] * 0.99
    live_gold.loc[len(live_gold)] = [metals.pd.Timestamp("2026-06-09"), 4323.0]
    live_silver = silver[silver["date"].between("2026-01-29", "2026-06-05")].copy()
    live_silver["price"] = live_silver["price"] * 0.95
    live_silver.loc[len(live_silver)] = [metals.pd.Timestamp("2026-06-09"), 66.13]

    gold_analogs, _, _ = metals._build_gold_analogs(gold)
    gold_channel = metals._build_channel_model(gold)
    silver_analogs, _, _ = metals._build_silver_analogs(silver)

    gold_fig = metals._make_gold_chart(gold_analogs, gold, gold_channel, live_gold)
    silver_fig = metals._make_silver_chart(silver_analogs, silver.iloc[-1], live_silver)

    gold_traces = {trace.name: trace for trace in gold_fig.data}
    silver_traces = {trace.name: trace for trace in silver_fig.data}

    assert "2026 live GC=F" in gold_traces
    assert "2026 LBMA PM fix static" in gold_traces
    assert gold_traces["2026 LBMA PM fix static"].line.dash == "dash"
    assert len(gold_traces["2026 live GC=F"].x) > 1

    assert "2026 live SI=F" in silver_traces
    assert "2026 LBMA silver static" in silver_traces
    assert silver_traces["2026 LBMA silver static"].line.dash == "dash"
    assert len(silver_traces["2026 live SI=F"].x) > 1
