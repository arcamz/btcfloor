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
