from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from btcfloor.dashboard_common import DASHBOARD_LINKS
from btcfloor.paths import ProjectPaths


def _load_static_site_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_static_site.py"
    spec = importlib.util.spec_from_file_location("build_static_site", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


static_site = _load_static_site_module()


def _write_required_dashboards(paths: ProjectPaths) -> None:
    for link in DASHBOARD_LINKS:
        (paths.interactive_dir / link.href).write_text(
            f"<html>{link.label}</html>",
            encoding="utf-8",
        )


def test_build_static_site_packages_private_artifact_tree(tmp_path: Path) -> None:
    paths = ProjectPaths.from_cwd(tmp_path)
    paths.ensure_dirs()
    _write_required_dashboards(paths)

    (paths.figure_dir / "sma_channel_decision_plot.png").write_bytes(b"figure")
    (paths.report_dir / "current_bottom_summary.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (paths.report_dir / "pipeline_health.json").write_text(
        '{"generated_at": "2026-06-08T00:09:26+02:00"}',
        encoding="utf-8",
    )
    (paths.report_dir / "data_quality.md").write_text("# Data quality\n", encoding="utf-8")
    (paths.report_dir / "ignored.txt").write_text("not packaged\n", encoding="utf-8")
    (paths.raw_dir / "coinmetrics_btc.csv").write_text("not packaged\n", encoding="utf-8")
    (paths.processed_dir / "btc_daily.csv").write_text("not packaged\n", encoding="utf-8")

    stale_file = paths.root / "dist" / "site" / "stale.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old", encoding="utf-8")

    site_root = static_site.build_static_site(paths)

    assert site_root == (paths.root / "dist" / "site").resolve()
    assert (site_root / ".nojekyll").exists()
    assert not stale_file.exists()
    assert not (site_root / "data").exists()

    index = (site_root / "index.html").read_text(encoding="utf-8")
    assert "2026-06-08T00:09:26+02:00" in index
    for link in DASHBOARD_LINKS:
        assert f"reports/interactive/{link.href}" in index
        assert (site_root / "reports" / "interactive" / link.href).exists()

    assert (site_root / "reports" / "figures" / "sma_channel_decision_plot.png").exists()
    assert (site_root / "reports" / "current_bottom_summary.csv").exists()
    assert (site_root / "reports" / "pipeline_health.json").exists()
    assert (site_root / "reports" / "data_quality.md").exists()
    assert not (site_root / "reports" / "ignored.txt").exists()


def test_build_static_site_requires_generated_dashboards(tmp_path: Path) -> None:
    paths = ProjectPaths.from_cwd(tmp_path)
    paths.ensure_dirs()

    with pytest.raises(FileNotFoundError, match="Missing primary dashboard"):
        static_site.build_static_site(paths)
