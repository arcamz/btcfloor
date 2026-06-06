from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_dashboard_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_interactive_dashboards.py"
    )
    spec = importlib.util.spec_from_file_location("build_interactive_dashboards", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cvdd_display_name = _load_dashboard_module()._cvdd_display_name


def test_cvdd_display_name_prefers_bitbo_when_available() -> None:
    summary = {
        "source": {
            "cvdd": {
                "bitbo": {"available": True},
                "looknode": {"available": True},
            }
        }
    }

    assert _cvdd_display_name(summary) == "CVDD (Bitbo)"


def test_cvdd_display_name_marks_looknode_fallback() -> None:
    summary = {
        "source": {
            "cvdd": {
                "bitbo": {"available": False},
                "looknode": {"available": True},
            }
        }
    }

    assert _cvdd_display_name(summary) == "CVDD (Looknode fallback)"
