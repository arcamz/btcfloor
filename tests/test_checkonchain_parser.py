from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_checkonchain_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_checkonchain_cohorts.py"
    spec = importlib.util.spec_from_file_location("plot_checkonchain_cohorts", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checkonchain = _load_checkonchain_module()


class _FakeResponse:
    def __init__(self, text: str, url: str) -> None:
        self.text = text
        self.url = url

    def raise_for_status(self) -> None:
        return None


def test_fetch_chart_follows_plotly_iframe_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper_url = "https://charts.example.test/path/chart.html"
    payload_url = "https://cdn.example.test/path/chart.html"
    payload = (
        '<script>Plotly.newPlot("chart", '
        '[{"name":"Price","x":["2026-07-12"],"y":[123.45]}], {});</script>'
    )
    requests_seen: list[str] = []

    def fake_get(url: str, **_: object) -> _FakeResponse:
        requests_seen.append(url)
        if url == wrapper_url:
            return _FakeResponse(
                '<iframe src="https://cdn.example.test/path/chart.html"></iframe>',
                wrapper_url,
            )
        assert url == payload_url
        return _FakeResponse(payload, payload_url)

    monkeypatch.setattr(checkonchain.requests, "get", fake_get)

    actual = checkonchain.fetch_checkonchain_chart(wrapper_url)

    assert requests_seen == [wrapper_url, payload_url]
    assert actual.to_dict("records") == [{"date": checkonchain.pd.Timestamp("2026-07-12"), "Price": 123.45}]


def test_iframe_sources_resolve_relative_urls() -> None:
    sources = checkonchain._iframe_sources(
        '<iframe src="../payload/chart.html"></iframe>',
        "https://charts.example.test/a/b/wrapper.html",
    )

    assert sources == ["https://charts.example.test/a/payload/chart.html"]
