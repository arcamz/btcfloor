from __future__ import annotations

from btcfloor import cli


def test_main_dispatches_download(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_download(force: bool) -> int:
        calls.append(("download", force))
        return 0

    monkeypatch.setattr(cli, "cmd_download", fake_download)

    assert cli.main(["download", "--force"]) == 0
    assert calls == [("download", True)]


def test_main_dispatches_analyze(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_analyze(force_download: bool) -> int:
        calls.append(("analyze", force_download))
        return 0

    monkeypatch.setattr(cli, "cmd_analyze", fake_analyze)

    assert cli.main(["analyze", "--force-download"]) == 0
    assert calls == [("analyze", True)]


def test_main_dispatches_chart(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_chart(force_download: bool) -> int:
        calls.append(("chart", force_download))
        return 0

    monkeypatch.setattr(cli, "cmd_chart", fake_chart)

    assert cli.main(["chart"]) == 0
    assert calls == [("chart", False)]
