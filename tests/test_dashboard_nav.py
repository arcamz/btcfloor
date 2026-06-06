from __future__ import annotations

from btcfloor.dashboard_common import DASHBOARD_LINKS, dashboard_nav


def test_dashboard_nav_contains_all_primary_dashboards() -> None:
    nav = dashboard_nav("btc_gold")

    for item in DASHBOARD_LINKS:
        assert item.href in nav
        assert item.label in nav

    assert 'btc_gold_rotation_dashboard.html" aria-current="page"' in nav
