from __future__ import annotations

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True)
class DashboardLink:
    key: str
    label: str
    href: str


DASHBOARD_LINKS = (
    DashboardLink("market", "BTC market", "btc_market_dashboard.html"),
    DashboardLink("floors", "BTC floors", "btc_floor_weekly.html"),
    DashboardLink("roi", "BTC ROI", "btc_roi_dashboard.html"),
    DashboardLink("btc_gold", "BTC/gold", "btc_gold_rotation_dashboard.html"),
    DashboardLink("metals", "Metals/GSR", "metals_relative_dashboard.html"),
    DashboardLink("health", "Data health", "pipeline_health_dashboard.html"),
)


def dashboard_nav(active_key: str) -> str:
    links = []
    for item in DASHBOARD_LINKS:
        current = ' aria-current="page"' if item.key == active_key else ""
        links.append(
            f'<a href="{escape(item.href)}"{current}>{escape(item.label)}</a>'
        )
    return '<nav class="dashboard-nav">' + "\n      ".join(links) + "</nav>"


def dashboard_nav_css() -> str:
    return """
    .dashboard-nav {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      justify-content: flex-end;
      align-items: center;
    }
    .dashboard-nav a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
      border-bottom: 1px solid var(--accent);
      white-space: nowrap;
    }
    .dashboard-nav a[aria-current="page"] {
      color: var(--ink);
      border-bottom-color: var(--ink);
    }
    @media (max-width: 900px) {
      .dashboard-nav {
        justify-content: flex-start;
      }
    }
    """
