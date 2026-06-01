from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config_dir: Path
    raw_dir: Path
    processed_dir: Path
    report_dir: Path
    figure_dir: Path
    interactive_dir: Path

    @classmethod
    def from_cwd(cls, cwd: Path | None = None) -> "ProjectPaths":
        root = (cwd or Path.cwd()).resolve()
        report_dir = root / "reports"
        return cls(
            root=root,
            config_dir=root / "config",
            raw_dir=root / "data" / "raw",
            processed_dir=root / "data" / "processed",
            report_dir=report_dir,
            figure_dir=report_dir / "figures",
            interactive_dir=report_dir / "interactive",
        )

    @property
    def raw_btc_csv(self) -> Path:
        return self.raw_dir / "coinmetrics_btc.csv"

    @property
    def price_fixes_csv(self) -> Path:
        return self.config_dir / "price_fixes.csv"

    @property
    def processed_btc_csv(self) -> Path:
        return self.processed_dir / "btc_daily.csv"

    @property
    def data_quality_report(self) -> Path:
        return self.report_dir / "data_quality.md"

    @property
    def initial_analysis_report(self) -> Path:
        return self.report_dir / "initial_analysis.md"

    def ensure_dirs(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.interactive_dir.mkdir(parents=True, exist_ok=True)
