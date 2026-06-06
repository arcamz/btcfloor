from __future__ import annotations

from pathlib import Path

from btcfloor.analysis import run_initial_analysis
from btcfloor.paths import ProjectPaths
from build_interactive_dashboards import build_dashboards
from build_metals_dashboard import build_metals_dashboard
from build_pipeline_health_dashboard import build_pipeline_health_dashboard
import plot_checkonchain_cohorts
from plot_floor_convergence_decision import write_plot as write_floor_convergence
from plot_sma_channel_decision import write_plot as write_sma_channel
from plot_tactical_trigger_strip import write_plot as write_tactical_strip


def _print_outputs(title: str, outputs: dict[str, Path] | list[Path]) -> None:
    print(title)
    items = outputs.items() if isinstance(outputs, dict) else enumerate(outputs)
    for key, path in items:
        print(f"  {key}: {Path(path)}")


def main() -> None:
    paths = ProjectPaths.from_cwd()
    paths.ensure_dirs()

    analysis_outputs = run_initial_analysis(force_download=True)
    _print_outputs("core analysis", analysis_outputs)

    write_floor_convergence(paths)
    write_sma_channel(paths)
    write_tactical_strip(paths)
    tactical_outputs = [
        paths.figure_dir / "floor_convergence_decision_dashboard.png",
        paths.figure_dir / "sma_channel_decision_plot.png",
        paths.figure_dir / "tactical_trigger_strip.png",
    ]
    _print_outputs("tactical plots", tactical_outputs)

    plot_checkonchain_cohorts.main()
    checkonchain_outputs = [
        paths.processed_dir / "checkonchain_cohort_metrics.csv",
        paths.report_dir / "checkonchain_cohort_summary.json",
        paths.figure_dir / "checkonchain_cohort_current_bands.png",
        paths.figure_dir / "checkonchain_cohort_cycle_lows.png",
        paths.figure_dir / "checkonchain_lth_realised_loss_cycle.png",
        paths.figure_dir / "checkonchain_low_signal_compare.png",
    ]
    _print_outputs("checkonchain cohorts", checkonchain_outputs)

    dashboard_outputs = build_dashboards(paths)
    _print_outputs("interactive dashboards", dashboard_outputs)

    metals_outputs = build_metals_dashboard(paths)
    _print_outputs("metals dashboard", metals_outputs)

    health_outputs = build_pipeline_health_dashboard(paths)
    _print_outputs("pipeline health", health_outputs)


if __name__ == "__main__":
    main()
