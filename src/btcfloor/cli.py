from __future__ import annotations

import argparse
from pathlib import Path

from btcfloor.analysis import run_initial_analysis
from btcfloor.data import (
    download_and_prepare,
    plot_price_diagnostics,
    to_weekly_close,
    validate_price_history,
    write_data_quality_report,
)
from btcfloor.expectile import expectile_model_name, fit_expectile_power_law
from btcfloor.interactive import write_interactive_weekly_floor_chart
from btcfloor.paths import ProjectPaths
from btcfloor.powerlaw import giovanni_power_law_floor_model


CHART_EXPECTILE_TAUS = (0.0001, 0.0005, 0.001, 0.005, 0.01)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="btcfloor",
        description="BTC floor price estimator research commands",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    download = subcommands.add_parser("download", help="Download and normalize BTC data")
    download.add_argument("--force", action="store_true", help="Redownload raw data")

    diagnose = subcommands.add_parser("diagnose", help="Write data quality report and plots")
    diagnose.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload raw data before diagnostics",
    )

    analyze = subcommands.add_parser("analyze", help="Run the initial floor analysis")
    analyze.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload raw data before analysis",
    )

    chart = subcommands.add_parser(
        "chart",
        help="Write standalone interactive weekly candle/floor chart",
    )
    chart.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload raw data before chart generation",
    )
    return parser


def cmd_download(force: bool) -> int:
    paths = ProjectPaths.from_cwd()
    daily = download_and_prepare(paths, force_download=force)
    print(f"Wrote {paths.processed_btc_csv} with {len(daily):,} valid daily rows")
    return 0


def cmd_diagnose(force_download: bool) -> int:
    paths = ProjectPaths.from_cwd()
    daily = download_and_prepare(paths, force_download=force_download)
    quality = validate_price_history(
        paths.raw_btc_csv,
        daily,
        price_fixes_path=paths.price_fixes_csv,
    )
    report = write_data_quality_report(paths, quality)
    figures = plot_price_diagnostics(daily, paths.figure_dir)
    print(f"Wrote {report}")
    for figure in figures:
        print(f"Wrote {figure}")
    return 0


def cmd_analyze(force_download: bool) -> int:
    outputs = run_initial_analysis(force_download=force_download)
    for label, path in outputs.items():
        print(f"{label}: {Path(path)}")
    return 0


def cmd_chart(force_download: bool) -> int:
    paths = ProjectPaths.from_cwd()
    daily = download_and_prepare(paths, force_download=force_download)
    weekly = to_weekly_close(daily)
    models = [
        giovanni_power_law_floor_model(),
        *[
            fit_expectile_power_law(
                weekly,
                tau=tau,
                name=expectile_model_name(tau),
            )
            for tau in CHART_EXPECTILE_TAUS
        ],
    ]
    output = write_interactive_weekly_floor_chart(
        daily,
        models,
        paths.interactive_dir / "btc_floor_weekly.html",
    )
    print(f"interactive_chart: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        return cmd_download(force=args.force)
    if args.command == "diagnose":
        return cmd_diagnose(force_download=args.force_download)
    if args.command == "analyze":
        return cmd_analyze(force_download=args.force_download)
    if args.command == "chart":
        return cmd_chart(force_download=args.force_download)
    parser.error(f"Unknown command: {args.command}")
    return 2
