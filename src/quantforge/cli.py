"""Command-line interface for QuantForge."""

from pathlib import Path
from typing import Optional

import typer

from quantforge.backtest.engine import run_and_persist_baseline_analysis
from quantforge.core.models import new_rsi_project
from quantforge.core.paths import project_dir
from quantforge.core.project_io import get_baseline_variant, read_project, write_project
from quantforge.variants.mutations import (
    create_volatility_filter_variant,
    run_and_persist_volatility_filter_variant_analysis,
)

app = typer.Typer(
    help=(
        "QuantForge is a local-first strategy workbench. "
        "It helps test user-provided hypotheses; it does not provide trading advice."
    )
)

NON_ADVISORY_NOTE = (
    "QuantForge helps test user-provided hypotheses; it does not provide trading advice. "
    "No strategy recommendation is being made."
)


@app.command()
def create(
    strategy_prompt: str = typer.Argument(..., help="User-provided strategy hypothesis."),
    ticker: str = typer.Option(..., "--ticker", help="Ticker for the initial asset universe."),
    start: str = typer.Option(..., "--start", help="Backtest start date for metadata."),
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        help="Directory where the QuantForge project will be created.",
    ),
) -> None:
    """Create a local strategy project."""
    if "rsi" not in strategy_prompt.lower():
        typer.echo("Only constrained RSI prompts are supported for project creation right now.")
        raise typer.Exit(code=1)

    project = new_rsi_project(strategy_prompt=strategy_prompt, ticker=ticker, start=start)
    project_path = project_dir(output_dir=output_dir, project_name=project.name)
    write_project(project_path=project_path, project=project)
    typer.echo(f"Created QuantForge project: {project_path}")
    typer.echo(NON_ADVISORY_NOTE)


@app.command()
def analyze(
    project_file: Path = typer.Argument(..., help="Path to strategy.qf.json."),
    data_csv: Optional[Path] = typer.Option(
        None,
        "--data-csv",
        help="Optional local OHLCV CSV for baseline RSI analysis.",
    ),
    variant_id: Optional[str] = typer.Option(
        None,
        "--variant-id",
        help="Optional variant id to analyze.",
    ),
) -> None:
    """Read project metadata and print a placeholder analysis."""
    if variant_id is not None:
        if data_csv is None:
            typer.echo("ERROR: --data-csv is required when --variant-id is provided.")
            raise typer.Exit(code=1)
        if variant_id != "variant_001_volatility_filter":
            typer.echo(f"ERROR: Unsupported variant id: {variant_id}")
            raise typer.Exit(code=1)
        try:
            run_and_persist_volatility_filter_variant_analysis(project_file, data_csv)
        except ValueError as error:
            typer.echo(str(error))
            raise typer.Exit(code=1) from error

        typer.echo("Variant analysis complete: variant_001_volatility_filter")
        typer.echo("")
        typer.echo("This variant analysis has been saved in the variant folder.")
        typer.echo("This is a historical backtest, not financial advice.")
        return

    project = read_project(project_file)
    data_window = project.get("data_window", {})

    if data_csv is not None:
        try:
            summary = run_and_persist_baseline_analysis(project, data_csv, project_file)
        except ValueError as error:
            typer.echo(str(error))
            raise typer.Exit(code=1) from error
        baseline = get_baseline_variant(project)
        parameters = baseline.get("parameters", {})

        typer.echo("QuantForge baseline analysis")
        typer.echo(f"strategy_id: {project.get('strategy_id')}")
        typer.echo(f"ticker: {parameters.get('ticker')}")
        typer.echo(f"strategy_type: {baseline.get('strategy_type')}")
        typer.echo(f"total_return: {summary['total_return']:.6f}")
        typer.echo(f"max_drawdown: {summary['max_drawdown']:.6f}")
        typer.echo(f"exposure: {summary['exposure']:.6f}")
        typer.echo(f"trade_count: {summary['trade_count']}")
        typer.echo(NON_ADVISORY_NOTE)
        return

    typer.echo("QuantForge placeholder analysis")
    typer.echo(f"strategy_id: {project.get('strategy_id')}")
    typer.echo(f"name: {project.get('name')}")
    typer.echo(f"asset_universe: {', '.join(project.get('asset_universe', []))}")
    typer.echo(f"timeframe: {project.get('timeframe')}")
    typer.echo(f"data_window.start: {data_window.get('start')}")
    typer.echo(f"data_window.end: {data_window.get('end')}")
    typer.echo(f"baseline_variant_id: {project.get('baseline_variant_id')}")
    typer.echo(f"variants: {len(project.get('variants', []))}")
    typer.echo(NON_ADVISORY_NOTE)


@app.command()
def modify(
    project_file: Path = typer.Argument(..., help="Path to strategy.qf.json."),
    user_instruction: str = typer.Argument(..., help="User-provided modification request."),
) -> None:
    """Create a placeholder strategy variant."""
    try:
        create_volatility_filter_variant(project_file, user_instruction)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error

    typer.echo("Variant 'variant_001_volatility_filter' created.")
    typer.echo("")
    typer.echo("This variant has not been backtested yet.")
    typer.echo("Variant backtesting will be available in a subsequent step.")


@app.command()
def compare() -> None:
    """Compare placeholder strategy outputs."""
    typer.echo(f"Compare placeholder. {NON_ADVISORY_NOTE}")
