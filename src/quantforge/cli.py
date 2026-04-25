"""Command-line interface for QuantForge."""

from pathlib import Path

import typer

from quantforge.core.models import new_rsi_project
from quantforge.core.paths import project_dir
from quantforge.core.project_io import read_project, write_project

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
def analyze(project_file: Path = typer.Argument(..., help="Path to strategy.qf.json.")) -> None:
    """Read project metadata and print a placeholder analysis."""
    project = read_project(project_file)
    data_window = project.get("data_window", {})

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
def modify() -> None:
    """Modify a placeholder strategy hypothesis."""
    typer.echo(f"Modify placeholder. {NON_ADVISORY_NOTE}")


@app.command()
def compare() -> None:
    """Compare placeholder strategy outputs."""
    typer.echo(f"Compare placeholder. {NON_ADVISORY_NOTE}")
