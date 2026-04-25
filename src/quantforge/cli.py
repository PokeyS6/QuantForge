"""Command-line interface for QuantForge."""

from pathlib import Path

import typer

from quantforge.core.models import new_rsi_project
from quantforge.core.paths import project_dir
from quantforge.core.project_io import write_project

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
def analyze() -> None:
    """Analyze a placeholder baseline."""
    typer.echo(f"Analyze placeholder. {NON_ADVISORY_NOTE}")


@app.command()
def modify() -> None:
    """Modify a placeholder strategy hypothesis."""
    typer.echo(f"Modify placeholder. {NON_ADVISORY_NOTE}")


@app.command()
def compare() -> None:
    """Compare placeholder strategy outputs."""
    typer.echo(f"Compare placeholder. {NON_ADVISORY_NOTE}")
