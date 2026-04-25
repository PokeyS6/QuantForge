"""Command-line interface for QuantForge."""

import typer

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
def create() -> None:
    """Create a placeholder strategy workspace."""
    typer.echo(f"Create placeholder. {NON_ADVISORY_NOTE}")


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
