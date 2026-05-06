"""Command-line interface for QuantForge."""

import json
from pathlib import Path
from typing import Optional

import typer

from quantforge.ai.artifacts import create_ai_variant_artifacts
from quantforge.ai.backtest import run_and_persist_ai_momentum_variant_analysis
from quantforge.ai.planner import LocalAIPlannerUnavailable, generate_modification_spec
from quantforge.ai.schema import ModificationSpecValidationError, validate_schema
from quantforge.ai.validator import validate_against_registry
from quantforge.backtest.engine import run_and_persist_baseline_analysis
from quantforge.core.models import new_rsi_project
from quantforge.core.paths import project_dir
from quantforge.core.project_io import get_baseline_variant, read_project, write_project
from quantforge.reports.comparison import build_comparison_report
from quantforge.variants.mutations import (
    create_ml_price_floor_variant,
    create_volatility_filter_variant,
    run_and_persist_ml_price_floor_variant_analysis,
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
        if variant_id == "variant_001_volatility_filter":
            analysis_runner = run_and_persist_volatility_filter_variant_analysis
        elif variant_id == "variant_002_ml_price_floor":
            analysis_runner = run_and_persist_ml_price_floor_variant_analysis
        else:
            try:
                analysis_runner = _ai_variant_analysis_runner(project_file, variant_id)
            except ValueError as error:
                typer.echo(str(error))
                raise typer.Exit(code=1) from error
        try:
            if analysis_runner == run_and_persist_ai_momentum_variant_analysis:
                analysis_runner(project_file, data_csv, variant_id)
            else:
                analysis_runner(project_file, data_csv)
        except ValueError as error:
            typer.echo(str(error))
            raise typer.Exit(code=1) from error

        typer.echo(f"Variant analysis complete: {variant_id}")
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


def _ai_variant_analysis_runner(project_file: Path, variant_id: str):
    variant_dir = project_file.parent / "variants" / variant_id
    variant_config_path = variant_dir / "strategy_config.json"
    if not variant_dir.exists():
        raise ValueError(f"ERROR: Variant {variant_id} not found.")
    if not variant_config_path.exists():
        raise ValueError(f"Variant config not found: {variant_config_path}")

    variant_config = json.loads(variant_config_path.read_text(encoding="utf-8"))
    modification_type = variant_config.get("modification_type")
    if modification_type == "momentum_filter":
        return run_and_persist_ai_momentum_variant_analysis

    raise ValueError(
        f"ERROR: Backtesting for AI modification type '{modification_type}' "
        "is not supported yet."
    )


@app.command()
def modify(
    project_file: Path = typer.Argument(..., help="Path to strategy.qf.json."),
    user_instruction: str = typer.Argument(..., help="User-provided modification request."),
    ai: bool = typer.Option(
        False,
        "--ai",
        help="Use the local AI planner to create a validated modification variant.",
    ),
) -> None:
    """Create a placeholder strategy variant."""
    if ai:
        _modify_with_ai(project_file, user_instruction)
        return

    instruction = user_instruction.lower()
    if "volatility filter" in instruction:
        variant_creator = create_volatility_filter_variant
        variant_id = "variant_001_volatility_filter"
    elif "ml price floor" in instruction:
        variant_creator = create_ml_price_floor_variant
        variant_id = "variant_002_ml_price_floor"
    else:
        typer.echo(
            "ERROR: Unsupported modification. Only 'volatility filter' and "
            "'ml price floor' are supported."
        )
        raise typer.Exit(code=1)

    try:
        variant_creator(project_file, user_instruction)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error

    typer.echo(f"Variant '{variant_id}' created.")
    typer.echo("")
    typer.echo("This variant has not been backtested yet.")
    typer.echo("Variant backtesting will be available in a subsequent step.")


def _modify_with_ai(project_file: Path, user_instruction: str) -> None:
    try:
        raw_output = generate_modification_spec(
            user_instruction,
            parent_variant_id="baseline",
        )
    except LocalAIPlannerUnavailable as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error

    try:
        spec = json.loads(raw_output)
        validate_schema(spec)
    except (json.JSONDecodeError, ModificationSpecValidationError) as error:
        _echo_ai_validation_failure()
        raise typer.Exit(code=1) from error

    if spec["supported"] is False:
        typer.echo("ERROR: Unsupported AI modification request.")
        typer.echo(spec["reason"])
        typer.echo("Suggested supported requests:")
        for suggestion in spec["suggested_supported_requests"]:
            typer.echo(f"- {suggestion}")
        raise typer.Exit(code=1)

    try:
        validate_against_registry(spec)
    except ModificationSpecValidationError as error:
        _echo_ai_validation_failure()
        raise typer.Exit(code=1) from error

    try:
        variant_dir = create_ai_variant_artifacts(project_file, spec)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error

    typer.echo(f"Creating AI-assisted variant: {variant_dir.name}")
    typer.echo(f"Modification type: {spec['modification_type']}")
    typer.echo(f"Rule change: {spec['entry_rule_change']}")
    typer.echo("")
    typer.echo("This variant has not been backtested yet.")
    typer.echo("Variant backtesting will be available in a subsequent step.")


def _echo_ai_validation_failure() -> None:
    typer.echo("ERROR: AI modification spec failed validation.")
    typer.echo("No variant was created.")


@app.command()
def compare(
    project_file: Path = typer.Argument(..., help="Path to strategy.qf.json."),
    all_variants: bool = typer.Option(
        False,
        "--all-variants",
        help="Compare all backtested variants against baseline.",
    ),
) -> None:
    """Compare baseline and variant metrics."""
    if not all_variants:
        typer.echo("ERROR: Only --all-variants is supported in this phase.")
        raise typer.Exit(code=1)

    project_root = project_file.parent
    report_path = project_root / "reports" / "comparison_report.md"
    if report_path.exists():
        typer.echo("ERROR: Comparison report already exists. Refusing to overwrite.")
        raise typer.Exit(code=1)

    try:
        report = build_comparison_report(project_root)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    _print_comparison_preview(report)


def _print_comparison_preview(report: str) -> None:
    lines = report.splitlines()
    section_indexes = [
        index for index, line in enumerate(lines) if line.startswith("## Baseline vs ")
    ]
    for section_number, section_start in enumerate(section_indexes):
        section_end = (
            section_indexes[section_number + 1]
            if section_number + 1 < len(section_indexes)
            else len(lines)
        )
        section_lines = lines[section_start:section_end]

        typer.echo(section_lines[0].removeprefix("## "))
        table_start = next(
            index
            for index, line in enumerate(section_lines)
            if line == "| Metric | Baseline | Variant | Change |"
        )
        table_end = table_start
        while table_end < len(section_lines) and section_lines[table_end].startswith("|"):
            typer.echo(section_lines[table_end])
            table_end += 1

        observation_start = next(
            index for index, line in enumerate(section_lines) if line == "### Observations"
        )
        observations = [
            line
            for line in section_lines[observation_start + 1 :]
            if line.startswith("- ")
        ][:3]
        for observation in observations:
            typer.echo(observation)
