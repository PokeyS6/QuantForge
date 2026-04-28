from typer.testing import CliRunner

from quantforge.cli import app

runner = CliRunner()


SUCCESS_OUTPUT = """Variant 'variant_001_volatility_filter' created.

This variant has not been backtested yet.
Variant backtesting will be available in a subsequent step.
"""


def _write_project(tmp_path, with_baseline: bool = True):
    project_root = tmp_path / "aapl-rsi-reversal"
    project_root.mkdir()
    project_file = project_root / "strategy.qf.json"
    project_file.write_text('{"variants": ["baseline"]}\n', encoding="utf-8")
    if with_baseline:
        baseline_dir = project_root / "variants" / "baseline"
        baseline_dir.mkdir(parents=True)
        (baseline_dir / "backtest_results.csv").write_text(
            "date,close,position,asset_return,strategy_return,equity\n",
            encoding="utf-8",
        )
    return project_file


def test_cli_help():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "create" in result.output
    assert "analyze" in result.output
    assert "modify" in result.output
    assert "compare" in result.output


def test_create_help():
    result = runner.invoke(app, ["create", "--help"])

    assert result.exit_code == 0


def test_analyze_help():
    result = runner.invoke(app, ["analyze", "--help"])

    assert result.exit_code == 0


def test_modify_help():
    result = runner.invoke(app, ["modify", "--help"])

    assert result.exit_code == 0


def test_compare_help():
    result = runner.invoke(app, ["compare", "--help"])

    assert result.exit_code == 0


def test_modify_creates_volatility_filter_variant(tmp_path):
    project_file = _write_project(tmp_path)

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a volatility filter"],
    )

    assert result.exit_code == 0
    assert result.output == SUCCESS_OUTPUT
    assert (
        project_file.parent / "variants" / "variant_001_volatility_filter"
    ).exists()


def test_modify_missing_baseline_backtest_exits_nonzero(tmp_path):
    project_file = _write_project(tmp_path, with_baseline=False)

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a volatility filter"],
    )

    assert result.exit_code != 0
    assert (
        result.output
        == "ERROR: Baseline analysis not found. Run 'quantforge analyze' first.\n"
    )


def test_modify_unsupported_instruction_exits_nonzero(tmp_path):
    project_file = _write_project(tmp_path)

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add moving average filter"],
    )

    assert result.exit_code != 0
    assert (
        result.output
        == "ERROR: Unsupported modification. Only 'volatility filter' is supported.\n"
    )


def test_modify_duplicate_variant_exits_nonzero(tmp_path):
    project_file = _write_project(tmp_path)

    first_result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a volatility filter"],
    )
    second_result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a volatility filter"],
    )

    assert first_result.exit_code == 0
    assert second_result.exit_code != 0
    assert (
        second_result.output
        == "ERROR: Variant variant_001_volatility_filter already exists. Refusing to overwrite.\n"
    )
