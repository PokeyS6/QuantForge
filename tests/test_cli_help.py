import json

from typer.testing import CliRunner

from quantforge.cli import app

runner = CliRunner()


SUCCESS_OUTPUT = """Variant 'variant_001_volatility_filter' created.

This variant has not been backtested yet.
Variant backtesting will be available in a subsequent step.
"""

VARIANT_ANALYZE_SUCCESS_OUTPUT = """Variant analysis complete: variant_001_volatility_filter

This variant analysis has been saved in the variant folder.
This is a historical backtest, not financial advice.
"""


def _write_project(tmp_path, with_baseline: bool = True):
    project_root = tmp_path / "aapl-rsi-reversal"
    project_root.mkdir()
    project_file = project_root / "strategy.qf.json"
    project_file.write_text(
        json.dumps(
            {
                "baseline_variant_id": "baseline",
                "variants": [
                    {
                        "variant_id": "baseline",
                        "strategy_type": "rsi_reversal",
                        "parameters": {
                            "ticker": "AAPL",
                            "entry_rsi": 30,
                            "exit_rsi": 70,
                            "rsi_window": 14,
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if with_baseline:
        baseline_dir = project_root / "variants" / "baseline"
        baseline_dir.mkdir(parents=True)
        (baseline_dir / "backtest_results.csv").write_text(
            "date,close,position,asset_return,strategy_return,equity\n",
            encoding="utf-8",
        )
    return project_file


def _write_prices_csv(path):
    path.write_text(
        "\n".join(
            [
                "date,open,high,low,close,volume",
                "2020-01-01,100,101,99,100,1000",
                "2020-01-02,99,100,98,99,1000",
                "2020-01-03,98,99,97,98,1000",
                "2020-01-04,97,98,96,97,1000",
                "2020-01-05,96,97,95,96,1000",
                "2020-01-06,97,98,96,97,1000",
                "2020-01-07,98,99,97,98,1000",
                "2020-01-08,99,100,98,99,1000",
                "2020-01-09,100,101,99,100,1000",
                "2020-01-10,101,102,100,101,1000",
                "2020-01-11,102,103,101,102,1000",
                "2020-01-12,103,104,102,103,1000",
                "2020-01-13,104,105,103,104,1000",
                "2020-01-14,105,106,104,105,1000",
                "2020-01-15,106,107,105,106,1000",
                "2020-01-16,107,108,106,107,1000",
                "2020-01-17,108,109,107,108,1000",
                "2020-01-18,109,110,108,109,1000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


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


def test_analyze_supported_variant_runs_and_persists_outputs(tmp_path):
    project_file = _write_project(tmp_path)
    data_csv = _write_prices_csv(tmp_path / "prices.csv")
    modify_result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a volatility filter"],
    )

    result = runner.invoke(
        app,
        [
            "analyze",
            str(project_file),
            "--data-csv",
            str(data_csv),
            "--variant-id",
            "variant_001_volatility_filter",
        ],
    )

    variant_dir = project_file.parent / "variants" / "variant_001_volatility_filter"
    assert modify_result.exit_code == 0
    assert result.exit_code == 0
    assert result.output == VARIANT_ANALYZE_SUCCESS_OUTPUT
    assert (variant_dir / "metrics.json").exists()
    assert (variant_dir / "report.md").exists()
    assert (variant_dir / "backtest_results.csv").exists()


def test_analyze_variant_requires_data_csv(tmp_path):
    project_file = _write_project(tmp_path)

    result = runner.invoke(
        app,
        [
            "analyze",
            str(project_file),
            "--variant-id",
            "variant_001_volatility_filter",
        ],
    )

    assert result.exit_code != 0
    assert result.output == "ERROR: --data-csv is required when --variant-id is provided.\n"


def test_analyze_unsupported_variant_id_exits_nonzero(tmp_path):
    project_file = _write_project(tmp_path)
    data_csv = _write_prices_csv(tmp_path / "prices.csv")

    result = runner.invoke(
        app,
        [
            "analyze",
            str(project_file),
            "--data-csv",
            str(data_csv),
            "--variant-id",
            "variant_unknown",
        ],
    )

    assert result.exit_code != 0
    assert result.output == "ERROR: Unsupported variant id: variant_unknown\n"


def test_analyze_variant_helper_value_error_is_surfaced(tmp_path):
    project_file = _write_project(tmp_path)
    data_csv = _write_prices_csv(tmp_path / "prices.csv")

    result = runner.invoke(
        app,
        [
            "analyze",
            str(project_file),
            "--data-csv",
            str(data_csv),
            "--variant-id",
            "variant_001_volatility_filter",
        ],
    )

    assert result.exit_code != 0
    assert result.output == "ERROR: Variant variant_001_volatility_filter not found.\n"


def test_analyze_without_variant_id_keeps_baseline_behavior(tmp_path):
    project_file = _write_project(tmp_path)

    result = runner.invoke(app, ["analyze", str(project_file)])

    assert result.exit_code == 0
    assert "QuantForge placeholder analysis" in result.output
