import json

from typer.testing import CliRunner

from quantforge.cli import app

runner = CliRunner()


SUCCESS_OUTPUT = """Variant 'variant_001_volatility_filter' created.

This variant has not been backtested yet.
Variant backtesting will be available in a subsequent step.
"""

ML_SUCCESS_OUTPUT = """Variant 'variant_002_ml_price_floor' created.

This variant has not been backtested yet.
Variant backtesting will be available in a subsequent step.
"""

VARIANT_ANALYZE_SUCCESS_OUTPUT = """Variant analysis complete: variant_001_volatility_filter

This variant analysis has been saved in the variant folder.
This is a historical backtest, not financial advice.
"""

ML_VARIANT_ANALYZE_SUCCESS_OUTPUT = """Variant analysis complete: variant_002_ml_price_floor

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


def _comparison_metrics(total_return: float = 0.1):
    return {
        "total_return": total_return,
        "annualized_return": 0.2,
        "max_drawdown": -0.3,
        "exposure": 0.4,
        "trade_count": 5,
        "win_rate": 0.6,
    }


def _write_compare_inputs(project_file):
    project_root = project_file.parent
    reports_dir = project_root / "reports"
    baseline_variant_dir = project_root / "variants" / "baseline"
    volatility_variant_dir = project_root / "variants" / "variant_001_volatility_filter"
    ml_variant_dir = project_root / "variants" / "variant_002_ml_price_floor"
    reports_dir.mkdir(parents=True)
    baseline_variant_dir.mkdir(parents=True, exist_ok=True)
    volatility_variant_dir.mkdir(parents=True)
    ml_variant_dir.mkdir(parents=True)
    (reports_dir / "baseline_metrics.json").write_text(
        json.dumps(_comparison_metrics(total_return=0.1)) + "\n",
        encoding="utf-8",
    )
    (baseline_variant_dir / "metrics.json").write_text(
        json.dumps(_comparison_metrics(total_return=0.25)) + "\n",
        encoding="utf-8",
    )
    (volatility_variant_dir / "metrics.json").write_text(
        json.dumps(_comparison_metrics(total_return=0.15)) + "\n",
        encoding="utf-8",
    )
    (ml_variant_dir / "metrics.json").write_text(
        json.dumps(_comparison_metrics(total_return=0.05)) + "\n",
        encoding="utf-8",
    )
    return project_root


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


def test_modify_creates_ml_price_floor_variant(tmp_path):
    project_file = _write_project(tmp_path)

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add an ML price floor"],
    )

    assert result.exit_code == 0
    assert result.output == ML_SUCCESS_OUTPUT
    assert (project_file.parent / "variants" / "variant_002_ml_price_floor").exists()


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
        == "ERROR: Unsupported modification. Only 'volatility filter' and 'ml price floor' are supported.\n"
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


def test_analyze_supported_ml_variant_runs_and_persists_outputs(tmp_path):
    project_file = _write_project(tmp_path)
    data_csv = _write_prices_csv(tmp_path / "prices.csv")
    modify_result = runner.invoke(
        app,
        ["modify", str(project_file), "Add an ML price floor"],
    )

    result = runner.invoke(
        app,
        [
            "analyze",
            str(project_file),
            "--data-csv",
            str(data_csv),
            "--variant-id",
            "variant_002_ml_price_floor",
        ],
    )

    variant_dir = project_file.parent / "variants" / "variant_002_ml_price_floor"
    assert modify_result.exit_code == 0
    assert result.exit_code == 0
    assert result.output == ML_VARIANT_ANALYZE_SUCCESS_OUTPUT
    assert (variant_dir / "metrics.json").exists()
    assert (variant_dir / "report.md").exists()
    assert (variant_dir / "backtest_results.csv").exists()
    assert (variant_dir / "signals.csv").exists()


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
    assert result.output == "ERROR: Variant variant_unknown not found.\n"


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


def test_compare_all_variants_writes_report_and_prints_preview(tmp_path):
    project_file = _write_project(tmp_path)
    project_root = _write_compare_inputs(project_file)

    result = runner.invoke(app, ["compare", str(project_file), "--all-variants"])

    report_path = project_root / "reports" / "comparison_report.md"
    report = report_path.read_text(encoding="utf-8")
    forbidden_words = ["improved", "better", "worse", "optimal", "recommended"]

    assert result.exit_code == 0
    assert report_path.exists()
    assert "## Baseline vs baseline" not in report
    assert "Baseline vs baseline" not in result.output
    assert "## Baseline vs variant_001_volatility_filter" in report
    assert "## Baseline vs variant_002_ml_price_floor" in report
    assert "Baseline vs variant_001_volatility_filter" in result.output
    assert "Baseline vs variant_002_ml_price_floor" in result.output
    assert "Metric | Baseline | Variant | Change" in result.output
    assert result.output.count("| Metric | Baseline | Variant | Change |") >= 2
    assert result.output.count("- ") == 6
    assert "- total_return increased by 0.050000." in result.output
    for forbidden_word in forbidden_words:
        assert forbidden_word not in result.output.lower()
        assert forbidden_word not in report.lower()


def test_compare_without_all_variants_errors(tmp_path):
    project_file = _write_project(tmp_path)

    result = runner.invoke(app, ["compare", str(project_file)])

    assert result.exit_code != 0
    assert result.output == "ERROR: Only --all-variants is supported in this phase.\n"


def test_compare_refuses_to_overwrite_existing_report(tmp_path):
    project_file = _write_project(tmp_path)
    project_root = _write_compare_inputs(project_file)
    report_path = project_root / "reports" / "comparison_report.md"
    report_path.write_text(
        "existing\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["compare", str(project_file), "--all-variants"])

    assert result.exit_code != 0
    assert result.output == "ERROR: Comparison report already exists. Refusing to overwrite.\n"
    assert report_path.read_text(encoding="utf-8") == "existing\n"


def test_compare_missing_baseline_metrics_error_is_surfaced(tmp_path):
    project_file = _write_project(tmp_path)

    result = runner.invoke(app, ["compare", str(project_file), "--all-variants"])

    assert result.exit_code != 0
    assert result.output == "ERROR: Baseline metrics not found. Run 'quantforge analyze' first.\n"


def test_compare_no_backtested_variants_error_is_surfaced(tmp_path):
    project_file = _write_project(tmp_path)
    reports_dir = project_file.parent / "reports"
    reports_dir.mkdir()
    (reports_dir / "baseline_metrics.json").write_text(
        json.dumps(_comparison_metrics()) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["compare", str(project_file), "--all-variants"])

    assert result.exit_code != 0
    assert result.output == "ERROR: No backtested variants available for comparison.\n"
