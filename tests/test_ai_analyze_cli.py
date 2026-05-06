import json

import pandas as pd
from typer.testing import CliRunner

from quantforge.cli import app


runner = CliRunner()

AI_MOMENTUM_SUCCESS_OUTPUT = """Variant analysis complete: variant_003_momentum_filter

This variant analysis has been saved in the variant folder.
This is a historical backtest, not financial advice.
"""


def write_project(tmp_path, with_baseline: bool = True):
    project_root = tmp_path / "aapl-rsi-reversal"
    project_root.mkdir()
    project_file = project_root / "strategy.qf.json"
    project_file.write_text(
        json.dumps(
            {
                "strategy_id": "rsi_reversal_aapl",
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
            },
            indent=2,
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


def write_prices_csv(path):
    rows = ["date,open,high,low,close,volume"]
    close = 100.0
    for index, date in enumerate(pd.date_range("2020-02-01", periods=45)):
        close += ((index % 7) - 3) * 0.7
        rows.append(
            f"{date.date()},{close - 0.2:.2f},{close + 1:.2f},"
            f"{close - 1:.2f},{close:.2f},1000"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def write_ai_variant(project_file, variant_id, modification_type="momentum_filter"):
    variant_dir = project_file.parent / "variants" / variant_id
    variant_dir.mkdir(parents=True)
    (variant_dir / "strategy_config.json").write_text(
        json.dumps(
            {
                "variant_id": variant_id,
                "parent_variant_id": "baseline",
                "modification_type": modification_type,
                "parameters": {
                    "lookback_days": 3,
                    "threshold": 0.0,
                },
                "entry_rule": "Add entry filter: return_3d > 0",
                "exit_rule": "unchanged",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (variant_dir / "change_summary.json").write_text(
        json.dumps(
            {
                "variant_id": variant_id,
                "parent_variant_id": "baseline",
                "user_instruction": "Add a momentum filter",
                "summary": "AI-assisted modification: momentum_filter",
                "assumptions": ["Momentum is computed using historical close prices only."],
                "warnings": ["This filter may reduce trade count."],
                "status": "created_not_backtested",
                "source": "local_ai_planner",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return variant_dir


def test_baseline_analyze_without_variant_id_unchanged(tmp_path):
    project_file = write_project(tmp_path)

    result = runner.invoke(app, ["analyze", str(project_file)])

    assert result.exit_code == 0
    assert "QuantForge placeholder analysis" in result.output
    assert "baseline_variant_id: baseline" in result.output


def test_ai_momentum_variant_analyze_writes_outputs_and_success_output(tmp_path):
    project_file = write_project(tmp_path)
    data_csv = write_prices_csv(tmp_path / "prices.csv")
    variant_dir = write_ai_variant(project_file, "variant_003_momentum_filter")

    result = runner.invoke(
        app,
        [
            "analyze",
            str(project_file),
            "--data-csv",
            str(data_csv),
            "--variant-id",
            "variant_003_momentum_filter",
        ],
    )

    assert result.exit_code == 0
    assert result.output == AI_MOMENTUM_SUCCESS_OUTPUT
    assert (variant_dir / "backtest_results.csv").exists()
    assert (variant_dir / "signals.csv").exists()
    assert (variant_dir / "metrics.json").exists()
    assert (variant_dir / "report.md").exists()


def test_ai_unsupported_modification_type_errors(tmp_path):
    project_file = write_project(tmp_path)
    data_csv = write_prices_csv(tmp_path / "prices.csv")
    write_ai_variant(
        project_file,
        "variant_003_moving_average_confirmation",
        modification_type="moving_average_confirmation",
    )

    result = runner.invoke(
        app,
        [
            "analyze",
            str(project_file),
            "--data-csv",
            str(data_csv),
            "--variant-id",
            "variant_003_moving_average_confirmation",
        ],
    )

    assert result.exit_code != 0
    assert (
        result.output
        == "ERROR: Backtesting for AI modification type 'moving_average_confirmation' is not supported yet.\n"
    )


def test_ai_variant_requires_data_csv(tmp_path):
    project_file = write_project(tmp_path)

    result = runner.invoke(
        app,
        ["analyze", str(project_file), "--variant-id", "variant_003_momentum_filter"],
    )

    assert result.exit_code != 0
    assert result.output == "ERROR: --data-csv is required when --variant-id is provided.\n"


def test_missing_ai_variant_folder_errors_clearly(tmp_path):
    project_file = write_project(tmp_path)
    data_csv = write_prices_csv(tmp_path / "prices.csv")

    result = runner.invoke(
        app,
        [
            "analyze",
            str(project_file),
            "--data-csv",
            str(data_csv),
            "--variant-id",
            "variant_003_momentum_filter",
        ],
    )

    assert result.exit_code != 0
    assert result.output == "ERROR: Variant variant_003_momentum_filter not found.\n"


def test_missing_ai_strategy_config_errors_clearly(tmp_path):
    project_file = write_project(tmp_path)
    data_csv = write_prices_csv(tmp_path / "prices.csv")
    variant_dir = project_file.parent / "variants" / "variant_003_momentum_filter"
    variant_dir.mkdir(parents=True)

    result = runner.invoke(
        app,
        [
            "analyze",
            str(project_file),
            "--data-csv",
            str(data_csv),
            "--variant-id",
            "variant_003_momentum_filter",
        ],
    )

    assert result.exit_code != 0
    assert result.output == f"Variant config not found: {variant_dir / 'strategy_config.json'}\n"


def test_ai_momentum_overwrite_error_is_surfaced(tmp_path):
    project_file = write_project(tmp_path)
    data_csv = write_prices_csv(tmp_path / "prices.csv")
    variant_dir = write_ai_variant(project_file, "variant_003_momentum_filter")
    (variant_dir / "metrics.json").write_text("{}\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "analyze",
            str(project_file),
            "--data-csv",
            str(data_csv),
            "--variant-id",
            "variant_003_momentum_filter",
        ],
    )

    assert result.exit_code != 0
    assert (
        result.output
        == "ERROR: Variant analysis already exists for variant_003_momentum_filter. Refusing to overwrite.\n"
    )
