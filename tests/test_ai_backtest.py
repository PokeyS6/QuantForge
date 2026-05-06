import json

import pandas as pd
import pytest

from quantforge.ai.backtest import run_and_persist_ai_momentum_variant_analysis


METRIC_KEYS = [
    "total_return",
    "annualized_return",
    "max_drawdown",
    "exposure",
    "trade_count",
    "win_rate",
]
RESULT_COLUMNS = ["date", "close", "position", "asset_return", "strategy_return", "equity"]
SIGNAL_COLUMNS = [
    "date",
    "close",
    "rsi",
    "entry_signal",
    "exit_signal",
    "momentum_return",
    "momentum_entry_allowed",
]


@pytest.fixture
def ai_momentum_project(tmp_path):
    project_file = write_project(tmp_path)
    baseline_results_path = write_baseline_results(project_file.parent)
    variant_dir = write_momentum_variant(project_file.parent, "variant_003_momentum_filter")
    data_csv = write_prices_csv(tmp_path / "prices.csv")
    return {
        "project_file": project_file,
        "baseline_results_path": baseline_results_path,
        "variant_dir": variant_dir,
        "data_csv": data_csv,
    }


def write_project(tmp_path, variants=None):
    variants = [
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
    ] if variants is None else variants
    project_root = tmp_path / "aapl-rsi-reversal"
    project_root.mkdir()
    project_file = project_root / "strategy.qf.json"
    project_file.write_text(
        json.dumps(
            {
                "strategy_id": "rsi_reversal_aapl",
                "baseline_variant_id": "baseline",
                "variants": variants,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return project_file


def write_baseline_results(project_root):
    baseline_dir = project_root / "variants" / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_results_path = baseline_dir / "backtest_results.csv"
    baseline_results_path.write_text(
        "date,close,position,asset_return,strategy_return,equity\n"
        "2020-01-01,100,0,0,0,1\n",
        encoding="utf-8",
    )
    return baseline_results_path


def write_momentum_variant(project_root, variant_id, modification_type="momentum_filter"):
    variant_dir = project_root / "variants" / variant_id
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
                "assumptions": [
                    "Momentum is computed using historical close prices only.",
                    "The filter affects entries only.",
                ],
                "warnings": [
                    "This filter may reduce trade count.",
                    "Historical behavior does not imply future performance.",
                ],
                "status": "created_not_backtested",
                "source": "local_ai_planner",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return variant_dir


def write_prices_csv(path):
    rows = ["date,open,high,low,close,volume"]
    close = 100.0
    for index, date in enumerate(pd.date_range("2020-01-01", periods=45)):
        close += ((index % 7) - 3) * 0.7
        open_price = close - 0.2
        high = close + 1.0
        low = close - 1.0
        rows.append(
            f"{date.date()},{open_price:.2f},{high:.2f},{low:.2f},{close:.2f},1000"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def output_paths(variant_dir):
    return [
        variant_dir / "backtest_results.csv",
        variant_dir / "signals.csv",
        variant_dir / "metrics.json",
        variant_dir / "report.md",
    ]


def assert_no_outputs(variant_dir):
    assert not any(path.exists() for path in output_paths(variant_dir))


def test_successful_momentum_backtest_writes_all_outputs(ai_momentum_project):
    variant_dir = ai_momentum_project["variant_dir"]

    metrics = run_and_persist_ai_momentum_variant_analysis(
        ai_momentum_project["project_file"],
        ai_momentum_project["data_csv"],
        variant_dir.name,
    )

    assert list(metrics.keys()) == METRIC_KEYS
    for path in output_paths(variant_dir):
        assert path.exists()


def test_backtest_results_csv_schema_exact(ai_momentum_project):
    variant_dir = ai_momentum_project["variant_dir"]

    run_and_persist_ai_momentum_variant_analysis(
        ai_momentum_project["project_file"],
        ai_momentum_project["data_csv"],
        variant_dir.name,
    )

    results = pd.read_csv(variant_dir / "backtest_results.csv")
    assert list(results.columns) == RESULT_COLUMNS


def test_signals_csv_schema_exact_with_momentum_columns(ai_momentum_project):
    variant_dir = ai_momentum_project["variant_dir"]

    run_and_persist_ai_momentum_variant_analysis(
        ai_momentum_project["project_file"],
        ai_momentum_project["data_csv"],
        variant_dir.name,
    )

    signals = pd.read_csv(variant_dir / "signals.csv")
    assert list(signals.columns) == SIGNAL_COLUMNS


def test_signals_csv_has_no_nans(ai_momentum_project):
    variant_dir = ai_momentum_project["variant_dir"]

    run_and_persist_ai_momentum_variant_analysis(
        ai_momentum_project["project_file"],
        ai_momentum_project["data_csv"],
        variant_dir.name,
    )

    signals = pd.read_csv(variant_dir / "signals.csv")
    assert not signals.isna().any().any()


def test_metrics_json_schema_exact(ai_momentum_project):
    variant_dir = ai_momentum_project["variant_dir"]

    run_and_persist_ai_momentum_variant_analysis(
        ai_momentum_project["project_file"],
        ai_momentum_project["data_csv"],
        variant_dir.name,
    )

    assert list(read_json(variant_dir / "metrics.json").keys()) == METRIC_KEYS


def test_report_contains_ai_momentum_sections_and_parameters(ai_momentum_project):
    variant_dir = ai_momentum_project["variant_dir"]

    run_and_persist_ai_momentum_variant_analysis(
        ai_momentum_project["project_file"],
        ai_momentum_project["data_csv"],
        variant_dir.name,
    )

    report = (variant_dir / "report.md").read_text(encoding="utf-8")
    assert "## AI-Assisted Modification" in report
    assert "The AI did not generate executable strategy code." in report
    assert "## Momentum Filter" in report
    assert "- lookback_days: 3" in report
    assert "- threshold: 0.0" in report


def test_report_contains_assumptions_and_warnings(ai_momentum_project):
    variant_dir = ai_momentum_project["variant_dir"]

    run_and_persist_ai_momentum_variant_analysis(
        ai_momentum_project["project_file"],
        ai_momentum_project["data_csv"],
        variant_dir.name,
    )

    report = (variant_dir / "report.md").read_text(encoding="utf-8")
    assert "## Assumptions" in report
    assert "- Momentum is computed using historical close prices only." in report
    assert "## Warnings" in report
    assert "- This filter may reduce trade count." in report
    assert "- Momentum behavior may differ across market regimes." in report


def test_change_summary_status_becomes_backtested(ai_momentum_project):
    variant_dir = ai_momentum_project["variant_dir"]

    run_and_persist_ai_momentum_variant_analysis(
        ai_momentum_project["project_file"],
        ai_momentum_project["data_csv"],
        variant_dir.name,
    )

    assert read_json(variant_dir / "change_summary.json")["status"] == "backtested"


def test_missing_baseline_raises_and_creates_no_outputs(ai_momentum_project):
    ai_momentum_project["baseline_results_path"].unlink()
    variant_dir = ai_momentum_project["variant_dir"]

    with pytest.raises(
        ValueError,
        match="ERROR: Baseline analysis not found. Run 'quantforge analyze' first.",
    ):
        run_and_persist_ai_momentum_variant_analysis(
            ai_momentum_project["project_file"],
            ai_momentum_project["data_csv"],
            variant_dir.name,
        )

    assert_no_outputs(variant_dir)


def test_missing_variant_raises(tmp_path):
    project_file = write_project(tmp_path)
    write_baseline_results(project_file.parent)
    data_csv = write_prices_csv(tmp_path / "prices.csv")

    with pytest.raises(ValueError, match="ERROR: Variant variant_missing not found."):
        run_and_persist_ai_momentum_variant_analysis(
            project_file,
            data_csv,
            "variant_missing",
        )


def test_unsupported_modification_type_raises_correct_error(ai_momentum_project):
    variant_dir = ai_momentum_project["variant_dir"]
    config_path = variant_dir / "strategy_config.json"
    config = read_json(config_path)
    config["modification_type"] = "moving_average_confirmation"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            "ERROR: Backtesting for AI modification type "
            "'moving_average_confirmation' is not supported yet."
        ),
    ):
        run_and_persist_ai_momentum_variant_analysis(
            ai_momentum_project["project_file"],
            ai_momentum_project["data_csv"],
            variant_dir.name,
        )

    assert_no_outputs(variant_dir)


def test_missing_lookback_days_raises_correct_error(ai_momentum_project):
    variant_dir = ai_momentum_project["variant_dir"]
    config_path = variant_dir / "strategy_config.json"
    config = read_json(config_path)
    del config["parameters"]["lookback_days"]
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="ERROR: Missing required parameter 'lookback_days' for momentum_filter",
    ):
        run_and_persist_ai_momentum_variant_analysis(
            ai_momentum_project["project_file"],
            ai_momentum_project["data_csv"],
            variant_dir.name,
        )

    assert_no_outputs(variant_dir)


def test_missing_threshold_raises_correct_error(ai_momentum_project):
    variant_dir = ai_momentum_project["variant_dir"]
    config_path = variant_dir / "strategy_config.json"
    config = read_json(config_path)
    del config["parameters"]["threshold"]
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="ERROR: Missing required parameter 'threshold' for momentum_filter",
    ):
        run_and_persist_ai_momentum_variant_analysis(
            ai_momentum_project["project_file"],
            ai_momentum_project["data_csv"],
            variant_dir.name,
        )

    assert_no_outputs(variant_dir)


def test_overwrite_protection_raises(ai_momentum_project):
    variant_dir = ai_momentum_project["variant_dir"]
    (variant_dir / "metrics.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            "ERROR: Variant analysis already exists for "
            "variant_003_momentum_filter. Refusing to overwrite."
        ),
    ):
        run_and_persist_ai_momentum_variant_analysis(
            ai_momentum_project["project_file"],
            ai_momentum_project["data_csv"],
            variant_dir.name,
        )


def test_baseline_outputs_are_not_modified(ai_momentum_project):
    baseline_results_path = ai_momentum_project["baseline_results_path"]
    original_baseline = baseline_results_path.read_text(encoding="utf-8")

    run_and_persist_ai_momentum_variant_analysis(
        ai_momentum_project["project_file"],
        ai_momentum_project["data_csv"],
        ai_momentum_project["variant_dir"].name,
    )

    assert baseline_results_path.read_text(encoding="utf-8") == original_baseline
