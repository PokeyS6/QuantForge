import json

import pandas as pd
import pytest

from quantforge.backtest.engine import run_and_persist_baseline_analysis
from quantforge.core.project_io import read_project
from quantforge.reports.comparison import (
    build_comparison_report,
    build_metrics_comparison_table,
    generate_comparison_observations,
)
from quantforge.variants.mutations import (
    apply_volatility_filter_to_signals,
    create_volatility_filter_variant,
    run_and_persist_volatility_filter_variant_analysis,
)


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


def _comparison_metrics(total_return: float = 0.1) -> dict:
    return {
        "total_return": total_return,
        "annualized_return": 0.2,
        "max_drawdown": -0.3,
        "exposure": 0.4,
        "trade_count": 5,
        "win_rate": 0.6,
    }


@pytest.fixture()
def baseline_project(tmp_path):
    project_root = tmp_path / "aapl-rsi-reversal"
    variants_dir = project_root / "variants" / "baseline"
    variants_dir.mkdir(parents=True)
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
            }
        )
        + "\n",
        encoding="utf-8",
    )
    data_csv = tmp_path / "prices.csv"
    _write_prices_csv(data_csv)
    return project_file, data_csv


@pytest.fixture()
def analyzed_project(tmp_path):
    project_root = tmp_path / "aapl-rsi-reversal"
    baseline_dir = project_root / "variants" / "baseline"
    baseline_dir.mkdir(parents=True)
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
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (baseline_dir / "backtest_results.csv").write_text(
        "date,close,position,asset_return,strategy_return,equity\n",
        encoding="utf-8",
    )
    return project_file


def test_run_and_persist_baseline_analysis_writes_files_and_exact_metrics_schema(baseline_project):
    project_file, data_csv = baseline_project
    project = read_project(project_file)

    run_and_persist_baseline_analysis(project, data_csv, project_file)

    metrics_path = project_file.parent / "reports" / "baseline_metrics.json"
    report_path = project_file.parent / "reports" / "baseline_report.md"
    results_path = project_file.parent / "variants" / "baseline" / "backtest_results.csv"
    signals_path = project_file.parent / "variants" / "baseline" / "signals.csv"

    assert metrics_path.exists()
    assert report_path.exists()
    assert results_path.exists()
    assert signals_path.exists()
    assert list(json.loads(metrics_path.read_text(encoding="utf-8"))) == [
        "total_return",
        "annualized_return",
        "max_drawdown",
        "exposure",
        "trade_count",
        "win_rate",
    ]


def test_run_and_persist_baseline_analysis_refuses_to_overwrite(baseline_project):
    project_file, data_csv = baseline_project
    project = read_project(project_file)

    run_and_persist_baseline_analysis(project, data_csv, project_file)

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        run_and_persist_baseline_analysis(project, data_csv, project_file)


def test_run_and_persist_baseline_analysis_csv_outputs_have_expected_columns_and_no_nans(
    baseline_project,
):
    project_file, data_csv = baseline_project
    project = read_project(project_file)

    run_and_persist_baseline_analysis(project, data_csv, project_file)

    results = pd.read_csv(project_file.parent / "variants" / "baseline" / "backtest_results.csv")
    signals = pd.read_csv(project_file.parent / "variants" / "baseline" / "signals.csv")

    assert list(results.columns) == [
        "date",
        "close",
        "position",
        "asset_return",
        "strategy_return",
        "equity",
    ]
    assert list(signals.columns) == ["date", "close", "rsi", "entry_signal", "exit_signal"]
    assert not results.isna().any().any()
    assert not signals.isna().any().any()


def test_run_and_persist_baseline_analysis_writes_required_markdown_report(
    baseline_project,
):
    project_file, data_csv = baseline_project
    project = read_project(project_file)

    run_and_persist_baseline_analysis(project, data_csv, project_file)

    report = (project_file.parent / "reports" / "baseline_report.md").read_text(
        encoding="utf-8"
    )

    assert "# Strategy Report — Baseline" in report
    assert "## Strategy Summary" in report
    assert "## Strategy Code" in report
    assert "## Assumptions" in report
    assert "## Metrics" in report
    assert "## Regime Analysis (Volatility)" in report
    assert "### Regime Metrics" in report
    assert "### Observation" in report
    assert "## Diagnostics" in report
    assert "## Interpretation (Non-Advisory)" in report
    assert report.index("## Metrics") < report.index("## Regime Analysis (Volatility)")
    assert report.index("## Regime Analysis (Volatility)") < report.index("## Diagnostics")
    assert "| Regime | Avg Daily Return | Exposure | Trade Count |" in report
    assert "| High Volatility |" in report
    assert "| Normal Volatility |" in report
    assert (
        "The high-volatility subset contains too few trades for meaningful comparison."
        in report
    )
    assert "- Low trade count:" in report
    assert "- High drawdown:" in report
    assert "- Slippage not modeled: Slippage is not modeled; results may be optimistic" in report
    assert "- Data source: user-provided CSV" in report
    assert "- Ticker: AAPL" in report
    assert "- Period: 2020-01-01 → 2020-01-18" in report
    assert "This analysis does not constitute financial advice." in report

    report_files = {path.name for path in (project_file.parent / "reports").iterdir()}
    baseline_files = {
        path.name for path in (project_file.parent / "variants" / "baseline").iterdir()
    }
    assert report_files == {"baseline_metrics.json", "baseline_report.md"}
    assert baseline_files == {"backtest_results.csv", "signals.csv"}


def test_create_volatility_filter_variant_writes_expected_artifacts(analyzed_project):
    variant_dir = create_volatility_filter_variant(
        analyzed_project,
        "Add a volatility filter",
    )

    strategy_config = json.loads(
        (variant_dir / "strategy_config.json").read_text(encoding="utf-8")
    )
    change_summary = json.loads(
        (variant_dir / "change_summary.json").read_text(encoding="utf-8")
    )
    diff = (variant_dir / "diff.md").read_text(encoding="utf-8")

    assert strategy_config == {
        "variant_id": "variant_001_volatility_filter",
        "parent_variant_id": "baseline",
        "modification_type": "volatility_filter",
        "volatility_window": 30,
        "volatility_threshold_quantile": 0.75,
        "entry_rule": "RSI < 30 AND rolling volatility <= threshold",
        "exit_rule": "RSI > 70",
    }
    assert change_summary == {
        "variant_id": "variant_001_volatility_filter",
        "parent_variant_id": "baseline",
        "user_instruction": "Add a volatility filter",
        "summary": "Adds a volatility filter that blocks RSI entries during unusually high-volatility periods.",
        "rationale": "Phase 2 diagnostics showed strategy behavior differs under high-volatility conditions.",
        "assumptions": [
            "Volatility is measured as the 30-day rolling standard deviation of daily returns.",
            "High volatility is defined as rolling volatility above the 75th percentile.",
            "The filter affects entries only; exits remain unchanged.",
        ],
        "status": "created_not_backtested",
    }
    assert diff == """# Variant Diff — variant_001_volatility_filter

Parent: baseline

## Summary

Adds a volatility filter to the baseline RSI entry rule.

## Baseline Entry Rule

Enter long when:

RSI < 30

## Variant Entry Rule

Enter long when:

RSI < 30  
AND rolling 30-day volatility <= 75th percentile volatility threshold

## Exit Rule

Unchanged:

RSI > 70

## Implementation Note

Strategy logic is currently implemented within the QuantForge analysis pipeline.
No separate strategy code file exists at this stage.

## Backtest Status

This variant has been created but not backtested yet.
"""


def test_create_volatility_filter_variant_missing_baseline_backtest_raises(tmp_path):
    project_root = tmp_path / "aapl-rsi-reversal"
    project_root.mkdir()
    project_file = project_root / "strategy.qf.json"
    project_file.write_text('{"variants": []}\n', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="ERROR: Baseline analysis not found. Run 'quantforge analyze' first.",
    ):
        create_volatility_filter_variant(project_file, "Add a volatility filter")


def test_create_volatility_filter_variant_duplicate_raises(analyzed_project):
    create_volatility_filter_variant(analyzed_project, "Add a volatility filter")

    with pytest.raises(
        ValueError,
        match="ERROR: Variant variant_001_volatility_filter already exists. Refusing to overwrite.",
    ):
        create_volatility_filter_variant(analyzed_project, "Add a volatility filter")


def test_create_volatility_filter_variant_unsupported_modification_raises(
    analyzed_project,
):
    with pytest.raises(
        ValueError,
        match="ERROR: Unsupported modification. Only 'volatility filter' is supported.",
    ):
        create_volatility_filter_variant(analyzed_project, "Add a moving average filter")


def test_create_volatility_filter_variant_updates_metadata_variants_list(
    analyzed_project,
):
    create_volatility_filter_variant(analyzed_project, "Add a volatility filter")

    metadata = json.loads(analyzed_project.read_text(encoding="utf-8"))

    assert metadata["variants"][-1] == "variant_001_volatility_filter"


def test_create_volatility_filter_variant_preserves_existing_variant_ids(
    analyzed_project,
):
    metadata = json.loads(analyzed_project.read_text(encoding="utf-8"))
    metadata["variants"] = ["baseline", "variant_existing"]
    analyzed_project.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    create_volatility_filter_variant(analyzed_project, "Add a volatility filter")

    updated_metadata = json.loads(analyzed_project.read_text(encoding="utf-8"))

    assert updated_metadata["variants"] == [
        "baseline",
        "variant_existing",
        "variant_001_volatility_filter",
    ]


def test_apply_volatility_filter_to_signals_uses_expanding_threshold_without_lookahead():
    index = pd.date_range("2020-01-01", periods=7)
    prices = pd.DataFrame(
        {"close": [100, 102, 106, 107, 160, 90, 91]},
        index=index,
    )
    signals = pd.DataFrame(
        {
            "close": prices["close"],
            "rsi": [20, 20, 20, 20, 20, 20, 20],
            "entry_signal": [True, True, True, True, True, True, True],
            "exit_signal": [False, True, False, False, True, False, False],
        },
        index=index,
    )
    volatility = prices["close"].pct_change().rolling(2).std()
    threshold = volatility.expanding().quantile(0.5)
    expected_entries = signals["entry_signal"] & volatility.notna() & threshold.notna() & (
        volatility <= threshold
    )

    filtered = apply_volatility_filter_to_signals(
        prices,
        signals,
        volatility_window=2,
        volatility_threshold_quantile=0.5,
    )

    assert list(filtered.columns) == ["close", "rsi", "entry_signal", "exit_signal"]
    assert filtered.index.equals(signals.index)
    assert filtered["entry_signal"].tolist() == expected_entries.tolist()
    assert filtered["exit_signal"].tolist() == signals["exit_signal"].tolist()
    assert filtered["entry_signal"].iloc[:2].tolist() == [False, False]


def test_apply_volatility_filter_to_signals_invalid_inputs_raise():
    prices = pd.DataFrame({"close": [100, 101, 102]})
    signals = pd.DataFrame(
        {
            "close": [100, 101, 102],
            "rsi": [20, 21, 22],
            "entry_signal": [True, True, True],
            "exit_signal": [False, False, False],
        }
    )

    with pytest.raises(ValueError, match="Prices must include a close column."):
        apply_volatility_filter_to_signals(
            prices.drop(columns=["close"]),
            signals,
            volatility_window=2,
            volatility_threshold_quantile=0.5,
        )
    with pytest.raises(
        ValueError,
        match="Signals must include close, rsi, entry_signal, and exit_signal columns.",
    ):
        apply_volatility_filter_to_signals(
            prices,
            signals.drop(columns=["rsi"]),
            volatility_window=2,
            volatility_threshold_quantile=0.5,
        )
    with pytest.raises(ValueError, match="volatility_window must be greater than 0."):
        apply_volatility_filter_to_signals(
            prices,
            signals,
            volatility_window=0,
            volatility_threshold_quantile=0.5,
        )
    with pytest.raises(
        ValueError,
        match="volatility_threshold_quantile must be between 0 and 1.",
    ):
        apply_volatility_filter_to_signals(
            prices,
            signals,
            volatility_window=2,
            volatility_threshold_quantile=1.1,
        )


def test_run_and_persist_volatility_filter_variant_analysis_missing_baseline_raises(
    tmp_path,
):
    project_root = tmp_path / "aapl-rsi-reversal"
    project_root.mkdir()
    project_file = project_root / "strategy.qf.json"
    project_file.write_text('{"variants": ["baseline"]}\n', encoding="utf-8")
    data_csv = tmp_path / "prices.csv"
    _write_prices_csv(data_csv)

    with pytest.raises(
        ValueError,
        match="ERROR: Baseline analysis not found. Run 'quantforge analyze' first.",
    ):
        run_and_persist_volatility_filter_variant_analysis(project_file, data_csv)


def test_run_and_persist_volatility_filter_variant_analysis_missing_variant_raises(
    analyzed_project,
    tmp_path,
):
    data_csv = tmp_path / "prices.csv"
    _write_prices_csv(data_csv)

    with pytest.raises(
        ValueError,
        match="ERROR: Variant variant_001_volatility_filter not found.",
    ):
        run_and_persist_volatility_filter_variant_analysis(analyzed_project, data_csv)


def test_run_and_persist_volatility_filter_variant_analysis_writes_outputs(
    analyzed_project,
    tmp_path,
):
    data_csv = tmp_path / "prices.csv"
    _write_prices_csv(data_csv)
    create_volatility_filter_variant(analyzed_project, "Add a volatility filter")
    project_root = analyzed_project.parent
    baseline_results_path = project_root / "variants" / "baseline" / "backtest_results.csv"
    original_baseline_results = baseline_results_path.read_text(encoding="utf-8")

    metrics = run_and_persist_volatility_filter_variant_analysis(
        analyzed_project,
        data_csv,
    )

    variant_dir = project_root / "variants" / "variant_001_volatility_filter"
    results = pd.read_csv(variant_dir / "backtest_results.csv")
    signals = pd.read_csv(variant_dir / "signals.csv")
    metrics_json = json.loads((variant_dir / "metrics.json").read_text(encoding="utf-8"))
    change_summary = json.loads(
        (variant_dir / "change_summary.json").read_text(encoding="utf-8")
    )
    report = (variant_dir / "report.md").read_text(encoding="utf-8")

    assert list(metrics_json) == [
        "total_return",
        "annualized_return",
        "max_drawdown",
        "exposure",
        "trade_count",
        "win_rate",
    ]
    assert metrics == metrics_json
    assert list(results.columns) == [
        "date",
        "close",
        "position",
        "asset_return",
        "strategy_return",
        "equity",
    ]
    assert list(signals.columns) == ["date", "close", "rsi", "entry_signal", "exit_signal"]
    assert not results.isna().any().any()
    assert not signals.isna().any().any()
    assert change_summary["status"] == "backtested"
    assert "# Strategy Report — Variant" in report
    assert "## Variant Summary" in report
    assert "## Change Summary" in report
    assert "## Strategy Code" in report
    assert "## Assumptions" in report
    assert "## Metrics" in report
    assert "## Regime Analysis (Volatility)" in report
    assert "## Diagnostics" in report
    assert "## Interpretation (Non-Advisory)" in report
    assert (
        "The volatility threshold is computed using only historical data available up to each point in time."
        in report
    )
    assert (
        "The volatility filter removed all entry signals; the strategy did not take any positions."
        in report
    )
    assert baseline_results_path.read_text(encoding="utf-8") == original_baseline_results


def test_run_and_persist_volatility_filter_variant_analysis_refuses_overwrite(
    analyzed_project,
    tmp_path,
):
    data_csv = tmp_path / "prices.csv"
    _write_prices_csv(data_csv)
    create_volatility_filter_variant(analyzed_project, "Add a volatility filter")

    run_and_persist_volatility_filter_variant_analysis(analyzed_project, data_csv)

    with pytest.raises(
        ValueError,
        match=(
            "ERROR: Variant analysis already exists for variant_001_volatility_filter. "
            "Refusing to overwrite."
        ),
    ):
        run_and_persist_volatility_filter_variant_analysis(analyzed_project, data_csv)


def test_run_and_persist_volatility_filter_variant_analysis_missing_baseline_parameters_raises(
    analyzed_project,
    tmp_path,
):
    metadata = json.loads(analyzed_project.read_text(encoding="utf-8"))
    metadata["variants"] = ["baseline", "variant_001_volatility_filter"]
    analyzed_project.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    data_csv = tmp_path / "prices.csv"
    _write_prices_csv(data_csv)
    create_volatility_filter_variant(analyzed_project, "Add a volatility filter")

    with pytest.raises(
        ValueError,
        match="Baseline RSI parameters not found in project metadata.",
    ):
        run_and_persist_volatility_filter_variant_analysis(analyzed_project, data_csv)


def test_run_and_persist_volatility_filter_variant_analysis_missing_config_field_raises(
    analyzed_project,
    tmp_path,
):
    data_csv = tmp_path / "prices.csv"
    _write_prices_csv(data_csv)
    variant_dir = create_volatility_filter_variant(
        analyzed_project,
        "Add a volatility filter",
    )
    config_path = variant_dir / "strategy_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["volatility_window"]
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Variant config missing required field: volatility_window",
    ):
        run_and_persist_volatility_filter_variant_analysis(analyzed_project, data_csv)


def test_build_metrics_comparison_table_formats_absolute_changes():
    baseline_metrics = _comparison_metrics(total_return=0.1)
    variant_metrics = _comparison_metrics(total_return=0.15)

    table = build_metrics_comparison_table(baseline_metrics, variant_metrics)

    assert "| Metric | Baseline | Variant | Change |" in table
    assert "|--------|----------|---------|--------|" in table
    assert "| total_return | 0.100000 | 0.150000 | 0.050000 |" in table
    assert "%" not in table


def test_generate_comparison_observations_avoid_forbidden_words():
    baseline_metrics = _comparison_metrics(total_return=0.1)
    variant_metrics = _comparison_metrics(total_return=0.05)

    observations = generate_comparison_observations(baseline_metrics, variant_metrics)
    observation_text = " ".join(observations).lower()

    assert "total_return decreased by 0.050000." in observations
    for forbidden_word in ["improved", "better", "worse", "optimal", "recommended"]:
        assert forbidden_word not in observation_text


def test_build_comparison_report_missing_baseline_metrics_raises(tmp_path):
    with pytest.raises(
        ValueError,
        match="ERROR: Baseline metrics not found. Run 'quantforge analyze' first.",
    ):
        build_comparison_report(tmp_path)


def test_build_comparison_report_no_backtested_variants_raises(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "baseline_metrics.json").write_text(
        json.dumps(_comparison_metrics()) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="ERROR: No backtested variants available for comparison.",
    ):
        build_comparison_report(tmp_path)


def test_build_comparison_report_schema_mismatch_raises(tmp_path):
    reports_dir = tmp_path / "reports"
    variant_dir = tmp_path / "variants" / "variant_001_volatility_filter"
    reports_dir.mkdir(parents=True)
    variant_dir.mkdir(parents=True)
    (reports_dir / "baseline_metrics.json").write_text(
        json.dumps(_comparison_metrics()) + "\n",
        encoding="utf-8",
    )
    variant_metrics = _comparison_metrics()
    del variant_metrics["win_rate"]
    (variant_dir / "metrics.json").write_text(
        json.dumps(variant_metrics) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="ERROR: Metrics schema mismatch between baseline and variant.",
    ):
        build_comparison_report(tmp_path)


def test_build_comparison_report_includes_multiple_variants(tmp_path):
    reports_dir = tmp_path / "reports"
    first_variant_dir = tmp_path / "variants" / "variant_001_volatility_filter"
    second_variant_dir = tmp_path / "variants" / "variant_existing"
    reports_dir.mkdir(parents=True)
    first_variant_dir.mkdir(parents=True)
    second_variant_dir.mkdir(parents=True)
    (reports_dir / "baseline_metrics.json").write_text(
        json.dumps(_comparison_metrics(total_return=0.1)) + "\n",
        encoding="utf-8",
    )
    (first_variant_dir / "metrics.json").write_text(
        json.dumps(_comparison_metrics(total_return=0.15)) + "\n",
        encoding="utf-8",
    )
    (second_variant_dir / "metrics.json").write_text(
        json.dumps(_comparison_metrics(total_return=0.05)) + "\n",
        encoding="utf-8",
    )

    report = build_comparison_report(tmp_path)
    lower_report = report.lower()

    assert report.startswith("# Strategy Comparison")
    assert "## Baseline vs variant_001_volatility_filter" in report
    assert "## Baseline vs variant_existing" in report
    assert report.count("### Metrics Comparison") == 2
    assert "See variants/variant_001_volatility_filter/diff.md for modification details." in report
    assert "This analysis does not constitute financial advice." in report
    for forbidden_word in ["improved", "better", "worse", "optimal", "recommended"]:
        assert forbidden_word not in lower_report
