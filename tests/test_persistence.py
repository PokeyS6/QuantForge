import json

import pandas as pd
import pytest

from quantforge.backtest.engine import run_and_persist_baseline_analysis
from quantforge.core.project_io import read_project


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
