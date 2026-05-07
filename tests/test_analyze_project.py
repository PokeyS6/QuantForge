from typer.testing import CliRunner

from quantforge.cli import app

runner = CliRunner()


def test_analyze_project_reads_metadata(tmp_path):
    create_result = runner.invoke(
        app,
        [
            "create",
            "RSI reversal when RSI crosses configured thresholds",
            "--ticker",
            "AAPL",
            "--start",
            "2020-01-01",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert create_result.exit_code == 0

    project_file = tmp_path / "aapl-rsi-reversal" / "strategy.qf.json"
    analyze_result = runner.invoke(app, ["analyze", str(project_file)])

    assert analyze_result.exit_code == 0
    assert "strategy_id: rsi_reversal_aapl" in analyze_result.output
    assert "AAPL" in analyze_result.output
    assert "timeframe: 1d" in analyze_result.output
    assert "baseline_variant_id: baseline" in analyze_result.output
    assert "No strategy recommendation is being made." in analyze_result.output
