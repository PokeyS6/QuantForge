from typer.testing import CliRunner

from quantforge.cli import app

runner = CliRunner()


def test_analyze_with_csv_runs_baseline_summary(tmp_path):
    create_result = runner.invoke(
        app,
        [
            "create",
            "Buy when RSI < 30, sell when RSI > 70",
            "--ticker",
            "AAPL",
            "--start",
            "2020-01-01",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert create_result.exit_code == 0

    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
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

    project_file = tmp_path / "aapl-rsi-reversal" / "strategy.qf.json"
    analyze_result = runner.invoke(
        app,
        ["analyze", str(project_file), "--data-csv", str(csv_path)],
    )

    assert analyze_result.exit_code == 0
    assert "strategy_id: rsi_reversal_aapl" in analyze_result.output
    assert "total_return:" in analyze_result.output
    assert "max_drawdown:" in analyze_result.output
    assert "trade_count:" in analyze_result.output
    assert "No strategy recommendation is being made." in analyze_result.output
