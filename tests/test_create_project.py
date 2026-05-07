import json

from typer.testing import CliRunner

from quantforge.cli import app

runner = CliRunner()


def test_create_project_writes_metadata(tmp_path):
    result = runner.invoke(
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

    assert result.exit_code == 0
    assert "Created QuantForge project:" in result.output
    assert "No strategy recommendation is being made." in result.output

    project_path = tmp_path / "aapl-rsi-reversal"
    metadata_path = project_path / "strategy.qf.json"
    baseline_path = project_path / "variants" / "baseline" / "strategy_config.json"

    assert metadata_path.exists()
    assert baseline_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    assert metadata["strategy_id"] == "rsi_reversal_aapl"
    assert metadata["project_schema_version"] == "0.1"
    assert metadata["asset_universe"] == ["AAPL"]
    assert metadata["timeframe"] == "1d"
    assert metadata["data_window"]["start"] == "2020-01-01"
    assert metadata["baseline_variant_id"] == "baseline"
    assert metadata["variants"][0]["variant_id"] == "baseline"
    assert baseline["parameters"]["ticker"] == "AAPL"
    assert baseline["parameters"]["entry_rsi"] == 30
    assert baseline["parameters"]["exit_rsi"] == 70
    assert "not recommending a trade" in " ".join(baseline["assumptions"])
