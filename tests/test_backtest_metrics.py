import pandas as pd
import pytest

from quantforge.backtest.metrics import summarize_backtest


def test_summarize_backtest_total_return_computed_correctly():
    results = pd.DataFrame(
        {
            "equity": [100, 110, 121],
            "strategy_return": [0, 0.1, 0.1],
            "position": [1, 1, 1],
        }
    )

    summary = summarize_backtest(results)

    assert summary["total_return"] == pytest.approx(0.21)


def test_summarize_backtest_max_drawdown_computed_correctly():
    results = pd.DataFrame(
        {
            "equity": [100, 120, 90, 108],
            "strategy_return": [0, 0.2, -0.25, 0.2],
            "position": [1, 1, 1, 1],
        }
    )

    summary = summarize_backtest(results)

    assert summary["max_drawdown"] == pytest.approx(-0.25)


def test_summarize_backtest_exposure_computed_correctly():
    results = pd.DataFrame(
        {
            "equity": [100, 101, 102, 103],
            "strategy_return": [0, 0.01, 0.01, 0.01],
            "position": [0, 1, 1, 0],
        }
    )

    summary = summarize_backtest(results)

    assert summary["exposure"] == pytest.approx(0.5)


def test_summarize_backtest_trade_count_computed_correctly():
    results = pd.DataFrame(
        {
            "equity": [100, 101, 102, 103, 104, 105],
            "strategy_return": [0, 0.01, 0.01, 0.01, 0.01, 0.01],
            "position": [0, 1, 1, 0, 1, 0],
        }
    )

    summary = summarize_backtest(results)

    assert summary["trade_count"] == 2


def test_summarize_backtest_missing_columns_raises_value_error():
    results = pd.DataFrame({"equity": [100], "position": [0]})

    with pytest.raises(ValueError, match="strategy_return"):
        summarize_backtest(results)


def test_summarize_backtest_empty_results_raises_value_error():
    results = pd.DataFrame(columns=["equity", "strategy_return", "position"])

    with pytest.raises(ValueError, match="empty"):
        summarize_backtest(results)
