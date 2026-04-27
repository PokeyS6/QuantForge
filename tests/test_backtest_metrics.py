import pandas as pd
import pytest

from quantforge.backtest.metrics import summarize_backtest


def test_summarize_backtest_schema_is_exact():
    results = pd.DataFrame(
        {
            "equity": [100, 110],
            "strategy_return": [0, 0.1],
            "position": [0, 1],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )

    summary = summarize_backtest(results)

    assert list(summary) == [
        "total_return",
        "annualized_return",
        "max_drawdown",
        "exposure",
        "trade_count",
        "win_rate",
    ]


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


def test_summarize_backtest_annualized_return_computed_correctly():
    results = pd.DataFrame(
        {
            "equity": [100, 110],
            "strategy_return": [0, 0.1],
            "position": [1, 1],
        },
        index=pd.to_datetime(["2020-01-01", "2021-01-01"]),
    )

    summary = summarize_backtest(results)

    assert summary["annualized_return"] == pytest.approx((1.1 ** (365.25 / 366)) - 1)


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


def test_summarize_backtest_win_rate_computed_correctly():
    results = pd.DataFrame(
        {
            "equity": [100, 100, 110, 110, 100, 100, 100],
            "strategy_return": [0, 0, 0.1, 0, -0.0909, 0, 0],
            "position": [0, 1, 1, 0, 1, 1, 0],
        },
        index=pd.date_range("2020-01-01", periods=7),
    )

    summary = summarize_backtest(results)

    assert summary["win_rate"] == pytest.approx(0.5)


def test_summarize_backtest_missing_columns_raises_value_error():
    results = pd.DataFrame({"equity": [100], "position": [0]})

    with pytest.raises(ValueError, match="strategy_return"):
        summarize_backtest(results)


def test_summarize_backtest_empty_results_raises_value_error():
    results = pd.DataFrame(columns=["equity", "strategy_return", "position"])

    with pytest.raises(ValueError, match="empty"):
        summarize_backtest(results)


def test_summarize_backtest_zero_initial_equity_raises():
    results = pd.DataFrame(
        {
            "equity": [0, 100],
            "strategy_return": [0, 0.1],
            "position": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="Initial equity cannot be zero"):
        summarize_backtest(results)
