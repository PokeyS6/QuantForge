import pandas as pd
import pytest

from quantforge.backtest.engine import run_long_backtest


def test_run_long_backtest_output_columns_exist():
    prices = pd.DataFrame({"close": [100, 110, 121]})
    positions = pd.Series([0, 1, 1])

    results = run_long_backtest(prices, positions)

    assert list(results.columns) == [
        "close",
        "position",
        "asset_return",
        "strategy_return",
        "equity",
    ]


def test_run_long_backtest_preserves_index():
    index = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"close": [100, 110, 121]}, index=index)
    positions = pd.Series([0, 1, 1], index=index)

    results = run_long_backtest(prices, positions)

    assert results.index.equals(index)


def test_run_long_backtest_first_strategy_return_is_zero():
    prices = pd.DataFrame({"close": [100, 110, 121]})
    positions = pd.Series([1, 1, 1])

    results = run_long_backtest(prices, positions)

    assert results["strategy_return"].iloc[0] == 0


def test_run_long_backtest_uses_previous_position_not_same_bar_position():
    prices = pd.DataFrame({"close": [100, 110, 121]})
    positions = pd.Series([0, 1, 1])

    results = run_long_backtest(prices, positions)

    assert results["asset_return"].tolist() == pytest.approx([0, 0.1, 0.1])
    assert results["strategy_return"].tolist() == pytest.approx([0, 0, 0.1])


def test_run_long_backtest_equity_compounds_correctly():
    prices = pd.DataFrame({"close": [100, 110, 121]})
    positions = pd.Series([1, 1, 1])

    results = run_long_backtest(prices, positions, initial_equity=1000)

    assert results["equity"].tolist() == pytest.approx([1000, 1100, 1210])


def test_run_long_backtest_missing_close_raises_value_error():
    prices = pd.DataFrame({"open": [100, 110]})
    positions = pd.Series([0, 1])

    with pytest.raises(ValueError, match="close"):
        run_long_backtest(prices, positions)


def test_run_long_backtest_index_mismatch_raises_value_error():
    prices = pd.DataFrame({"close": [100, 110]}, index=[0, 1])
    positions = pd.Series([0, 1], index=[1, 2])

    with pytest.raises(ValueError, match="index"):
        run_long_backtest(prices, positions)


def test_run_long_backtest_invalid_position_values_raise_value_error():
    prices = pd.DataFrame({"close": [100, 110]})
    positions = pd.Series([0, 2])

    with pytest.raises(ValueError, match="0 or 1"):
        run_long_backtest(prices, positions)


def test_run_long_backtest_invalid_initial_equity_raises_value_error():
    prices = pd.DataFrame({"close": [100, 110]})
    positions = pd.Series([0, 1])

    with pytest.raises(ValueError, match="initial_equity"):
        run_long_backtest(prices, positions, initial_equity=0)
