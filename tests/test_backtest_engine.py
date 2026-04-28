import pandas as pd
import pytest

from quantforge.backtest.engine import compute_volatility_regime_analysis, run_long_backtest


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


def test_compute_volatility_regime_analysis_returns_expected_regime_values():
    returns = pd.Series(
        [0.001] * 20
        + [0.02, -0.015, 0.018, -0.02, 0.022, -0.017, 0.019, -0.021, 0.023, -0.018]
        + [0.003, -0.002, 0.004, -0.003, 0.005, -0.004, 0.006, -0.005, 0.007, -0.006]
    )
    close = 100 * (1 + returns).cumprod()
    index = pd.date_range("2020-01-01", periods=len(close))
    position = pd.Series(
        [0] * 30 + [1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        index=index,
    )
    results = pd.DataFrame(
        {
            "close": close.to_list(),
            "strategy_return": [0.001 * value for value in range(len(close))],
            "position": position,
        },
        index=index,
    )
    expected_volatility = results["close"].pct_change().rolling(30).std().dropna()
    expected_threshold = expected_volatility.quantile(0.75)
    expected_results = results.loc[expected_volatility.index]
    expected_high = expected_results.loc[expected_volatility > expected_threshold]
    expected_normal = expected_results.loc[expected_volatility <= expected_threshold]

    analysis = compute_volatility_regime_analysis(results)

    assert analysis["threshold"] == pytest.approx(expected_threshold)
    assert (
        analysis["high_volatility"]["sample_count"]
        + analysis["normal_volatility"]["sample_count"]
        == len(expected_volatility)
    )
    assert analysis["high_volatility"]["sample_count"] == len(expected_high)
    assert analysis["normal_volatility"]["sample_count"] == len(expected_normal)
    assert analysis["high_volatility"]["exposure"] == pytest.approx(
        expected_high["position"].mean()
    )
    assert analysis["normal_volatility"]["exposure"] == pytest.approx(
        expected_normal["position"].mean()
    )
    assert analysis["high_volatility"]["trade_count"] == int(
        (
            expected_high["position"].eq(1)
            & expected_high["position"].shift(1, fill_value=0).eq(0)
        ).sum()
    )
    assert analysis["normal_volatility"]["trade_count"] == int(
        (
            expected_normal["position"].eq(1)
            & expected_normal["position"].shift(1, fill_value=0).eq(0)
        ).sum()
    )
    assert analysis["too_few_high_vol_trades"] is True
