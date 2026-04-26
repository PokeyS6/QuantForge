import pandas as pd
import pytest

from quantforge.strategies.indicators import calculate_rsi


def test_calculate_wilder_rsi_preserves_index():
    index = pd.date_range("2020-01-01", periods=20)
    close = pd.Series(range(20), index=index)

    rsi = calculate_rsi(close, window=14)

    assert rsi.index.equals(index)


def test_calculate_wilder_rsi_returns_values_between_zero_and_one_hundred_after_warmup():
    close = pd.Series([1, 2, 3, 2, 4, 3, 5, 4, 6, 5])

    rsi = calculate_rsi(close, window=3).dropna()

    assert not rsi.empty
    assert rsi.between(0, 100).all()


def test_calculate_wilder_rsi_increasing_prices_produce_high_rsi_after_warmup():
    close = pd.Series(range(1, 30))

    rsi = calculate_rsi(close, window=14)

    assert rsi.dropna().iloc[-1] == 100


def test_calculate_wilder_rsi_decreasing_prices_produce_low_rsi_after_warmup():
    close = pd.Series(range(30, 1, -1))

    rsi = calculate_rsi(close, window=14)

    assert rsi.dropna().iloc[-1] == 0


def test_calculate_wilder_rsi_matches_manual_recursive_value():
    close = pd.Series([10, 12, 11, 13, 14, 13])

    rsi = calculate_rsi(close, window=3)

    assert rsi.iloc[:3].isna().all()
    assert rsi.iloc[3] == pytest.approx(80.0)
    assert rsi.iloc[4] == pytest.approx(84.6153846154)
    assert rsi.iloc[5] == pytest.approx(62.8571428571)


def test_calculate_wilder_rsi_invalid_window_raises_value_error():
    close = pd.Series([1, 2, 3])

    with pytest.raises(ValueError, match="window"):
        calculate_rsi(close, window=0)


def test_calculate_wilder_rsi_empty_series_raises_value_error():
    close = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="empty"):
        calculate_rsi(close)
