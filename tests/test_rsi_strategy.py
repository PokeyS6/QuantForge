import pandas as pd
import pytest

from quantforge.strategies.rsi import generate_rsi_signals


def test_generate_rsi_signals_output_columns_exist():
    prices = pd.DataFrame({"close": range(1, 30)})

    signals = generate_rsi_signals(prices)

    assert list(signals.columns) == ["close", "rsi", "entry_signal", "exit_signal"]


def test_generate_rsi_signals_preserves_index():
    index = pd.date_range("2020-01-01", periods=29)
    prices = pd.DataFrame({"close": range(1, 30)}, index=index)

    signals = generate_rsi_signals(prices)

    assert signals.index.equals(index)


def test_generate_rsi_signals_boolean_signal_columns():
    prices = pd.DataFrame({"close": range(1, 30)})

    signals = generate_rsi_signals(prices)

    assert signals["entry_signal"].dtype == bool
    assert signals["exit_signal"].dtype == bool


def test_generate_rsi_signals_entry_signal_triggers_for_low_rsi():
    prices = pd.DataFrame({"close": range(30, 1, -1)})

    signals = generate_rsi_signals(prices, entry_rsi=30, exit_rsi=70)

    assert signals["entry_signal"].any()
    assert signals["entry_signal"].iloc[-1]


def test_generate_rsi_signals_exit_signal_triggers_for_high_rsi():
    prices = pd.DataFrame({"close": range(1, 30)})

    signals = generate_rsi_signals(prices, entry_rsi=30, exit_rsi=70)

    assert signals["exit_signal"].any()
    assert signals["exit_signal"].iloc[-1]


def test_generate_rsi_signals_missing_close_raises_value_error():
    prices = pd.DataFrame({"open": [1, 2, 3]})

    with pytest.raises(ValueError, match="close"):
        generate_rsi_signals(prices)


@pytest.mark.parametrize(
    ("entry_rsi", "exit_rsi", "match"),
    [
        (-1, 70, "entry_rsi"),
        (30, 101, "exit_rsi"),
        (70, 30, "less than"),
    ],
)
def test_generate_rsi_signals_invalid_thresholds_raise_value_error(entry_rsi, exit_rsi, match):
    prices = pd.DataFrame({"close": range(1, 30)})

    with pytest.raises(ValueError, match=match):
        generate_rsi_signals(prices, entry_rsi=entry_rsi, exit_rsi=exit_rsi)


def test_generate_rsi_signals_invalid_rsi_window_raises_value_error():
    prices = pd.DataFrame({"close": range(1, 30)})

    with pytest.raises(ValueError, match="rsi_window"):
        generate_rsi_signals(prices, rsi_window=0)
