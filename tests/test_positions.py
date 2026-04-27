import pandas as pd
import pytest

from quantforge.backtest.trades import build_long_positions


def test_build_long_positions_basic_entry_then_exit_flow():
    signals = pd.DataFrame(
        {
            "entry_signal": [False, True, False, False, False],
            "exit_signal": [False, False, False, True, False],
        }
    )

    positions = build_long_positions(signals)

    assert positions.tolist() == [0, 1, 1, 0, 0]
    assert positions.name == "position"


def test_build_long_positions_multiple_entries_without_exits_remain_long():
    signals = pd.DataFrame(
        {
            "entry_signal": [False, True, True, False],
            "exit_signal": [False, False, False, False],
        }
    )

    positions = build_long_positions(signals)

    assert positions.tolist() == [0, 1, 1, 1]


def test_build_long_positions_multiple_exits_without_entries_remain_flat():
    signals = pd.DataFrame(
        {
            "entry_signal": [False, False, False],
            "exit_signal": [True, True, False],
        }
    )

    positions = build_long_positions(signals)

    assert positions.tolist() == [0, 0, 0]


def test_build_long_positions_exit_wins_when_entry_and_exit_same_row():
    signals = pd.DataFrame(
        {
            "entry_signal": [True, True, False],
            "exit_signal": [False, True, False],
        }
    )

    positions = build_long_positions(signals)

    assert positions.tolist() == [1, 0, 0]


def test_build_long_positions_preserves_index():
    index = pd.date_range("2020-01-01", periods=3)
    signals = pd.DataFrame(
        {
            "entry_signal": [False, True, False],
            "exit_signal": [False, False, True],
        },
        index=index,
    )

    positions = build_long_positions(signals)

    assert positions.index.equals(index)


def test_build_long_positions_invalid_input_raises_value_error():
    signals = pd.DataFrame({"entry_signal": [True, False]})

    with pytest.raises(ValueError, match="exit_signal"):
        build_long_positions(signals)
