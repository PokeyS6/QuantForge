import pandas as pd
import pytest

from quantforge.ai.momentum import apply_momentum_filter_to_signals


def prices() -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [100.0, 102.0, 104.0, 103.0, 108.0]},
        index=pd.date_range("2020-01-01", periods=5),
    )


def signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [100.0, 102.0, 104.0, 103.0, 108.0],
            "rsi": [40.0, 42.0, 44.0, 43.0, 45.0],
            "entry_signal": [True, True, True, False, True],
            "exit_signal": [False, False, True, False, True],
        },
        index=pd.date_range("2020-01-01", periods=5),
    )


def test_apply_momentum_filter_output_columns_exact():
    filtered = apply_momentum_filter_to_signals(
        prices(),
        signals(),
        lookback_days=2,
        threshold=0.02,
    )

    assert list(filtered.columns) == [
        "close",
        "rsi",
        "entry_signal",
        "exit_signal",
        "momentum_return",
        "momentum_entry_allowed",
    ]


def test_apply_momentum_filter_preserves_index():
    input_prices = prices()

    filtered = apply_momentum_filter_to_signals(
        input_prices,
        signals(),
        lookback_days=2,
        threshold=0.02,
    )

    pd.testing.assert_index_equal(filtered.index, input_prices.index)


def test_apply_momentum_filter_computes_past_close_pct_change():
    input_prices = prices()

    filtered = apply_momentum_filter_to_signals(
        input_prices,
        signals(),
        lookback_days=2,
        threshold=0.02,
    )

    expected = input_prices["close"].pct_change(2)
    pd.testing.assert_series_equal(
        filtered["momentum_return"],
        expected,
        check_names=False,
    )


def test_apply_momentum_filter_nan_warmup_rows_block_entries():
    filtered = apply_momentum_filter_to_signals(
        prices(),
        signals(),
        lookback_days=2,
        threshold=0.02,
    )

    assert filtered["momentum_entry_allowed"].iloc[:2].tolist() == [False, False]
    assert filtered["entry_signal"].iloc[:2].tolist() == [False, False]


def test_apply_momentum_filter_threshold_comparison_is_strictly_greater():
    input_prices = prices()
    threshold_equal_to_third_row = input_prices["close"].pct_change(2).iloc[2]

    filtered = apply_momentum_filter_to_signals(
        input_prices,
        signals(),
        lookback_days=2,
        threshold=threshold_equal_to_third_row,
    )

    assert filtered["momentum_entry_allowed"].iloc[2] == False
    assert filtered["entry_signal"].iloc[2] == False


def test_apply_momentum_filter_original_false_entries_stay_false():
    filtered = apply_momentum_filter_to_signals(
        prices(),
        signals(),
        lookback_days=2,
        threshold=-1.0,
    )

    assert filtered["momentum_entry_allowed"].iloc[3] == True
    assert filtered["entry_signal"].iloc[3] == False


def test_apply_momentum_filter_exit_signal_unchanged():
    input_signals = signals()

    filtered = apply_momentum_filter_to_signals(
        prices(),
        input_signals,
        lookback_days=2,
        threshold=0.02,
    )

    pd.testing.assert_series_equal(filtered["exit_signal"], input_signals["exit_signal"])


def test_apply_momentum_filter_missing_close_raises():
    with pytest.raises(ValueError, match="Prices must include a close column."):
        apply_momentum_filter_to_signals(
            prices().drop(columns=["close"]),
            signals(),
            lookback_days=2,
            threshold=0.02,
        )


def test_apply_momentum_filter_missing_signal_column_raises():
    with pytest.raises(
        ValueError,
        match="Signals must include close, rsi, entry_signal, and exit_signal columns.",
    ):
        apply_momentum_filter_to_signals(
            prices(),
            signals().drop(columns=["rsi"]),
            lookback_days=2,
            threshold=0.02,
        )


def test_apply_momentum_filter_index_mismatch_raises():
    mismatched_signals = signals()
    mismatched_signals.index = pd.date_range("2021-01-01", periods=5)

    with pytest.raises(ValueError, match="Prices and signals indexes must match."):
        apply_momentum_filter_to_signals(
            prices(),
            mismatched_signals,
            lookback_days=2,
            threshold=0.02,
        )


def test_apply_momentum_filter_invalid_lookback_days_raises():
    with pytest.raises(ValueError, match="lookback_days must be greater than 0."):
        apply_momentum_filter_to_signals(
            prices(),
            signals(),
            lookback_days=0,
            threshold=0.02,
        )
