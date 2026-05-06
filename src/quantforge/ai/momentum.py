"""Deterministic momentum filter helpers for AI-assisted variants."""

import pandas as pd


REQUIRED_SIGNAL_COLUMNS = ["close", "rsi", "entry_signal", "exit_signal"]
OUTPUT_COLUMNS = [
    "close",
    "rsi",
    "entry_signal",
    "exit_signal",
    "momentum_return",
    "momentum_entry_allowed",
]


def apply_momentum_filter_to_signals(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    lookback_days: int,
    threshold: float,
) -> pd.DataFrame:
    """Apply a past-close momentum entry filter to RSI signals."""
    if "close" not in prices.columns:
        raise ValueError("Prices must include a close column.")

    missing_signal_columns = [
        column for column in REQUIRED_SIGNAL_COLUMNS if column not in signals.columns
    ]
    if missing_signal_columns:
        raise ValueError("Signals must include close, rsi, entry_signal, and exit_signal columns.")

    if not prices.index.equals(signals.index):
        raise ValueError("Prices and signals indexes must match.")

    if lookback_days <= 0:
        raise ValueError("lookback_days must be greater than 0.")

    momentum_return = prices["close"].pct_change(lookback_days)
    momentum_entry_allowed = (momentum_return > threshold).fillna(False)

    filtered = signals[REQUIRED_SIGNAL_COLUMNS].copy()
    filtered["entry_signal"] = filtered["entry_signal"] & momentum_entry_allowed
    filtered["momentum_return"] = momentum_return
    filtered["momentum_entry_allowed"] = momentum_entry_allowed
    return filtered[OUTPUT_COLUMNS]
