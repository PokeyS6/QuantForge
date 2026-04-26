"""Indicator placeholders for QuantForge strategies."""

import pandas as pd


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Wilder's RSI from close prices."""
    if window <= 0:
        raise ValueError("RSI window must be greater than 0.")
    if close.empty:
        raise ValueError("Close price series must not be empty.")

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = pd.Series(index=close.index, dtype=float)
    average_loss = pd.Series(index=close.index, dtype=float)
    if len(close) > window:
        average_gain.iloc[window] = gains.iloc[1 : window + 1].mean()
        average_loss.iloc[window] = losses.iloc[1 : window + 1].mean()

        for position in range(window + 1, len(close)):
            average_gain.iloc[position] = (
                average_gain.iloc[position - 1] * (window - 1) + gains.iloc[position]
            ) / window
            average_loss.iloc[position] = (
                average_loss.iloc[position - 1] * (window - 1) + losses.iloc[position]
            ) / window

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_gain == 0) & (average_loss > 0), 0)
    rsi = rsi.mask((average_gain == 0) & (average_loss == 0), 50)
    return rsi.clip(lower=0, upper=100)
