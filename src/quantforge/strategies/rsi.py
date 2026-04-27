"""RSI strategy placeholder for QuantForge."""

import pandas as pd

from quantforge.strategies.indicators import calculate_rsi


def generate_rsi_signals(
    prices: pd.DataFrame,
    entry_rsi: float = 30,
    exit_rsi: float = 70,
    rsi_window: int = 14,
) -> pd.DataFrame:
    """Generate deterministic RSI threshold signals from close prices."""
    if "close" not in prices.columns:
        raise ValueError("Prices must include a close column.")
    if not 0 <= entry_rsi <= 100:
        raise ValueError("entry_rsi must be between 0 and 100.")
    if not 0 <= exit_rsi <= 100:
        raise ValueError("exit_rsi must be between 0 and 100.")
    if entry_rsi >= exit_rsi:
        raise ValueError("entry_rsi must be less than exit_rsi.")
    if rsi_window <= 0:
        raise ValueError("rsi_window must be greater than 0.")

    signals = pd.DataFrame(index=prices.index)
    signals["close"] = prices["close"]
    signals["rsi"] = calculate_rsi(prices["close"], window=rsi_window)
    signals["entry_signal"] = (signals["rsi"] < entry_rsi).fillna(False)
    signals["exit_signal"] = (signals["rsi"] > exit_rsi).fillna(False)
    return signals
