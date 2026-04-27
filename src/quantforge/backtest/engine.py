"""Backtest engine placeholder for QuantForge."""

import pandas as pd


def run_long_backtest(
    prices: pd.DataFrame,
    positions: pd.Series,
    initial_equity: float = 1.0,
) -> pd.DataFrame:
    """Run a minimal long-only backtest from close prices and positions."""
    if "close" not in prices.columns:
        raise ValueError("Prices must include a close column.")
    if not prices.index.equals(positions.index):
        raise ValueError("Positions index must match prices index.")
    if not positions.isin([0, 1]).all():
        raise ValueError("Positions must contain only 0 or 1 values.")
    if initial_equity <= 0:
        raise ValueError("initial_equity must be greater than 0.")

    results = pd.DataFrame(index=prices.index)
    results["close"] = prices["close"]
    results["position"] = positions
    results["asset_return"] = prices["close"].pct_change().fillna(0)
    results["strategy_return"] = positions.shift(1).fillna(0) * results["asset_return"]
    results["equity"] = initial_equity * (1 + results["strategy_return"]).cumprod()
    return results
