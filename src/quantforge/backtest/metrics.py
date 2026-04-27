"""Backtest metrics placeholder for QuantForge."""

import pandas as pd

REQUIRED_COLUMNS = ["equity", "strategy_return", "position"]


def summarize_backtest(results: pd.DataFrame) -> dict[str, float | int]:
    """Summarize basic deterministic metrics from backtest results."""
    if results.empty:
        raise ValueError("Backtest results must not be empty.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in results.columns]
    if missing_columns:
        raise ValueError(f"Missing required result columns: {', '.join(missing_columns)}")

    equity = results["equity"]
    position = results["position"]
    drawdown = equity / equity.cummax() - 1
    entries = (position.eq(1) & position.shift(1, fill_value=0).eq(0)).sum()

    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "max_drawdown": float(drawdown.min()),
        "exposure": float(position.mean()),
        "trade_count": int(entries),
    }
