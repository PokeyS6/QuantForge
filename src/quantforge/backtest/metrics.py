"""Backtest metrics placeholder for QuantForge."""

import pandas as pd

REQUIRED_COLUMNS = ["equity", "strategy_return", "position"]


def _annualized_return(total_return: float, results: pd.DataFrame) -> float:
    if not isinstance(results.index, pd.DatetimeIndex) or len(results.index) < 2:
        return total_return

    days = (results.index[-1] - results.index[0]).days
    if days <= 0:
        return total_return

    return (1 + total_return) ** (365.25 / days) - 1


def _win_rate(results: pd.DataFrame) -> float:
    position = results["position"]
    equity = results["equity"]
    previous_position = position.shift(1, fill_value=0)
    entry_indexes = list(results.index[position.eq(1) & previous_position.eq(0)])
    exit_indexes = list(results.index[position.eq(0) & previous_position.eq(1)])

    completed_returns = []
    for entry_index in entry_indexes:
        exits_after_entry = [exit_index for exit_index in exit_indexes if exit_index > entry_index]
        if not exits_after_entry:
            continue
        exit_index = exits_after_entry[0]
        completed_returns.append(equity.loc[exit_index] / equity.loc[entry_index] - 1)

    if not completed_returns:
        return 0.0

    wins = sum(trade_return > 0 for trade_return in completed_returns)
    return wins / len(completed_returns)


def summarize_backtest(results: pd.DataFrame) -> dict[str, float | int]:
    """Summarize basic deterministic metrics from backtest results."""
    if results.empty:
        raise ValueError("Backtest results must not be empty.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in results.columns]
    if missing_columns:
        raise ValueError(f"Missing required result columns: {', '.join(missing_columns)}")

    equity = results["equity"]
    position = results["position"]
    if equity.iloc[0] == 0:
        raise ValueError("Initial equity cannot be zero.")

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    drawdown = equity / equity.cummax() - 1
    entries = (position.eq(1) & position.shift(1, fill_value=0).eq(0)).sum()

    return {
        "total_return": total_return,
        "annualized_return": float(_annualized_return(total_return, results)),
        "max_drawdown": float(drawdown.min()),
        "exposure": float(position.mean()),
        "trade_count": int(entries),
        "win_rate": float(_win_rate(results)),
    }
