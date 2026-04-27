"""Backtest engine placeholder for QuantForge."""

from pathlib import Path

import pandas as pd

from quantforge.backtest.metrics import summarize_backtest
from quantforge.backtest.trades import build_long_positions
from quantforge.core.project_io import get_baseline_variant
from quantforge.data.csv_loader import load_ohlcv_csv
from quantforge.strategies.rsi import generate_rsi_signals


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


def run_baseline_rsi_analysis(project: dict, data_csv: Path) -> dict:
    """Run baseline RSI analysis from project metadata and a local CSV."""
    baseline = get_baseline_variant(project)
    parameters = baseline.get("parameters", {})
    prices = load_ohlcv_csv(data_csv)
    signals = generate_rsi_signals(
        prices,
        entry_rsi=parameters.get("entry_rsi", 30),
        exit_rsi=parameters.get("exit_rsi", 70),
        rsi_window=parameters.get("rsi_window", 14),
    )
    positions = build_long_positions(signals)
    results = run_long_backtest(prices, positions)
    summary = summarize_backtest(results)

    return {
        "strategy_id": project.get("strategy_id"),
        "ticker": parameters.get("ticker"),
        "strategy_type": baseline.get("strategy_type"),
        "total_return": summary["total_return"],
        "max_drawdown": summary["max_drawdown"],
        "exposure": summary["exposure"],
        "trade_count": summary["trade_count"],
    }
