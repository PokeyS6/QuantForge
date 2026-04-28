"""Backtest engine placeholder for QuantForge."""

import json
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


def run_and_persist_baseline_analysis(project: dict, data_csv: Path, project_file: Path) -> dict:
    """Run baseline RSI analysis and persist auditable outputs."""
    project_root = project_file.parent
    reports_dir = project_root / "reports"
    variants_dir = project_root / "variants" / "baseline"
    metrics_path = reports_dir / "baseline_metrics.json"
    results_path = variants_dir / "backtest_results.csv"
    signals_path = variants_dir / "signals.csv"

    if metrics_path.exists() or results_path.exists():
        raise ValueError(
            "ERROR: Baseline results already exist. Refusing to overwrite. "
            "Delete existing results or use a new project."
        )

    reports_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)

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
    metrics = {
        "total_return": summary["total_return"],
        "annualized_return": summary["annualized_return"],
        "max_drawdown": summary["max_drawdown"],
        "exposure": summary["exposure"],
        "trade_count": summary["trade_count"],
        "win_rate": summary["win_rate"],
    }

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    results_output = results.rename_axis("date").reset_index()
    results_output = results_output[["date", "close", "position", "asset_return", "strategy_return", "equity"]]
    results_output = results_output.sort_values("date")
    results_output.to_csv(results_path, index=False)

    signals_output = signals.rename_axis("date").reset_index()
    signals_output = signals_output[["date", "close", "rsi", "entry_signal", "exit_signal"]]
    signals_output = signals_output.sort_values("date").dropna()
    signals_output.to_csv(signals_path, index=False)

    return metrics
