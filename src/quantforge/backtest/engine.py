"""Backtest engine placeholder for QuantForge."""

import json
from pathlib import Path

import pandas as pd

from quantforge.backtest.metrics import summarize_backtest
from quantforge.backtest.trades import build_long_positions
from quantforge.core.project_io import get_baseline_variant
from quantforge.data.csv_loader import load_ohlcv_csv
from quantforge.strategies.rsi import generate_rsi_signals


def _format_report_date(value) -> str:
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


def _baseline_report_markdown(
    ticker: str,
    first_date,
    last_date,
    metrics: dict,
) -> str:
    low_trade_count = (
        "Trade count is below 50; results may be less reliable"
        if metrics["trade_count"] < 50
        else "not triggered"
    )
    high_drawdown = (
        "Max drawdown is below -30%; review downside risk"
        if metrics["max_drawdown"] < -0.3
        else "not triggered"
    )
    return "\n".join(
        [
            "# Strategy Report — Baseline",
            "",
            "## Strategy Summary",
            "- Strategy: RSI Reversal",
            f"- Ticker: {ticker}",
            f"- Period: {_format_report_date(first_date)} → {_format_report_date(last_date)}",
            "",
            "## Strategy Code",
            "The baseline strategy logic is implemented in the QuantForge analysis pipeline.",
            "Current implementation: RSI-based entry/exit logic using Wilder RSI.",
            "",
            "## Assumptions",
            "- Long-only",
            "- Assumed slippage: 0.05% per trade (not applied in backtest)",
            "- No transaction costs modeled",
            "- Uses daily OHLCV data",
            "",
            "## Metrics",
            f"- Total Return: {metrics['total_return']:.6f}",
            f"- Annualized Return: {metrics['annualized_return']:.6f}",
            f"- Max Drawdown: {metrics['max_drawdown']:.6f}",
            f"- Exposure: {metrics['exposure']:.6f}",
            f"- Trade Count: {metrics['trade_count']}",
            f"- Win Rate: {metrics['win_rate']:.6f}",
            "",
            "## Diagnostics",
            f"- Low trade count: {low_trade_count}",
            f"- High drawdown: {high_drawdown}",
            "- Slippage not modeled: Slippage is not modeled; results may be optimistic",
            "- Data source: user-provided CSV",
            "",
            "## Interpretation (Non-Advisory)",
            "This strategy was evaluated on historical data only.",
            "Performance may not generalize to future market conditions.",
            "This analysis does not constitute financial advice.",
            "",
        ]
    )


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


def _regime_summary(regime: pd.DataFrame) -> dict:
    position = regime["position"]
    trade_count = (position.eq(1) & position.shift(1, fill_value=0).eq(0)).sum()
    return {
        "sample_count": int(len(regime)),
        "avg_daily_return": float(regime["strategy_return"].mean()),
        "trade_count": int(trade_count),
        "exposure": float(position.mean()),
    }


def compute_volatility_regime_analysis(results: pd.DataFrame) -> dict:
    """Compute basic strategy behavior by volatility regime."""
    returns = results["close"].pct_change()
    volatility = returns.rolling(30).std().dropna()
    threshold = volatility.quantile(0.75)
    regime_results = results.loc[volatility.index]
    high_vol = volatility > threshold
    normal_vol = volatility <= threshold
    high_volatility = _regime_summary(regime_results.loc[high_vol])
    normal_volatility = _regime_summary(regime_results.loc[normal_vol])

    return {
        "threshold": float(threshold),
        "high_volatility": high_volatility,
        "normal_volatility": normal_volatility,
        "too_few_high_vol_trades": high_volatility["trade_count"] < 5,
    }


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
    report_path = reports_dir / "baseline_report.md"
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

    report = _baseline_report_markdown(
        ticker=parameters.get("ticker"),
        first_date=results.index[0],
        last_date=results.index[-1],
        metrics=metrics,
    )
    report_path.write_text(report, encoding="utf-8")

    return metrics
