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
    regime_analysis: dict,
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
    high_volatility = regime_analysis["high_volatility"]
    normal_volatility = regime_analysis["normal_volatility"]
    observation = _regime_observation(regime_analysis)
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
            "## Regime Analysis (Volatility)",
            "",
            "Volatility is measured as the 30-day rolling standard deviation of daily returns.",
            "Rows without enough lookback history are excluded from this analysis.",
            "",
            f"- High-volatility threshold: {regime_analysis['threshold']:.6f}",
            f"- High-volatility rows: {high_volatility['sample_count']}",
            f"- Normal-volatility rows: {normal_volatility['sample_count']}",
            "",
            "### Regime Metrics",
            "",
            "| Regime | Avg Daily Return | Exposure | Trade Count |",
            "|--------|-----------------|----------|-------------|",
            "| High Volatility | "
            f"{high_volatility['avg_daily_return']:.6f} | "
            f"{high_volatility['exposure']:.6f} | "
            f"{high_volatility['trade_count']} |",
            "| Normal Volatility | "
            f"{normal_volatility['avg_daily_return']:.6f} | "
            f"{normal_volatility['exposure']:.6f} | "
            f"{normal_volatility['trade_count']} |",
            "",
            "### Observation",
            "",
            observation,
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


def _comparison_label(left: float, right: float, tolerance: float = 1e-6) -> str:
    if abs(left - right) <= tolerance:
        return "similar"
    if left > right:
        return "higher"
    return "lower"


def _regime_observation(regime_analysis: dict) -> str:
    if regime_analysis["too_few_high_vol_trades"]:
        return "The high-volatility subset contains too few trades for meaningful comparison."

    high_volatility = regime_analysis["high_volatility"]
    normal_volatility = regime_analysis["normal_volatility"]
    return_comparison = _comparison_label(
        high_volatility["avg_daily_return"],
        normal_volatility["avg_daily_return"],
    )
    exposure_comparison = _comparison_label(
        high_volatility["exposure"],
        normal_volatility["exposure"],
    )
    return "\n".join(
        [
            "The strategy’s return profile differed between volatility regimes.",
            "",
            "In high-volatility periods, average daily return was "
            f"{return_comparison} and exposure was {exposure_comparison} "
            "compared with normal-volatility periods.",
            "",
            "Caution: The high-volatility subset may be small and may not represent future market conditions.",
            "This analysis does not predict future performance.",
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
    final_output_paths = [metrics_path, report_path, results_path, signals_path]

    if any(path.exists() for path in final_output_paths):
        raise ValueError(
            "ERROR: Baseline results already exist. Refusing to overwrite. "
            "Delete existing results or use a new project."
        )

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
    regime_analysis = compute_volatility_regime_analysis(results)
    metrics = {
        "total_return": summary["total_return"],
        "annualized_return": summary["annualized_return"],
        "max_drawdown": summary["max_drawdown"],
        "exposure": summary["exposure"],
        "trade_count": summary["trade_count"],
        "win_rate": summary["win_rate"],
    }

    results_output = results.rename_axis("date").reset_index()
    results_output = results_output[["date", "close", "position", "asset_return", "strategy_return", "equity"]]
    results_output = results_output.sort_values("date")

    signals_output = signals.rename_axis("date").reset_index()
    signals_output = signals_output[["date", "close", "rsi", "entry_signal", "exit_signal"]]
    signals_output = signals_output.sort_values("date").dropna()

    report = _baseline_report_markdown(
        ticker=parameters.get("ticker"),
        first_date=results.index[0],
        last_date=results.index[-1],
        metrics=metrics,
        regime_analysis=regime_analysis,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = {
        metrics_path: reports_dir / ".baseline_metrics.json.tmp",
        report_path: reports_dir / ".baseline_report.md.tmp",
        results_path: variants_dir / ".backtest_results.csv.tmp",
        signals_path: variants_dir / ".signals.csv.tmp",
    }
    try:
        temp_paths[metrics_path].write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        results_output.to_csv(temp_paths[results_path], index=False)
        signals_output.to_csv(temp_paths[signals_path], index=False)
        temp_paths[report_path].write_text(report, encoding="utf-8")

        for final_path in [metrics_path, results_path, signals_path, report_path]:
            _replace_temp_file(temp_paths[final_path], final_path)
    except Exception:
        _cleanup_failed_baseline_persistence(
            temp_paths=list(temp_paths.values()),
            final_output_paths=final_output_paths,
        )
        raise

    return metrics


def _replace_temp_file(temp_path: Path, final_path: Path) -> None:
    temp_path.replace(final_path)


def _cleanup_failed_baseline_persistence(
    temp_paths: list[Path],
    final_output_paths: list[Path],
) -> None:
    for temp_path in temp_paths:
        temp_path.unlink(missing_ok=True)
    for final_path in final_output_paths:
        final_path.unlink(missing_ok=True)
