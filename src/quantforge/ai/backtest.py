"""Backtest persistence helpers for AI-assisted variants."""

import json
from pathlib import Path

from quantforge.ai.momentum import apply_momentum_filter_to_signals
from quantforge.backtest.metrics import summarize_backtest
from quantforge.backtest.trades import build_long_positions
from quantforge.backtest.engine import run_long_backtest
from quantforge.data.csv_loader import load_ohlcv_csv
from quantforge.strategies.rsi import generate_rsi_signals


METRIC_KEYS = [
    "total_return",
    "annualized_return",
    "max_drawdown",
    "exposure",
    "trade_count",
    "win_rate",
]


def run_and_persist_ai_momentum_variant_analysis(
    project_file: Path,
    data_csv: Path,
    variant_id: str,
) -> dict:
    """Run and persist an already-created AI momentum-filter variant backtest."""
    project_root = project_file.parent
    baseline_results_path = project_root / "variants" / "baseline" / "backtest_results.csv"
    variant_dir = project_root / "variants" / variant_id
    variant_config_path = variant_dir / "strategy_config.json"
    results_path = variant_dir / "backtest_results.csv"
    signals_path = variant_dir / "signals.csv"
    metrics_path = variant_dir / "metrics.json"
    report_path = variant_dir / "report.md"
    change_summary_path = variant_dir / "change_summary.json"

    if not baseline_results_path.exists():
        raise ValueError("ERROR: Baseline analysis not found. Run 'quantforge analyze' first.")
    if not variant_dir.exists():
        raise ValueError(f"ERROR: Variant {variant_id} not found.")
    if not variant_config_path.exists():
        raise ValueError(f"Variant config not found: {variant_config_path}")
    if results_path.exists() or metrics_path.exists() or report_path.exists():
        raise ValueError(
            f"ERROR: Variant analysis already exists for {variant_id}. Refusing to overwrite."
        )

    project = json.loads(project_file.read_text(encoding="utf-8"))
    variant_config = json.loads(variant_config_path.read_text(encoding="utf-8"))
    modification_type = variant_config.get("modification_type")
    if modification_type != "momentum_filter":
        raise ValueError(
            f"ERROR: Backtesting for AI modification type '{modification_type}' "
            "is not supported yet."
        )

    parameters = variant_config.get("parameters", {})
    lookback_days = _require_parameter(parameters, "lookback_days")
    threshold = _require_parameter(parameters, "threshold")
    baseline_parameters = _baseline_rsi_parameters(project)
    change_summary = _read_change_summary(change_summary_path)

    prices = load_ohlcv_csv(data_csv)
    baseline_signals = generate_rsi_signals(
        prices,
        entry_rsi=baseline_parameters.get("entry_rsi", 30),
        exit_rsi=baseline_parameters.get("exit_rsi", 70),
        rsi_window=baseline_parameters.get("rsi_window", 14),
    )
    signals = apply_momentum_filter_to_signals(
        prices,
        baseline_signals,
        lookback_days=lookback_days,
        threshold=threshold,
    )
    positions = build_long_positions(signals)
    results = run_long_backtest(prices, positions)
    summary = summarize_backtest(results)
    metrics = {key: summary[key] for key in METRIC_KEYS}

    results_output = results.rename_axis("date").reset_index()
    results_output = results_output[
        ["date", "close", "position", "asset_return", "strategy_return", "equity"]
    ]
    results_output = results_output.sort_values("date")

    signals_output = signals.rename_axis("date").reset_index()
    signals_output = signals_output[
        [
            "date",
            "close",
            "rsi",
            "entry_signal",
            "exit_signal",
            "momentum_return",
            "momentum_entry_allowed",
        ]
    ]
    signals_output = signals_output.sort_values("date").dropna()

    results_output.to_csv(results_path, index=False)
    signals_output.to_csv(signals_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    change_summary["status"] = "backtested"
    change_summary_path.write_text(
        json.dumps(change_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _momentum_report_markdown(
            variant_id=variant_id,
            lookback_days=lookback_days,
            threshold=threshold,
            metrics=metrics,
            assumptions=change_summary.get("assumptions", []),
            warnings=change_summary.get("warnings", []),
        ),
        encoding="utf-8",
    )

    return metrics


def _require_parameter(parameters: dict, name: str):
    if name not in parameters:
        raise ValueError(f"ERROR: Missing required parameter '{name}' for momentum_filter")
    return parameters[name]


def _baseline_rsi_parameters(project: dict) -> dict:
    baseline_variant_id = project.get("baseline_variant_id", "baseline")
    for variant in project.get("variants", []):
        if isinstance(variant, dict) and variant.get("variant_id") == baseline_variant_id:
            return variant.get("parameters", {})
    return {}


def _read_change_summary(change_summary_path: Path) -> dict:
    if not change_summary_path.exists():
        return {"assumptions": [], "warnings": [], "status": "created_not_backtested"}
    return json.loads(change_summary_path.read_text(encoding="utf-8"))


def _momentum_report_markdown(
    variant_id: str,
    lookback_days: int,
    threshold: float,
    metrics: dict,
    assumptions: list[str],
    warnings: list[str],
) -> str:
    assumption_lines = [f"- {assumption}" for assumption in assumptions]
    warning_lines = [f"- {warning}" for warning in warnings]
    warning_lines.append("- Momentum behavior may differ across market regimes.")

    return "\n".join(
        [
            "# Strategy Report — Variant",
            "",
            "## Variant Summary",
            f"- Variant ID: {variant_id}",
            "- Parent: baseline",
            "- Modification: momentum_filter",
            "- Status: backtested",
            "",
            "## AI-Assisted Modification",
            "",
            "This variant was created from a validated AI-assisted modification specification.",
            "The AI did not generate executable strategy code.",
            "The deterministic QuantForge builder applied the validated modification.",
            "",
            "## Momentum Filter",
            "",
            "The entry rule requires recent momentum to exceed the configured threshold.",
            "",
            "Parameters:",
            f"- lookback_days: {lookback_days}",
            f"- threshold: {threshold}",
            "",
            "## Metrics",
            f"- Total Return: {metrics['total_return']:.6f}",
            f"- Annualized Return: {metrics['annualized_return']:.6f}",
            f"- Max Drawdown: {metrics['max_drawdown']:.6f}",
            f"- Exposure: {metrics['exposure']:.6f}",
            f"- Trade Count: {metrics['trade_count']}",
            f"- Win Rate: {metrics['win_rate']:.6f}",
            "",
            "## Assumptions",
            *assumption_lines,
            "",
            "## Warnings",
            *warning_lines,
            "",
            "## Interpretation (Non-Advisory)",
            "This strategy was evaluated on historical data only.",
            "Performance may not generalize to future market conditions.",
            "This analysis does not constitute financial advice.",
            "",
        ]
    )
