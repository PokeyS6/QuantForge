"""Strategy mutation helpers for QuantForge variants."""

import json
from pathlib import Path

import pandas as pd

from quantforge.backtest.engine import compute_volatility_regime_analysis, run_long_backtest
from quantforge.backtest.metrics import summarize_backtest
from quantforge.backtest.trades import build_long_positions
from quantforge.data.csv_loader import load_ohlcv_csv
from quantforge.ml.price_floor import (
    FEATURE_COLUMNS,
    apply_ml_price_floor_filter_to_signals,
    build_downside_risk_labels,
    build_ml_price_floor_features,
    compute_walk_forward_downside_risk_predictions,
)
from quantforge.strategies.rsi import generate_rsi_signals


VARIANT_ID = "variant_001_volatility_filter"
ML_PRICE_FLOOR_VARIANT_ID = "variant_002_ml_price_floor"
METRIC_KEYS = [
    "total_return",
    "annualized_return",
    "max_drawdown",
    "exposure",
    "trade_count",
    "win_rate",
]

STRATEGY_CONFIG = {
    "variant_id": VARIANT_ID,
    "parent_variant_id": "baseline",
    "modification_type": "volatility_filter",
    "volatility_window": 30,
    "volatility_threshold_quantile": 0.75,
    "entry_rule": "RSI < 30 AND rolling volatility <= threshold",
    "exit_rule": "RSI > 70",
}

CHANGE_SUMMARY = {
    "variant_id": VARIANT_ID,
    "parent_variant_id": "baseline",
    "user_instruction": "Add a volatility filter",
    "summary": "Adds a volatility filter that blocks RSI entries during unusually high-volatility periods.",
    "rationale": "Phase 2 diagnostics showed strategy behavior differs under high-volatility conditions.",
    "assumptions": [
        "Volatility is measured as the 30-day rolling standard deviation of daily returns.",
        "High volatility is defined as rolling volatility above the 75th percentile.",
        "The filter affects entries only; exits remain unchanged.",
    ],
    "status": "created_not_backtested",
}

ML_PRICE_FLOOR_STRATEGY_CONFIG = {
    "variant_id": ML_PRICE_FLOOR_VARIANT_ID,
    "parent_variant_id": "baseline",
    "modification_type": "ml_price_floor",
    "model_type": "logistic_regression",
    "prediction_target": "downside_risk",
    "forecast_horizon_days": 5,
    "downside_threshold": -0.03,
    "risk_probability_threshold": 0.5,
    "training_mode": "walk_forward_expanding",
    "min_training_rows": 252,
    "features": [
        "return_1d",
        "return_5d",
        "return_10d",
        "rolling_volatility_30d",
        "distance_from_sma_20",
    ],
    "entry_rule": "RSI < 30 AND predicted downside risk probability <= 0.5",
    "exit_rule": "RSI > 70",
}

ML_PRICE_FLOOR_CHANGE_SUMMARY = {
    "variant_id": ML_PRICE_FLOOR_VARIANT_ID,
    "parent_variant_id": "baseline",
    "user_instruction": "Add ML price floor",
    "summary": "Adds an ML downside-risk filter that blocks RSI entries when predicted downside risk is above the configured probability threshold.",
    "rationale": "Adds a hardcoded reference ML filter for the final demo while preserving auditability and no-lookahead constraints.",
    "assumptions": [
        "Features use historical close-price data only.",
        "The model is trained with walk-forward expanding windows.",
        "The filter affects entries only; exits remain unchanged.",
    ],
    "status": "created_not_backtested",
}

DIFF_MARKDOWN = """# Variant Diff — variant_001_volatility_filter

Parent: baseline

## Summary

Adds a volatility filter to the baseline RSI entry rule.

## Baseline Entry Rule

Enter long when:

RSI < 30

## Variant Entry Rule

Enter long when:

RSI < 30  
AND rolling 30-day volatility <= 75th percentile volatility threshold

## Exit Rule

Unchanged:

RSI > 70

## Implementation Note

Strategy logic is currently implemented within the QuantForge analysis pipeline.
No separate strategy code file exists at this stage.

## Backtest Status

This variant has been created but not backtested yet.
"""

ML_PRICE_FLOOR_DIFF_MARKDOWN = """# Variant Diff — variant_002_ml_price_floor

Parent: baseline

## Summary

Adds an ML downside-risk filter to the baseline RSI entry rule.

## Baseline Entry Rule

Enter long when:

RSI < 30

## Variant Entry Rule

Enter long when:

RSI < 30  
AND predicted downside risk probability <= 0.5

## Exit Rule

Unchanged:

RSI > 70

## Implementation Note

Strategy logic is implemented in the QuantForge analysis pipeline.
No separate strategy code file exists at this stage.

## Backtest Status

This variant has been created but not backtested yet.
"""


def apply_volatility_filter_to_signals(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    volatility_window: int,
    volatility_threshold_quantile: float,
) -> pd.DataFrame:
    """Apply a no-lookahead volatility entry filter to RSI signals."""
    required_signal_columns = {"close", "rsi", "entry_signal", "exit_signal"}
    if "close" not in prices.columns:
        raise ValueError("Prices must include a close column.")
    missing_signal_columns = required_signal_columns - set(signals.columns)
    if missing_signal_columns:
        raise ValueError("Signals must include close, rsi, entry_signal, and exit_signal columns.")
    if volatility_window <= 0:
        raise ValueError("volatility_window must be greater than 0.")
    if not 0 <= volatility_threshold_quantile <= 1:
        raise ValueError("volatility_threshold_quantile must be between 0 and 1.")

    returns = prices["close"].pct_change()
    volatility = returns.rolling(volatility_window).std()
    threshold = volatility.expanding().quantile(volatility_threshold_quantile)
    volatility = volatility.reindex(signals.index)
    threshold = threshold.reindex(signals.index)

    entries_allowed = volatility.notna() & threshold.notna() & (volatility <= threshold)
    filtered = signals[["close", "rsi", "entry_signal", "exit_signal"]].copy()
    filtered["entry_signal"] = filtered["entry_signal"] & entries_allowed
    return filtered


def _variant_report_markdown(
    variant_config: dict,
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
    zero_trade_note = (
        [
            "",
            "The volatility filter removed all entry signals; the strategy did not take any positions.",
        ]
        if metrics["trade_count"] == 0
        else []
    )

    return "\n".join(
        [
            "# Strategy Report — Variant",
            "",
            "## Variant Summary",
            f"- Variant ID: {variant_config['variant_id']}",
            f"- Parent: {variant_config['parent_variant_id']}",
            f"- Modification: {variant_config['modification_type']}",
            "- Status: backtested",
            "",
            "## Change Summary",
            "Adds a volatility filter that blocks RSI entries during unusually high-volatility periods.",
            "",
            "## Strategy Code",
            "Strategy logic is implemented in the QuantForge analysis pipeline.",
            "See diff.md for modification details.",
            "",
            "## Assumptions",
            "- Long-only",
            f"- Volatility filter uses {variant_config['volatility_window']}-day rolling volatility",
            "- Threshold is computed using only past data (no lookahead)",
            "- No slippage modeled",
            "",
            "The volatility threshold is computed using only historical data available up to each point in time.",
            "",
            "## Metrics",
            f"- Total Return: {metrics['total_return']:.6f}",
            f"- Annualized Return: {metrics['annualized_return']:.6f}",
            f"- Max Drawdown: {metrics['max_drawdown']:.6f}",
            f"- Exposure: {metrics['exposure']:.6f}",
            f"- Trade Count: {metrics['trade_count']}",
            f"- Win Rate: {metrics['win_rate']:.6f}",
            *zero_trade_note,
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


def _baseline_rsi_parameters(project: dict) -> dict:
    baseline_variant_id = project.get("baseline_variant_id", "baseline")
    for variant in project.get("variants", []):
        if (
            isinstance(variant, dict)
            and variant.get("variant_id") == baseline_variant_id
            and variant.get("strategy_type") == "rsi_reversal"
            and isinstance(variant.get("parameters"), dict)
        ):
            return variant["parameters"]
    raise ValueError("Baseline RSI parameters not found in project metadata.")


def _require_variant_config_field(variant_config: dict, field: str):
    if field not in variant_config:
        raise ValueError(f"Variant config missing required field: {field}")
    return variant_config[field]


def _metric_subset(summary: dict) -> dict:
    return {key: summary[key] for key in METRIC_KEYS}


def _final_ml_coefficients(features: pd.DataFrame, labels: pd.Series, min_training_rows: int):
    training_data = features.copy()
    training_data["downside_risk"] = labels
    training_data = training_data.dropna()

    if len(training_data) < min_training_rows:
        return None

    y_train = training_data["downside_risk"]
    if y_train.nunique() < 2:
        return None

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(training_data[features.columns], y_train)
    return dict(zip(features.columns, model.coef_[0]))


def _format_percent(value: float) -> str:
    return f"{value * 100:g}%"


def _regime_report_lines(regime_analysis: dict) -> list[str]:
    high_volatility = regime_analysis["high_volatility"]
    normal_volatility = regime_analysis["normal_volatility"]
    return [
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
    ]


def _ml_price_floor_report_markdown(
    variant_config: dict,
    metrics: dict,
    regime_analysis: dict,
    coefficients: dict | None,
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
    coefficient_lines = (
        ["No model coefficients available because there was insufficient training data."]
        if coefficients is None
        else [
            "Feature | Coefficient",
            "--- | ---",
            *[
                f"{feature} | {coefficients[feature]:.6f}"
                for feature in FEATURE_COLUMNS
            ],
        ]
    )
    zero_trade_note = (
        [
            "",
            "The ML price floor removed all entry signals; the strategy did not take any positions.",
        ]
        if metrics["trade_count"] == 0
        else []
    )

    return "\n".join(
        [
            "# Strategy Report — Variant",
            "",
            "## Variant Summary",
            f"- Variant ID: {variant_config['variant_id']}",
            f"- Parent: {variant_config['parent_variant_id']}",
            f"- Modification: {variant_config['modification_type']}",
            "- Status: backtested",
            "",
            "## Change Summary",
            "Adds an ML downside-risk filter that blocks RSI entries when predicted downside risk is above the configured probability threshold.",
            "",
            "## Strategy Code",
            "Strategy logic is implemented in the QuantForge analysis pipeline.",
            "See diff.md for modification details.",
            "",
            "## ML Explanation",
            "The ML price floor blocks RSI entries when walk-forward downside-risk probability is above the configured threshold.",
            "",
            "## Assumptions",
            "- Long-only",
            f"- Model: {variant_config['model_type'].replace('_', ' ')}",
            f"- Training: {variant_config['training_mode'].replace('_', ' ')}",
            "- No future data used",
            f"- horizon = {variant_config['forecast_horizon_days']}",
            f"- threshold = {_format_percent(variant_config['downside_threshold'])}",
            f"- probability cutoff = {variant_config['risk_probability_threshold']}",
            f"- min training rows = {variant_config['min_training_rows']}",
            "",
            "The downside-risk prediction is computed using only historical data available before each prediction point.",
            "",
            "## Metrics",
            f"- Total Return: {metrics['total_return']:.6f}",
            f"- Annualized Return: {metrics['annualized_return']:.6f}",
            f"- Max Drawdown: {metrics['max_drawdown']:.6f}",
            f"- Exposure: {metrics['exposure']:.6f}",
            f"- Trade Count: {metrics['trade_count']}",
            f"- Win Rate: {metrics['win_rate']:.6f}",
            *zero_trade_note,
            "",
            "## ML Feature Coefficients",
            *coefficient_lines,
            "",
            "## ML Warnings",
            "- The model may overfit historical data.",
            "- The model may reduce trade count.",
            "- The model provides no predictive guarantee.",
            "",
            "## Transparency Note",
            "This filter is a hardcoded reference implementation.",
            "Future versions will support natural language strategy modifications.",
            "",
            "## Insufficient Data Note",
            "When insufficient training data exists, entries are blocked because the filter condition cannot be satisfied.",
            "",
            *_regime_report_lines(regime_analysis),
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


def run_and_persist_volatility_filter_variant_analysis(
    project_file: Path,
    data_csv: Path,
) -> dict:
    """Run and persist the volatility-filter variant backtest."""
    project_root = project_file.parent
    baseline_results_path = project_root / "variants" / "baseline" / "backtest_results.csv"
    variant_dir = project_root / "variants" / VARIANT_ID
    variant_config_path = variant_dir / "strategy_config.json"
    results_path = variant_dir / "backtest_results.csv"
    signals_path = variant_dir / "signals.csv"
    metrics_path = variant_dir / "metrics.json"
    report_path = variant_dir / "report.md"

    if not baseline_results_path.exists():
        raise ValueError("ERROR: Baseline analysis not found. Run 'quantforge analyze' first.")
    if not variant_dir.exists():
        raise ValueError(f"ERROR: Variant {VARIANT_ID} not found.")
    if not variant_config_path.exists():
        raise ValueError(f"Variant config not found: {variant_config_path}")
    if results_path.exists() or metrics_path.exists() or report_path.exists():
        raise ValueError(
            f"ERROR: Variant analysis already exists for {VARIANT_ID}. Refusing to overwrite."
        )

    project = json.loads(project_file.read_text(encoding="utf-8"))
    variant_config = json.loads(variant_config_path.read_text(encoding="utf-8"))
    baseline_parameters = _baseline_rsi_parameters(project)
    volatility_window = _require_variant_config_field(variant_config, "volatility_window")
    volatility_threshold_quantile = _require_variant_config_field(
        variant_config,
        "volatility_threshold_quantile",
    )

    prices = load_ohlcv_csv(data_csv)
    baseline_signals = generate_rsi_signals(
        prices,
        entry_rsi=baseline_parameters.get("entry_rsi", 30),
        exit_rsi=baseline_parameters.get("exit_rsi", 70),
        rsi_window=baseline_parameters.get("rsi_window", 14),
    )
    signals = apply_volatility_filter_to_signals(
        prices,
        baseline_signals,
        volatility_window=volatility_window,
        volatility_threshold_quantile=volatility_threshold_quantile,
    )
    positions = build_long_positions(signals)
    results = run_long_backtest(prices, positions)
    summary = summarize_backtest(results)
    metrics = _metric_subset(summary)
    regime_analysis = compute_volatility_regime_analysis(results)

    results_output = results.rename_axis("date").reset_index()
    results_output = results_output[
        ["date", "close", "position", "asset_return", "strategy_return", "equity"]
    ]
    results_output = results_output.sort_values("date")
    results_output.to_csv(results_path, index=False)

    signals_output = signals.rename_axis("date").reset_index()
    signals_output = signals_output[["date", "close", "rsi", "entry_signal", "exit_signal"]]
    signals_output = signals_output.sort_values("date").dropna()
    signals_output.to_csv(signals_path, index=False)

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    change_summary_path = variant_dir / "change_summary.json"
    change_summary = json.loads(change_summary_path.read_text(encoding="utf-8"))
    change_summary["status"] = "backtested"
    change_summary_path.write_text(
        json.dumps(change_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    report_path.write_text(
        _variant_report_markdown(
            variant_config=variant_config,
            metrics=metrics,
            regime_analysis=regime_analysis,
        ),
        encoding="utf-8",
    )

    return metrics


def run_and_persist_ml_price_floor_variant_analysis(
    project_file: Path,
    data_csv: Path,
) -> dict:
    """Run and persist the ML price-floor variant backtest."""
    project_root = project_file.parent
    baseline_results_path = project_root / "variants" / "baseline" / "backtest_results.csv"
    variant_dir = project_root / "variants" / ML_PRICE_FLOOR_VARIANT_ID
    variant_config_path = variant_dir / "strategy_config.json"
    results_path = variant_dir / "backtest_results.csv"
    signals_path = variant_dir / "signals.csv"
    metrics_path = variant_dir / "metrics.json"
    report_path = variant_dir / "report.md"

    if not baseline_results_path.exists():
        raise ValueError("ERROR: Baseline analysis not found. Run 'quantforge analyze' first.")
    if not variant_dir.exists():
        raise ValueError(f"ERROR: Variant {ML_PRICE_FLOOR_VARIANT_ID} not found.")
    if not variant_config_path.exists():
        raise ValueError(f"Variant config not found: {variant_config_path}")
    if results_path.exists() or metrics_path.exists() or report_path.exists():
        raise ValueError(
            f"ERROR: Variant analysis already exists for {ML_PRICE_FLOOR_VARIANT_ID}. "
            "Refusing to overwrite."
        )

    project = json.loads(project_file.read_text(encoding="utf-8"))
    variant_config = json.loads(variant_config_path.read_text(encoding="utf-8"))
    baseline_parameters = _baseline_rsi_parameters(project)
    forecast_horizon_days = _require_variant_config_field(
        variant_config,
        "forecast_horizon_days",
    )
    downside_threshold = _require_variant_config_field(variant_config, "downside_threshold")
    min_training_rows = _require_variant_config_field(variant_config, "min_training_rows")
    risk_probability_threshold = _require_variant_config_field(
        variant_config,
        "risk_probability_threshold",
    )

    prices = load_ohlcv_csv(data_csv)
    baseline_signals = generate_rsi_signals(
        prices,
        entry_rsi=baseline_parameters.get("entry_rsi", 30),
        exit_rsi=baseline_parameters.get("exit_rsi", 70),
        rsi_window=baseline_parameters.get("rsi_window", 14),
    )
    features = build_ml_price_floor_features(prices)
    labels = build_downside_risk_labels(
        prices,
        forecast_horizon_days=forecast_horizon_days,
        downside_threshold=downside_threshold,
    )
    p_downside_risk = compute_walk_forward_downside_risk_predictions(
        features,
        labels,
        min_training_rows=min_training_rows,
    )
    signals = apply_ml_price_floor_filter_to_signals(
        baseline_signals,
        p_downside_risk,
        risk_probability_threshold=risk_probability_threshold,
    )
    positions = build_long_positions(signals)
    results = run_long_backtest(prices, positions)
    summary = summarize_backtest(results)
    metrics = _metric_subset(summary)
    regime_analysis = compute_volatility_regime_analysis(results)
    coefficients = _final_ml_coefficients(features, labels, min_training_rows)

    results_output = results.rename_axis("date").reset_index()
    results_output = results_output[
        ["date", "close", "position", "asset_return", "strategy_return", "equity"]
    ]
    results_output = results_output.sort_values("date")
    results_output.to_csv(results_path, index=False)

    signals_output = signals.rename_axis("date").reset_index()
    signals_output = signals_output[
        [
            "date",
            "close",
            "rsi",
            "entry_signal",
            "exit_signal",
            "p_downside_risk",
            "ml_entry_allowed",
        ]
    ]
    signals_output = signals_output.sort_values("date").dropna()
    signals_output.to_csv(signals_path, index=False)

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    change_summary_path = variant_dir / "change_summary.json"
    change_summary = json.loads(change_summary_path.read_text(encoding="utf-8"))
    change_summary["status"] = "backtested"
    change_summary_path.write_text(
        json.dumps(change_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    report_path.write_text(
        _ml_price_floor_report_markdown(
            variant_config=variant_config,
            metrics=metrics,
            regime_analysis=regime_analysis,
            coefficients=coefficients,
        ),
        encoding="utf-8",
    )

    return metrics


def create_volatility_filter_variant(project_file: Path, user_instruction: str) -> Path:
    """Create placeholder artifacts for a volatility-filter variant."""
    project_root = project_file.parent
    baseline_results_path = project_root / "variants" / "baseline" / "backtest_results.csv"
    variant_dir = project_root / "variants" / VARIANT_ID

    if not baseline_results_path.exists():
        raise ValueError("ERROR: Baseline analysis not found. Run 'quantforge analyze' first.")
    if "volatility filter" not in user_instruction.lower():
        raise ValueError(
            "ERROR: Unsupported modification. Only 'volatility filter' is supported."
        )
    if variant_dir.exists():
        raise ValueError(
            f"ERROR: Variant {VARIANT_ID} already exists. Refusing to overwrite."
        )

    variant_dir.mkdir(parents=True)
    (variant_dir / "strategy_config.json").write_text(
        json.dumps(STRATEGY_CONFIG, indent=2) + "\n",
        encoding="utf-8",
    )
    (variant_dir / "change_summary.json").write_text(
        json.dumps(CHANGE_SUMMARY, indent=2) + "\n",
        encoding="utf-8",
    )
    (variant_dir / "diff.md").write_text(DIFF_MARKDOWN, encoding="utf-8")

    project = json.loads(project_file.read_text(encoding="utf-8"))
    variants = project.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError("Project metadata 'variants' must be a list.")
    if VARIANT_ID not in variants:
        variants.append(VARIANT_ID)
    project["variants"] = variants
    project_file.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")

    return variant_dir


def create_ml_price_floor_variant(project_file: Path, user_instruction: str) -> Path:
    """Create placeholder artifacts for an ML price-floor variant."""
    project_root = project_file.parent
    baseline_results_path = project_root / "variants" / "baseline" / "backtest_results.csv"
    variant_dir = project_root / "variants" / ML_PRICE_FLOOR_VARIANT_ID

    if "ml price floor" not in user_instruction.lower():
        raise ValueError("ERROR: Unsupported modification. Only 'ml price floor' is supported.")
    if not baseline_results_path.exists():
        raise ValueError("ERROR: Baseline analysis not found. Run 'quantforge analyze' first.")
    if variant_dir.exists():
        raise ValueError(
            f"ERROR: Variant {ML_PRICE_FLOOR_VARIANT_ID} already exists. Refusing to overwrite."
        )

    variant_dir.mkdir(parents=True)
    (variant_dir / "strategy_config.json").write_text(
        json.dumps(ML_PRICE_FLOOR_STRATEGY_CONFIG, indent=2) + "\n",
        encoding="utf-8",
    )
    (variant_dir / "change_summary.json").write_text(
        json.dumps(ML_PRICE_FLOOR_CHANGE_SUMMARY, indent=2) + "\n",
        encoding="utf-8",
    )
    (variant_dir / "diff.md").write_text(
        ML_PRICE_FLOOR_DIFF_MARKDOWN,
        encoding="utf-8",
    )

    project = json.loads(project_file.read_text(encoding="utf-8"))
    variants = project.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError("Project metadata 'variants' must be a list.")
    if ML_PRICE_FLOOR_VARIANT_ID not in variants:
        variants.append(ML_PRICE_FLOOR_VARIANT_ID)
    project["variants"] = variants
    project_file.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")

    return variant_dir
