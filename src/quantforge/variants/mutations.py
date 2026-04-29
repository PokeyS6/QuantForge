"""Strategy mutation helpers for QuantForge variants."""

import json
from pathlib import Path

import pandas as pd


VARIANT_ID = "variant_001_volatility_filter"

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
