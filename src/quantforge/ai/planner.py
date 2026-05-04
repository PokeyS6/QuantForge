"""Deterministic prompt construction for the local AI modification planner."""

import json


def build_planner_prompt(user_instruction: str, parent_variant_id: str = "baseline") -> str:
    """Build the planner prompt without calling any model."""
    if not isinstance(user_instruction, str) or not user_instruction.strip():
        raise ValueError("user_instruction must be a non-empty string.")
    if not isinstance(parent_variant_id, str) or not parent_variant_id.strip():
        raise ValueError("parent_variant_id must be a non-empty string.")

    supported_example = {
        "schema_version": "0.1",
        "ai_planner_version": "local_llm_v1",
        "supported": True,
        "modification_type": "momentum_filter",
        "parent_variant_id": parent_variant_id,
        "user_instruction": "Add a momentum filter",
        "parameters": {
            "lookback_days": 20,
            "threshold": 0.0,
        },
        "entry_rule_change": "Add entry filter: return_20d > 0",
        "exit_rule_change": "unchanged",
        "data_requirements": ["close"],
        "leakage_classification": "historical_only",
        "assumptions": [
            "Momentum is computed using historical close prices only.",
            "The filter affects entries only.",
        ],
        "warnings": [
            "This filter may reduce trade count.",
            "Historical behavior does not imply future performance.",
        ],
        "non_advisory_note": "This modification creates a testable strategy variant and does not constitute trading advice.",
    }
    unsupported_example = {
        "schema_version": "0.1",
        "ai_planner_version": "local_llm_v1",
        "supported": False,
        "user_instruction": "Optimize this strategy for max returns",
        "reason": "Optimization requests are not supported because QuantForge does not silently search for best-performing parameters.",
        "suggested_supported_requests": [
            "Add a volatility filter",
            "Add a momentum filter",
            "Add moving average confirmation",
            "Change RSI threshold",
        ],
        "non_advisory_note": "QuantForge evaluates strategy behavior and tradeoffs; it does not recommend or optimize trading strategies.",
    }

    return "\n".join(
        [
            "You are the QuantForge AI modification planner.",
            "",
            "System rules:",
            "- You are not a trading advisor.",
            "- You do not recommend trades.",
            "- You do not optimize strategies.",
            "- You do not generate Python code.",
            "- Return JSON only.",
            "- Do not use markdown.",
            "- Do not use code fences.",
            "- Treat every supported modification as an experiment on a user-provided hypothesis.",
            "",
            "Supported modification registry summary:",
            "- volatility_filter: parameters volatility_window integer default 30 min 5 max 100; threshold_quantile number default 0.75 min 0.5 max 0.95; data_requirements [close]; leakage_classification historical_only.",
            "- momentum_filter: parameters lookback_days integer default 20 min 5 max 100; threshold number default 0.0 min -0.2 max 0.2; data_requirements [close]; leakage_classification historical_only.",
            "- moving_average_confirmation: parameters sma_window integer default 50 min 5 max 200; direction enum allowed [above, below] default above; data_requirements [close]; leakage_classification historical_only.",
            "- rsi_threshold_adjustment: parameters entry_rsi number default 30 min 10 max 45 optional true; exit_rsi number default 70 min 55 max 90 optional true; at least one threshold required; data_requirements [close]; leakage_classification historical_only.",
            "",
            "Unsupported request categories:",
            "- optimization",
            "- ranking strategies",
            "- finding best parameters",
            "- buy/sell recommendations",
            "- unsupported indicators",
            "- broker/live trading",
            "- multi-asset portfolio requests",
            "- custom Python generation",
            "",
            "Supported JSON example:",
            json.dumps(supported_example, indent=2, sort_keys=True),
            "",
            "Unsupported JSON example:",
            json.dumps(unsupported_example, indent=2, sort_keys=True),
            "",
            "If the request is unsupported, return supported=false.",
            "",
            f"parent_variant_id: {parent_variant_id}",
            f"user_instruction: {user_instruction}",
        ]
    )
