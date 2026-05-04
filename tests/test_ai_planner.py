import pytest

from quantforge.ai.planner import build_planner_prompt


REQUIRED_CONSTRAINTS = [
    "You are the QuantForge AI modification planner.",
    "You are not a trading advisor.",
    "You do not recommend trades.",
    "You do not optimize strategies.",
    "You do not generate Python code.",
    "Return JSON only.",
    "Do not use markdown.",
    "Do not use code fences.",
    "supported=false",
]


def test_prompt_contains_required_constraints():
    prompt = build_planner_prompt("Add a volatility filter")

    for phrase in REQUIRED_CONSTRAINTS:
        assert phrase in prompt
    assert "local_llm_v1" in prompt


def test_prompt_contains_all_four_modification_types():
    prompt = build_planner_prompt("Add a momentum filter")

    assert "volatility_filter" in prompt
    assert "momentum_filter" in prompt
    assert "moving_average_confirmation" in prompt
    assert "rsi_threshold_adjustment" in prompt


def test_prompt_contains_supported_example():
    prompt = build_planner_prompt("Add a volatility filter")

    assert "Supported JSON example:" in prompt
    assert '"supported": true' in prompt
    assert '"ai_planner_version": "local_llm_v1"' in prompt
    assert '"modification_type": "momentum_filter"' in prompt
    assert '"user_instruction": "Add a momentum filter"' in prompt
    assert '"entry_rule_change": "Add entry filter: return_20d > 0"' in prompt
    assert (
        '"non_advisory_note": "This modification creates a testable strategy variant and does not constitute trading advice."'
        in prompt
    )


def test_prompt_contains_unsupported_example():
    prompt = build_planner_prompt("Connect to a broker")

    assert "Unsupported JSON example:" in prompt
    assert '"supported": false' in prompt
    assert '"ai_planner_version": "local_llm_v1"' in prompt
    assert '"user_instruction": "Optimize this strategy for max returns"' in prompt
    assert (
        '"reason": "Optimization requests are not supported because QuantForge does not silently search for best-performing parameters."'
        in prompt
    )
    assert '"Change RSI threshold"' in prompt


def test_prompt_contains_unsupported_categories():
    prompt = build_planner_prompt("Optimize returns")

    assert "- optimization" in prompt
    assert "- ranking strategies" in prompt
    assert "- finding best parameters" in prompt
    assert "- buy/sell recommendations" in prompt
    assert "- unsupported indicators" in prompt
    assert "- broker/live trading" in prompt
    assert "- multi-asset portfolio requests" in prompt
    assert "- custom Python generation" in prompt


def test_prompt_contains_registry_defaults_and_bounds():
    prompt = build_planner_prompt("Adjust RSI thresholds")

    assert (
        "- volatility_filter: parameters volatility_window integer default 30 min 5 max 100; "
        "threshold_quantile number default 0.75 min 0.5 max 0.95"
        in prompt
    )
    assert (
        "- momentum_filter: parameters lookback_days integer default 20 min 5 max 100; "
        "threshold number default 0.0 min -0.2 max 0.2"
        in prompt
    )
    assert (
        "- moving_average_confirmation: parameters sma_window integer default 50 min 5 max 200; "
        "direction enum allowed [above, below] default above"
        in prompt
    )
    assert (
        "- rsi_threshold_adjustment: parameters entry_rsi number default 30 min 10 max 45 optional true; "
        "exit_rsi number default 70 min 55 max 90 optional true"
        in prompt
    )


def test_prompt_includes_user_instruction_and_parent_variant_id():
    prompt = build_planner_prompt(
        "Add moving average confirmation",
        parent_variant_id="variant_existing",
    )

    assert "user_instruction: Add moving average confirmation" in prompt
    assert "parent_variant_id: variant_existing" in prompt
    assert '"parent_variant_id": "variant_existing"' in prompt


def test_invalid_user_instruction_raises():
    with pytest.raises(ValueError, match="user_instruction must be a non-empty string"):
        build_planner_prompt("")
    with pytest.raises(ValueError, match="user_instruction must be a non-empty string"):
        build_planner_prompt(None)


def test_invalid_parent_variant_id_raises():
    with pytest.raises(ValueError, match="parent_variant_id must be a non-empty string"):
        build_planner_prompt("Add a volatility filter", parent_variant_id="")
    with pytest.raises(ValueError, match="parent_variant_id must be a non-empty string"):
        build_planner_prompt("Add a volatility filter", parent_variant_id=None)
