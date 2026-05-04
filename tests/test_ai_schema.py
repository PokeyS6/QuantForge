import pytest

from quantforge.ai.schema import ModificationSpecValidationError, validate_schema


def supported_spec():
    return {
        "schema_version": "0.1",
        "ai_planner_version": "planner-0.1",
        "supported": True,
        "user_instruction": "Add a volatility filter",
        "non_advisory_note": "This tests a user-provided hypothesis.",
        "modification_type": "volatility_filter",
        "parent_variant_id": "baseline",
        "parameters": {"volatility_window": 30},
        "entry_rule_change": "Add volatility condition to entries.",
        "exit_rule_change": "No exit change.",
        "data_requirements": ["daily OHLCV data"],
        "leakage_classification": "no_lookahead",
        "assumptions": ["Entries are filtered using historical data."],
        "warnings": ["Historical behavior may not generalize."],
    }


def unsupported_spec():
    return {
        "schema_version": "0.1",
        "ai_planner_version": "planner-0.1",
        "supported": False,
        "user_instruction": "Add broker execution",
        "non_advisory_note": "This is outside the supported local analysis scope.",
        "reason": "Broker execution is not supported.",
        "suggested_supported_requests": ["Add a volatility filter"],
    }


def test_valid_supported_spec_passes():
    validate_schema(supported_spec())


def test_valid_unsupported_spec_passes():
    validate_schema(unsupported_spec())


def test_missing_required_field_fails():
    spec = supported_spec()
    del spec["parameters"]

    with pytest.raises(ModificationSpecValidationError, match="Missing required field"):
        validate_schema(spec)


def test_wrong_schema_version_fails():
    spec = supported_spec()
    spec["schema_version"] = "1.0"

    with pytest.raises(
        ModificationSpecValidationError,
        match="schema_version must equal '0.1'",
    ):
        validate_schema(spec)


def test_supported_not_boolean_fails():
    spec = supported_spec()
    spec["supported"] = "true"

    with pytest.raises(ModificationSpecValidationError, match="supported must be a boolean"):
        validate_schema(spec)


def test_forbidden_advisory_phrase_fails():
    spec = supported_spec()
    spec["non_advisory_note"] = "This is the best strategy."

    with pytest.raises(ModificationSpecValidationError, match="Forbidden advisory phrase"):
        validate_schema(spec)


def test_nested_forbidden_advisory_phrase_fails():
    spec = supported_spec()
    spec["parameters"]["description"] = ["This will outperform the baseline."]

    with pytest.raises(ModificationSpecValidationError, match="Forbidden advisory phrase"):
        validate_schema(spec)
