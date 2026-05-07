import pytest

from quantforge.ai.schema import ModificationSpecValidationError
from quantforge.ai.validator import validate_against_registry


def supported_spec(modification_type: str, parameters: dict) -> dict:
    return {
        "schema_version": "0.1",
        "ai_planner_version": "planner-0.1",
        "supported": True,
        "user_instruction": f"Add {modification_type}",
        "non_advisory_note": "This tests a user-provided hypothesis.",
        "modification_type": modification_type,
        "parent_variant_id": "baseline",
        "parameters": parameters,
        "entry_rule_change": "Change entry rule.",
        "exit_rule_change": "No exit change.",
        "data_requirements": ["close"],
        "leakage_classification": "historical_only",
        "assumptions": ["Uses historical close data."],
        "warnings": ["Historical behavior may not generalize."],
    }


def unsupported_spec() -> dict:
    return {
        "schema_version": "0.1",
        "ai_planner_version": "planner-0.1",
        "supported": False,
        "user_instruction": "Add broker execution",
        "non_advisory_note": "This is outside local analysis scope.",
        "reason": "Broker execution is not supported.",
        "suggested_supported_requests": ["Add a volatility filter"],
    }


def test_valid_momentum_filter_passes():
    spec = supported_spec(
        "momentum_filter",
        {"lookback_days": 20, "threshold": 0.0},
    )

    validate_against_registry(spec)


def test_valid_volatility_filter_passes():
    spec = supported_spec(
        "volatility_filter",
        {"volatility_window": 30, "threshold_quantile": 0.75},
    )

    validate_against_registry(spec)


def test_valid_moving_average_confirmation_passes():
    spec = supported_spec(
        "moving_average_confirmation",
        {"sma_window": 50, "direction": "above"},
    )

    validate_against_registry(spec)


def test_valid_rsi_threshold_adjustment_with_only_entry_rsi_passes():
    spec = supported_spec("rsi_threshold_adjustment", {"entry_rsi": 35})

    validate_against_registry(spec)


def test_valid_rsi_threshold_adjustment_with_only_exit_rsi_passes():
    spec = supported_spec("rsi_threshold_adjustment", {"exit_rsi": 75})

    validate_against_registry(spec)


def test_rsi_threshold_adjustment_with_neither_threshold_fails():
    spec = supported_spec("rsi_threshold_adjustment", {})

    with pytest.raises(
        ModificationSpecValidationError,
        match="requires entry_rsi or exit_rsi",
    ):
        validate_against_registry(spec)


def test_unsupported_spec_skips_registry_validation_and_passes():
    spec = unsupported_spec()

    validate_against_registry(spec)


def test_unknown_modification_type_fails():
    spec = supported_spec("broker_execution", {})

    with pytest.raises(
        ModificationSpecValidationError,
        match="Unsupported modification type 'broker_execution'",
    ):
        validate_against_registry(spec)


def test_unknown_parameter_fails():
    spec = supported_spec(
        "volatility_filter",
        {
            "volatility_window": 30,
            "threshold_quantile": 0.75,
            "extra": 1,
        },
    )

    with pytest.raises(ModificationSpecValidationError, match="Unknown parameter 'extra'"):
        validate_against_registry(spec)


def test_missing_required_parameter_fails():
    spec = supported_spec("momentum_filter", {"lookback_days": 20})

    with pytest.raises(
        ModificationSpecValidationError,
        match="Missing required parameter 'threshold'",
    ):
        validate_against_registry(spec)


def test_wrong_parameter_type_fails():
    spec = supported_spec(
        "volatility_filter",
        {"volatility_window": True, "threshold_quantile": 0.75},
    )

    with pytest.raises(
        ModificationSpecValidationError,
        match="volatility_window must be an integer",
    ):
        validate_against_registry(spec)


def test_out_of_bounds_parameter_fails():
    spec = supported_spec(
        "momentum_filter",
        {"lookback_days": 20, "threshold": 0.3},
    )

    with pytest.raises(
        ModificationSpecValidationError,
        match="Parameter 'threshold' is out of bounds",
    ):
        validate_against_registry(spec)


def test_invalid_enum_value_fails():
    spec = supported_spec(
        "moving_average_confirmation",
        {"sma_window": 50, "direction": "sideways"},
    )

    with pytest.raises(ModificationSpecValidationError, match="direction must be one of"):
        validate_against_registry(spec)


def test_leakage_mismatch_fails():
    spec = supported_spec(
        "volatility_filter",
        {"volatility_window": 30, "threshold_quantile": 0.75},
    )
    spec["leakage_classification"] = "uses_future_data"

    with pytest.raises(
        ModificationSpecValidationError,
        match="leakage_classification does not match registry",
    ):
        validate_against_registry(spec)


def test_data_requirement_mismatch_fails():
    spec = supported_spec(
        "volatility_filter",
        {"volatility_window": 30, "threshold_quantile": 0.75},
    )
    spec["data_requirements"] = ["close", "volume"]

    with pytest.raises(
        ModificationSpecValidationError,
        match="data_requirements do not match registry",
    ):
        validate_against_registry(spec)
