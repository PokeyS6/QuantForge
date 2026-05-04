from quantforge.ai.registry import REGISTRY


SUPPORTED_TYPES = {
    "volatility_filter",
    "momentum_filter",
    "moving_average_confirmation",
    "rsi_threshold_adjustment",
}


def test_registry_contains_exact_supported_modification_types():
    assert set(REGISTRY) == SUPPORTED_TYPES


def test_each_registry_entry_has_required_sections():
    for entry in REGISTRY.values():
        assert set(entry) == {
            "parameters",
            "data_requirements",
            "leakage_classification",
        }


def test_each_data_requirement_is_close_only():
    for entry in REGISTRY.values():
        assert entry["data_requirements"] == ["close"]


def test_each_leakage_classification_is_historical_only():
    for entry in REGISTRY.values():
        assert entry["leakage_classification"] == "historical_only"


def test_volatility_filter_parameter_definitions_match_exact_values():
    assert REGISTRY["volatility_filter"]["parameters"] == {
        "volatility_window": {
            "type": "integer",
            "default": 30,
            "min": 5,
            "max": 100,
        },
        "threshold_quantile": {
            "type": "number",
            "default": 0.75,
            "min": 0.5,
            "max": 0.95,
        },
    }


def test_momentum_filter_parameter_definitions_match_exact_values():
    assert REGISTRY["momentum_filter"]["parameters"] == {
        "lookback_days": {
            "type": "integer",
            "default": 20,
            "min": 5,
            "max": 100,
        },
        "threshold": {
            "type": "number",
            "default": 0.0,
            "min": -0.2,
            "max": 0.2,
        },
    }


def test_moving_average_confirmation_parameter_definitions_match_exact_values():
    assert REGISTRY["moving_average_confirmation"]["parameters"] == {
        "sma_window": {
            "type": "integer",
            "default": 50,
            "min": 5,
            "max": 200,
        },
        "direction": {
            "type": "enum",
            "allowed": ["above", "below"],
            "default": "above",
        },
    }


def test_rsi_threshold_adjustment_parameter_definitions_match_exact_values():
    assert REGISTRY["rsi_threshold_adjustment"]["parameters"] == {
        "entry_rsi": {
            "type": "number",
            "default": 30,
            "min": 10,
            "max": 45,
            "optional": True,
        },
        "exit_rsi": {
            "type": "number",
            "default": 70,
            "min": 55,
            "max": 90,
            "optional": True,
        },
    }


def test_no_unsupported_modification_families_exist():
    assert "ml_price_floor" not in REGISTRY
    assert "broker_execution" not in REGISTRY
    assert "portfolio_optimization" not in REGISTRY
