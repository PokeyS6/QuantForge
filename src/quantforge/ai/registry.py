"""Deterministic AI modification registry constants."""


REGISTRY = {
    "volatility_filter": {
        "parameters": {
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
        },
        "data_requirements": ["close"],
        "leakage_classification": "historical_only",
    },
    "momentum_filter": {
        "parameters": {
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
        },
        "data_requirements": ["close"],
        "leakage_classification": "historical_only",
    },
    "moving_average_confirmation": {
        "parameters": {
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
        },
        "data_requirements": ["close"],
        "leakage_classification": "historical_only",
    },
    "rsi_threshold_adjustment": {
        "parameters": {
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
        },
        "data_requirements": ["close"],
        "leakage_classification": "historical_only",
    },
}
