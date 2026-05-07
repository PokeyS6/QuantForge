"""Registry validation for schema-valid AI modification specs."""

from quantforge.ai.registry import REGISTRY
from quantforge.ai.schema import ModificationSpecValidationError


def validate_against_registry(spec: dict) -> None:
    """Validate a schema-valid AI modification spec against the registry."""
    if spec["supported"] is False:
        return

    modification_type = spec["modification_type"]
    if modification_type not in REGISTRY:
        raise ModificationSpecValidationError(
            f"Unsupported modification type '{modification_type}'"
        )

    registry_entry = REGISTRY[modification_type]
    _validate_parameters(
        modification_type=modification_type,
        parameters=spec["parameters"],
        registry_parameters=registry_entry["parameters"],
    )

    if spec["data_requirements"] != registry_entry["data_requirements"]:
        raise ModificationSpecValidationError("data_requirements do not match registry.")
    if spec["leakage_classification"] != registry_entry["leakage_classification"]:
        raise ModificationSpecValidationError(
            "leakage_classification does not match registry."
        )


def _validate_parameters(
    modification_type: str,
    parameters: dict,
    registry_parameters: dict,
) -> None:
    unknown_parameters = set(parameters) - set(registry_parameters)
    if unknown_parameters:
        parameter = sorted(unknown_parameters)[0]
        raise ModificationSpecValidationError(f"Unknown parameter '{parameter}'")

    if modification_type == "rsi_threshold_adjustment" and not (
        "entry_rsi" in parameters or "exit_rsi" in parameters
    ):
        raise ModificationSpecValidationError(
            "rsi_threshold_adjustment requires entry_rsi or exit_rsi."
        )

    for parameter_name, parameter_spec in registry_parameters.items():
        if parameter_spec.get("optional") and parameter_name not in parameters:
            continue
        if parameter_name not in parameters:
            raise ModificationSpecValidationError(
                f"Missing required parameter '{parameter_name}'"
            )

        value = parameters[parameter_name]
        _validate_parameter_type(parameter_name, value, parameter_spec)
        _validate_parameter_bounds(parameter_name, value, parameter_spec)


def _validate_parameter_type(parameter_name: str, value, parameter_spec: dict) -> None:
    parameter_type = parameter_spec["type"]
    if parameter_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ModificationSpecValidationError(
                f"{parameter_name} must be an integer."
            )
        return

    if parameter_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModificationSpecValidationError(f"{parameter_name} must be a number.")
        return

    if parameter_type == "enum":
        if value not in parameter_spec["allowed"]:
            raise ModificationSpecValidationError(
                f"{parameter_name} must be one of {parameter_spec['allowed']}."
            )
        return

    raise ModificationSpecValidationError(f"Unsupported parameter type: {parameter_type}")


def _validate_parameter_bounds(parameter_name: str, value, parameter_spec: dict) -> None:
    if "min" in parameter_spec and value < parameter_spec["min"]:
        raise ModificationSpecValidationError(
            f"Parameter '{parameter_name}' is out of bounds "
            f"(expected {_format_bounds_expectation(parameter_spec)}, got {value})"
        )
    if "max" in parameter_spec and value > parameter_spec["max"]:
        raise ModificationSpecValidationError(
            f"Parameter '{parameter_name}' is out of bounds "
            f"(expected {_format_bounds_expectation(parameter_spec)}, got {value})"
        )


def _format_bounds_expectation(parameter_spec: dict) -> str:
    if "min" in parameter_spec and "max" in parameter_spec:
        return f"{parameter_spec['min']}–{parameter_spec['max']}"
    if "min" in parameter_spec:
        return f">= {parameter_spec['min']}"
    if "max" in parameter_spec:
        return f"<= {parameter_spec['max']}"
    return "unbounded"
