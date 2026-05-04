"""Deterministic schema validation for AI modification specs."""

from collections.abc import Iterable
from typing import Any


class ModificationSpecValidationError(ValueError):
    """Raised when an AI modification spec fails validation."""


BASE_REQUIRED_FIELDS = [
    "schema_version",
    "ai_planner_version",
    "supported",
    "user_instruction",
    "non_advisory_note",
]

SUPPORTED_REQUIRED_FIELDS = [
    "modification_type",
    "parent_variant_id",
    "parameters",
    "entry_rule_change",
    "exit_rule_change",
    "data_requirements",
    "leakage_classification",
    "assumptions",
    "warnings",
]

UNSUPPORTED_REQUIRED_FIELDS = [
    "reason",
    "suggested_supported_requests",
]

FORBIDDEN_ADVISORY_PHRASES = [
    "best strategy",
    "optimal strategy",
    "guaranteed",
    "should trade",
    "buy this",
    "sell this",
    "profitable",
    "will outperform",
    "safe trade",
]


def validate_schema(spec: dict) -> None:
    """Validate an AI modification spec without side effects."""
    if not isinstance(spec, dict):
        raise ModificationSpecValidationError("Spec must be a dict.")

    _require_fields(spec, BASE_REQUIRED_FIELDS)
    _validate_string(spec, "schema_version", allow_empty=False)
    if spec["schema_version"] != "0.1":
        raise ModificationSpecValidationError("schema_version must equal '0.1'.")

    _validate_string(spec, "ai_planner_version", allow_empty=False)
    _validate_bool(spec, "supported")
    _validate_string(spec, "user_instruction", allow_empty=False)
    _validate_string(spec, "non_advisory_note", allow_empty=False)

    if spec["supported"]:
        _require_fields(spec, SUPPORTED_REQUIRED_FIELDS)
        _validate_dict(spec, "parameters")
        _validate_list_of_strings(spec, "data_requirements")
        _validate_list_of_strings(spec, "assumptions")
        _validate_list_of_strings(spec, "warnings")
    else:
        _require_fields(spec, UNSUPPORTED_REQUIRED_FIELDS)
        _validate_list_of_strings(spec, "suggested_supported_requests")

    _reject_forbidden_phrases(spec)


def _require_fields(spec: dict, fields: Iterable[str]) -> None:
    for field in fields:
        if field not in spec:
            raise ModificationSpecValidationError(f"Missing required field: {field}")


def _validate_string(spec: dict, field: str, allow_empty: bool) -> None:
    value = spec[field]
    if not isinstance(value, str):
        raise ModificationSpecValidationError(f"{field} must be a string.")
    if not allow_empty and not value.strip():
        raise ModificationSpecValidationError(f"{field} must be non-empty.")


def _validate_bool(spec: dict, field: str) -> None:
    if not isinstance(spec[field], bool):
        raise ModificationSpecValidationError(f"{field} must be a boolean.")


def _validate_dict(spec: dict, field: str) -> None:
    if not isinstance(spec[field], dict):
        raise ModificationSpecValidationError(f"{field} must be a dict.")


def _validate_list_of_strings(spec: dict, field: str) -> None:
    value = spec[field]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ModificationSpecValidationError(f"{field} must be a list of strings.")


def _reject_forbidden_phrases(value: Any) -> None:
    if isinstance(value, str):
        normalized = value.lower()
        for phrase in FORBIDDEN_ADVISORY_PHRASES:
            if phrase in normalized:
                raise ModificationSpecValidationError(
                    f"Forbidden advisory phrase found: {phrase}"
                )
        return

    if isinstance(value, dict):
        for nested_value in value.values():
            _reject_forbidden_phrases(nested_value)
        return

    if isinstance(value, list):
        for item in value:
            _reject_forbidden_phrases(item)
