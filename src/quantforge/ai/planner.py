"""Deterministic prompt construction for the local AI modification planner."""

import json
import os
import re
import subprocess


LOCAL_AI_UNAVAILABLE_MESSAGE = (
    "ERROR: Local AI planner unavailable. Configure a local model before using "
    "AI-assisted modifications."
)


class LocalAIPlannerUnavailable(RuntimeError):
    """Raised when the local Ollama planner cannot be used."""


class LocalAIPlannerOutputError(ValueError):
    """Raised when local planner output cannot be parsed as a JSON object."""


def _normalize_model_json(raw: str) -> str:
    """Replace raw control characters only when they appear inside JSON strings."""
    raw = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw)
    normalized = []
    in_string = False
    escaped = False

    for character in raw:
        if escaped:
            normalized.append(character)
            escaped = False
            continue

        if character == "\\":
            normalized.append(character)
            escaped = in_string
            continue

        if character == '"':
            normalized.append(character)
            in_string = not in_string
            continue

        if in_string and ord(character) < 0x20:
            normalized.append(" ")
            continue

        normalized.append(character)

    return "".join(normalized)


def loads_model_json(raw: str) -> dict:
    """Parse local model output after narrowly repairing JSON string controls."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as initial_error:
        normalized = _normalize_model_json(raw)
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError as repaired_error:
            raise LocalAIPlannerOutputError(
                "Invalid JSON output from local model: "
                f"{repaired_error.msg} at line {repaired_error.lineno} "
                f"column {repaired_error.colno}."
            ) from repaired_error
        if normalized == raw:
            raise LocalAIPlannerOutputError(
                "Invalid JSON output from local model: "
                f"{initial_error.msg} at line {initial_error.lineno} "
                f"column {initial_error.colno}."
            ) from initial_error

    if not isinstance(parsed, dict):
        raise LocalAIPlannerOutputError(
            "Invalid JSON output from local model: expected a JSON object."
        )

    return parsed


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
        "reason": "Optimization requests are not supported because QuantForge does not silently search for highest-return parameter sets.",
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
            "- finding highest-return parameter sets",
            "- directional trade instructions",
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


def generate_modification_spec(
    user_instruction: str,
    parent_variant_id: str = "baseline",
) -> str:
    """Call local Ollama and return the raw planner output."""
    model = os.environ.get("QUANTFORGE_LOCAL_LLM_MODEL", "").strip()
    if not model:
        raise LocalAIPlannerUnavailable(LOCAL_AI_UNAVAILABLE_MESSAGE)

    prompt = build_planner_prompt(user_instruction, parent_variant_id)
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise LocalAIPlannerUnavailable(LOCAL_AI_UNAVAILABLE_MESSAGE) from error

    output = result.stdout.strip()
    if result.returncode != 0 or not output:
        raise LocalAIPlannerUnavailable(LOCAL_AI_UNAVAILABLE_MESSAGE)

    return output
