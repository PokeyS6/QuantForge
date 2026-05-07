import pytest

from quantforge.ai.planner import (
    LOCAL_AI_UNAVAILABLE_MESSAGE,
    LocalAIPlannerOutputError,
    LocalAIPlannerUnavailable,
    build_planner_prompt,
    generate_modification_spec,
    loads_model_json,
)


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


def test_generate_modification_spec_missing_model_raises(monkeypatch):
    monkeypatch.delenv("QUANTFORGE_LOCAL_LLM_MODEL", raising=False)

    with pytest.raises(LocalAIPlannerUnavailable, match=LOCAL_AI_UNAVAILABLE_MESSAGE):
        generate_modification_spec("Add a volatility filter")


def test_generate_modification_spec_empty_model_raises(monkeypatch):
    monkeypatch.setenv("QUANTFORGE_LOCAL_LLM_MODEL", "  ")

    with pytest.raises(LocalAIPlannerUnavailable, match=LOCAL_AI_UNAVAILABLE_MESSAGE):
        generate_modification_spec("Add a volatility filter")


def test_generate_modification_spec_failed_subprocess_raises(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess_result(stdout="", stderr="failed", returncode=1)

    monkeypatch.setenv("QUANTFORGE_LOCAL_LLM_MODEL", "llama3")
    monkeypatch.setattr("quantforge.ai.planner.subprocess.run", fake_run)

    with pytest.raises(LocalAIPlannerUnavailable, match=LOCAL_AI_UNAVAILABLE_MESSAGE):
        generate_modification_spec("Add a volatility filter")


def test_generate_modification_spec_missing_ollama_raises(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("ollama")

    monkeypatch.setenv("QUANTFORGE_LOCAL_LLM_MODEL", "llama3")
    monkeypatch.setattr("quantforge.ai.planner.subprocess.run", fake_run)

    with pytest.raises(LocalAIPlannerUnavailable, match=LOCAL_AI_UNAVAILABLE_MESSAGE):
        generate_modification_spec("Add a volatility filter")


def test_generate_modification_spec_blank_stdout_raises(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess_result(stdout=" \n ", stderr="", returncode=0)

    monkeypatch.setenv("QUANTFORGE_LOCAL_LLM_MODEL", "llama3")
    monkeypatch.setattr("quantforge.ai.planner.subprocess.run", fake_run)

    with pytest.raises(LocalAIPlannerUnavailable, match=LOCAL_AI_UNAVAILABLE_MESSAGE):
        generate_modification_spec("Add a volatility filter")


def test_generate_modification_spec_success_returns_raw_string(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess_result(stdout="  not json but raw output  \n", stderr="", returncode=0)

    monkeypatch.setenv("QUANTFORGE_LOCAL_LLM_MODEL", "llama3")
    monkeypatch.setattr("quantforge.ai.planner.subprocess.run", fake_run)

    assert generate_modification_spec("Add a volatility filter") == "not json but raw output"


def test_generate_modification_spec_sends_prompt_to_stdin(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess_result(stdout='{"supported": false}', stderr="", returncode=0)

    monkeypatch.setenv("QUANTFORGE_LOCAL_LLM_MODEL", "llama3")
    monkeypatch.setattr("quantforge.ai.planner.subprocess.run", fake_run)

    generate_modification_spec("Add a momentum filter", parent_variant_id="variant_123")

    command, kwargs = calls[0]
    assert command == ["ollama", "run", "llama3"]
    assert "You are the QuantForge AI modification planner." in kwargs["input"]
    assert "user_instruction: Add a momentum filter" in kwargs["input"]
    assert "parent_variant_id: variant_123" in kwargs["input"]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False


def test_generate_modification_spec_does_not_parse_json_or_validate_output(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess_result(
            stdout="unsupported text that is not json and not schema-valid",
            stderr="",
            returncode=0,
        )

    monkeypatch.setenv("QUANTFORGE_LOCAL_LLM_MODEL", "llama3")
    monkeypatch.setattr("quantforge.ai.planner.subprocess.run", fake_run)

    assert (
        generate_modification_spec("Add a volatility filter")
        == "unsupported text that is not json and not schema-valid"
    )


def test_loads_model_json_repairs_raw_newline_inside_string():
    raw = """{
  "schema_version": "0.1",
  "supported": true,
  "non_advisory_note": "This modification creates a testable strategy
variant and does not constitute trading advice."
}"""

    parsed = loads_model_json(raw)

    assert (
        parsed["non_advisory_note"]
        == "This modification creates a testable strategy variant and does not constitute trading advice."
    )


def test_loads_model_json_rejects_unrecoverable_json_with_clear_error():
    with pytest.raises(
        LocalAIPlannerOutputError,
        match="Invalid JSON output from local model",
    ):
        loads_model_json("not json")


def test_loads_model_json_rejects_non_object_json():
    with pytest.raises(
        LocalAIPlannerOutputError,
        match="expected a JSON object",
    ):
        loads_model_json('["not", "an", "object"]')


def subprocess_result(stdout: str, stderr: str, returncode: int):
    return type(
        "CompletedProcessStub",
        (),
        {"stdout": stdout, "stderr": stderr, "returncode": returncode},
    )()
