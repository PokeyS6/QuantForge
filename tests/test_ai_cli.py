import json

from typer.testing import CliRunner

from quantforge.ai.planner import LOCAL_AI_UNAVAILABLE_MESSAGE, LocalAIPlannerUnavailable
from quantforge.cli import app


runner = CliRunner()


def valid_momentum_spec() -> dict:
    return {
        "schema_version": "0.1",
        "ai_planner_version": "local_llm_v1",
        "supported": True,
        "modification_type": "momentum_filter",
        "parent_variant_id": "baseline",
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


def unsupported_spec() -> dict:
    return {
        "schema_version": "0.1",
        "ai_planner_version": "local_llm_v1",
        "supported": False,
        "user_instruction": "Optimize this strategy for max returns",
        "reason": "Optimization requests are not supported.",
        "suggested_supported_requests": [
            "Add a volatility filter",
            "Add a momentum filter",
        ],
        "non_advisory_note": "QuantForge evaluates behavior and does not recommend trades.",
    }


def write_project(tmp_path, with_baseline: bool = True):
    project_root = tmp_path / "aapl-rsi-reversal"
    project_root.mkdir()
    project_file = project_root / "strategy.qf.json"
    project_file.write_text(
        json.dumps(
            {
                "strategy_id": "rsi_reversal_aapl",
                "baseline_variant_id": "baseline",
                "variants": ["baseline"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if with_baseline:
        baseline_dir = project_root / "variants" / "baseline"
        baseline_dir.mkdir(parents=True)
        (baseline_dir / "backtest_results.csv").write_text(
            "date,close,position,asset_return,strategy_return,equity\n",
            encoding="utf-8",
        )
    return project_file


def ai_variant_dirs(project_file):
    variants_dir = project_file.parent / "variants"
    if not variants_dir.exists():
        return []
    return sorted(
        path
        for path in variants_dir.iterdir()
        if path.is_dir() and path.name.startswith("variant_")
    )


def test_modify_without_ai_existing_behavior_unchanged_and_planner_not_called(
    tmp_path,
    monkeypatch,
):
    project_file = write_project(tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("AI planner should not be called without --ai")

    monkeypatch.setattr("quantforge.cli.generate_modification_spec", fail_if_called)

    result = runner.invoke(app, ["modify", str(project_file), "Add a volatility filter"])

    assert result.exit_code == 0
    assert result.output == """Variant 'variant_001_volatility_filter' created.

This variant has not been backtested yet.
Variant backtesting will be available in a subsequent step.
"""


def test_modify_with_ai_valid_momentum_spec_creates_variant_artifacts(
    tmp_path,
    monkeypatch,
):
    project_file = write_project(tmp_path)
    monkeypatch.setattr(
        "quantforge.cli.generate_modification_spec",
        lambda *args, **kwargs: json.dumps(valid_momentum_spec()),
    )

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a momentum filter", "--ai"],
    )

    variant_dir = project_file.parent / "variants" / "variant_001_momentum_filter"
    assert result.exit_code == 0
    assert result.output == """Creating AI-assisted variant: variant_001_momentum_filter
Modification type: momentum_filter
Rule change: Add entry filter: return_20d > 0

This variant has not been backtested yet.
Variant backtesting will be available in a subsequent step.
"""
    assert (variant_dir / "modification_spec.json").exists()
    assert (variant_dir / "strategy_config.json").exists()
    assert (variant_dir / "change_summary.json").exists()
    assert (variant_dir / "diff.md").exists()


def test_modify_with_ai_repairs_raw_newline_inside_json_string(tmp_path, monkeypatch):
    project_file = write_project(tmp_path)
    raw_output = json.dumps(valid_momentum_spec()).replace(
        "testable strategy variant",
        "testable strategy\nvariant",
    )
    monkeypatch.setattr(
        "quantforge.cli.generate_modification_spec",
        lambda *args, **kwargs: raw_output,
    )

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a momentum filter", "--ai"],
    )

    assert result.exit_code == 0
    assert "Creating AI-assisted variant: variant_001_momentum_filter\n" in result.output
    variant_dir = project_file.parent / "variants" / "variant_001_momentum_filter"
    saved_spec = json.loads((variant_dir / "modification_spec.json").read_text())
    assert (
        saved_spec["non_advisory_note"]
        == "This modification creates a testable strategy variant and does not constitute trading advice."
    )


def test_modify_with_ai_invalid_json_creates_no_variant_folder(tmp_path, monkeypatch):
    project_file = write_project(tmp_path)
    monkeypatch.setattr(
        "quantforge.cli.generate_modification_spec",
        lambda *args, **kwargs: "not json",
    )

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a momentum filter", "--ai"],
    )

    assert result.exit_code != 0
    assert result.output.startswith(
        "ERROR: AI modification spec failed validation: "
        "Invalid JSON output from local model:"
    )
    assert result.output.endswith("""No variant was created.
""")
    assert "Expecting value at line 1 column 1." in result.output
    assert ai_variant_dirs(project_file) == []


def test_modify_with_ai_invalid_control_json_reports_parse_cause(
    tmp_path,
    monkeypatch,
):
    project_file = write_project(tmp_path)
    monkeypatch.setattr(
        "quantforge.cli.generate_modification_spec",
        lambda *args, **kwargs: '{"supported": true, "note": "unterminated\n',
    )

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a momentum filter", "--ai"],
    )

    assert result.exit_code != 0
    assert "ERROR: AI modification spec failed validation: Invalid JSON output from local model:" in result.output
    assert "No variant was created." in result.output
    assert ai_variant_dirs(project_file) == []


def test_modify_with_ai_schema_failure_creates_no_variant_folder(tmp_path, monkeypatch):
    project_file = write_project(tmp_path)
    spec = valid_momentum_spec()
    del spec["schema_version"]
    monkeypatch.setattr(
        "quantforge.cli.generate_modification_spec",
        lambda *args, **kwargs: json.dumps(spec),
    )

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a momentum filter", "--ai"],
    )

    assert result.exit_code != 0
    assert result.output == """ERROR: AI modification spec failed validation: Missing required field: schema_version
No variant was created.
"""
    assert ai_variant_dirs(project_file) == []


def test_modify_with_ai_unsupported_spec_prints_reason_and_suggestions(
    tmp_path,
    monkeypatch,
):
    project_file = write_project(tmp_path)
    monkeypatch.setattr(
        "quantforge.cli.generate_modification_spec",
        lambda *args, **kwargs: json.dumps(unsupported_spec()),
    )

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Optimize this strategy", "--ai"],
    )

    assert result.exit_code != 0
    assert result.output == """ERROR: Unsupported AI modification request.
Optimization requests are not supported.
Suggested supported requests:
- Add a volatility filter
- Add a momentum filter
"""
    assert ai_variant_dirs(project_file) == []


def test_modify_with_ai_unsupported_modification_type_creates_no_variant_folder(
    tmp_path,
    monkeypatch,
):
    project_file = write_project(tmp_path)
    spec = valid_momentum_spec()
    spec["modification_type"] = "golden_cross"
    spec["parameters"] = {}
    monkeypatch.setattr(
        "quantforge.cli.generate_modification_spec",
        lambda *args, **kwargs: json.dumps(spec),
    )

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a momentum filter", "--ai"],
    )

    assert result.exit_code != 0
    assert result.output == """ERROR: AI modification spec failed validation: Unsupported modification type 'golden_cross'
No variant was created.
"""
    assert ai_variant_dirs(project_file) == []


def test_modify_with_ai_missing_required_parameter_creates_no_variant_folder(
    tmp_path,
    monkeypatch,
):
    project_file = write_project(tmp_path)
    spec = valid_momentum_spec()
    del spec["parameters"]["threshold"]
    monkeypatch.setattr(
        "quantforge.cli.generate_modification_spec",
        lambda *args, **kwargs: json.dumps(spec),
    )

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a momentum filter", "--ai"],
    )

    assert result.exit_code != 0
    assert result.output == """ERROR: AI modification spec failed validation: Missing required parameter 'threshold'
No variant was created.
"""
    assert ai_variant_dirs(project_file) == []


def test_modify_with_ai_out_of_bounds_parameter_creates_no_variant_folder(
    tmp_path,
    monkeypatch,
):
    project_file = write_project(tmp_path)
    spec = valid_momentum_spec()
    spec["parameters"]["lookback_days"] = 300
    monkeypatch.setattr(
        "quantforge.cli.generate_modification_spec",
        lambda *args, **kwargs: json.dumps(spec),
    )

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a momentum filter", "--ai"],
    )

    assert result.exit_code != 0
    assert result.output == """ERROR: AI modification spec failed validation: Parameter 'lookback_days' is out of bounds (expected 5–100, got 300)
No variant was created.
"""
    assert ai_variant_dirs(project_file) == []


def test_modify_with_ai_planner_unavailable_surfaces_exact_error(
    tmp_path,
    monkeypatch,
):
    project_file = write_project(tmp_path)

    def unavailable(*args, **kwargs):
        raise LocalAIPlannerUnavailable(LOCAL_AI_UNAVAILABLE_MESSAGE)

    monkeypatch.setattr("quantforge.cli.generate_modification_spec", unavailable)

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a momentum filter", "--ai"],
    )

    assert result.exit_code != 0
    assert result.output == f"{LOCAL_AI_UNAVAILABLE_MESSAGE}\n"


def test_modify_with_ai_variant_numbering_uses_max_plus_one(tmp_path, monkeypatch):
    project_file = write_project(tmp_path)
    variants_dir = project_file.parent / "variants"
    (variants_dir / "variant_001_volatility_filter").mkdir()
    (variants_dir / "variant_007_test").mkdir()
    monkeypatch.setattr(
        "quantforge.cli.generate_modification_spec",
        lambda *args, **kwargs: json.dumps(valid_momentum_spec()),
    )

    result = runner.invoke(
        app,
        ["modify", str(project_file), "Add a momentum filter", "--ai"],
    )

    assert result.exit_code == 0
    assert "Creating AI-assisted variant: variant_008_momentum_filter\n" in result.output
    assert (variants_dir / "variant_008_momentum_filter").exists()
