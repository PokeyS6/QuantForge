import json
import sys

import pytest

from quantforge.ai.artifacts import create_ai_variant_artifacts


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


def write_project(project_root, variants=None):
    variants = ["baseline"] if variants is None else variants
    project_file = project_root / "strategy.qf.json"
    project_file.write_text(
        json.dumps(
            {
                "strategy_id": "rsi_reversal_test",
                "baseline_variant_id": "baseline",
                "variants": variants,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return project_file


def write_baseline_results(project_root):
    baseline_dir = project_root / "variants" / "baseline"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "backtest_results.csv").write_text(
        "date,close,position,asset_return,strategy_return,equity\n",
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_valid_momentum_spec_creates_all_artifacts(tmp_path):
    project_file = write_project(tmp_path)
    write_baseline_results(tmp_path)

    variant_dir = create_ai_variant_artifacts(project_file, valid_momentum_spec())

    assert variant_dir.name == "variant_001_momentum_filter"
    assert (variant_dir / "modification_spec.json").exists()
    assert (variant_dir / "strategy_config.json").exists()
    assert (variant_dir / "change_summary.json").exists()
    assert (variant_dir / "diff.md").exists()


def test_modification_spec_json_equals_input_spec_exactly(tmp_path):
    project_file = write_project(tmp_path)
    write_baseline_results(tmp_path)
    spec = valid_momentum_spec()

    variant_dir = create_ai_variant_artifacts(project_file, spec)

    assert read_json(variant_dir / "modification_spec.json") == spec


def test_strategy_config_json_schema_exact(tmp_path):
    project_file = write_project(tmp_path)
    write_baseline_results(tmp_path)

    variant_dir = create_ai_variant_artifacts(project_file, valid_momentum_spec())

    assert read_json(variant_dir / "strategy_config.json") == {
        "variant_id": "variant_001_momentum_filter",
        "parent_variant_id": "baseline",
        "modification_type": "momentum_filter",
        "parameters": {
            "lookback_days": 20,
            "threshold": 0.0,
        },
        "entry_rule": "Add entry filter: return_20d > 0",
        "exit_rule": "unchanged",
    }


def test_change_summary_json_schema_exact(tmp_path):
    project_file = write_project(tmp_path)
    write_baseline_results(tmp_path)

    variant_dir = create_ai_variant_artifacts(project_file, valid_momentum_spec())

    assert read_json(variant_dir / "change_summary.json") == {
        "variant_id": "variant_001_momentum_filter",
        "parent_variant_id": "baseline",
        "user_instruction": "Add a momentum filter",
        "summary": "AI-assisted modification: momentum_filter",
        "assumptions": [
            "Momentum is computed using historical close prices only.",
            "The filter affects entries only.",
        ],
        "warnings": [
            "This filter may reduce trade count.",
            "Historical behavior does not imply future performance.",
        ],
        "status": "created_not_backtested",
        "source": "local_ai_planner",
    }


def test_diff_markdown_contains_required_structure(tmp_path):
    project_file = write_project(tmp_path)
    write_baseline_results(tmp_path)

    variant_dir = create_ai_variant_artifacts(project_file, valid_momentum_spec())
    diff = (variant_dir / "diff.md").read_text(encoding="utf-8")

    assert diff == """# Variant Diff — variant_001_momentum_filter

Parent: baseline

## Summary

This variant was created from a validated AI-assisted modification specification.

## Modification Type

momentum_filter

## Entry Rule Change

Add entry filter: return_20d > 0

## Exit Rule Change

unchanged

## Implementation Note

The AI did not generate executable strategy code.
Strategy logic is implemented by QuantForge's deterministic builder.

## Backtest Status

This variant has been created but not backtested yet.
"""


def test_missing_baseline_raises_and_creates_no_variant_folder(tmp_path):
    project_file = write_project(tmp_path)

    with pytest.raises(
        ValueError,
        match="ERROR: Baseline analysis not found. Run 'quantforge analyze' first.",
    ):
        create_ai_variant_artifacts(project_file, valid_momentum_spec())

    variants_dir = tmp_path / "variants"
    assert not variants_dir.exists()


def test_existing_variant_folders_use_next_max_plus_one_id(tmp_path):
    project_file = write_project(tmp_path)
    write_baseline_results(tmp_path)
    (tmp_path / "variants" / "variant_001_volatility_filter").mkdir()
    (tmp_path / "variants" / "variant_002_ml_price_floor").mkdir()
    (tmp_path / "variants" / "variant_007_test").mkdir()

    variant_dir = create_ai_variant_artifacts(project_file, valid_momentum_spec())

    assert variant_dir.name == "variant_008_momentum_filter"


def test_project_metadata_variants_preserves_existing_entries_and_appends(tmp_path):
    project_file = write_project(
        tmp_path,
        variants=["baseline", "variant_existing"],
    )
    write_baseline_results(tmp_path)

    create_ai_variant_artifacts(project_file, valid_momentum_spec())

    assert read_json(project_file)["variants"] == [
        "baseline",
        "variant_existing",
        "variant_001_momentum_filter",
    ]


def test_variants_not_list_raises_and_creates_no_variant_folder(tmp_path):
    project_file = write_project(tmp_path)
    project = read_json(project_file)
    project["variants"] = {"baseline": True}
    project_file.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")
    write_baseline_results(tmp_path)

    with pytest.raises(ValueError, match="Project metadata 'variants' must be a list."):
        create_ai_variant_artifacts(project_file, valid_momentum_spec())

    assert not (tmp_path / "variants" / "variant_001_momentum_filter").exists()


def test_write_failure_creates_no_partial_ai_variant_folder(tmp_path, monkeypatch):
    project_file = write_project(tmp_path)
    original_project_text = project_file.read_text(encoding="utf-8")
    write_baseline_results(tmp_path)
    original_write_text = type(project_file).write_text

    def failing_write_text(path, *args, **kwargs):
        if path.name == "strategy_config.json":
            raise OSError("simulated write failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(type(project_file), "write_text", failing_write_text)

    with pytest.raises(OSError, match="simulated write failure"):
        create_ai_variant_artifacts(project_file, valid_momentum_spec())

    variants_dir = tmp_path / "variants"
    assert not (variants_dir / "variant_001_momentum_filter").exists()
    assert not (variants_dir / ".variant_001_momentum_filter.tmp").exists()
    assert project_file.read_text(encoding="utf-8") == original_project_text


def test_project_metadata_replace_failure_rolls_back_ai_variant_folder(
    tmp_path,
    monkeypatch,
):
    project_file = write_project(tmp_path)
    original_project_text = project_file.read_text(encoding="utf-8")
    write_baseline_results(tmp_path)

    def failing_replace(source, target):
        if target == project_file:
            raise OSError("simulated metadata replace failure")
        source.replace(target)

    monkeypatch.setattr("quantforge.ai.artifacts._replace_path", failing_replace)

    with pytest.raises(OSError, match="simulated metadata replace failure"):
        create_ai_variant_artifacts(project_file, valid_momentum_spec())

    variants_dir = tmp_path / "variants"
    assert not (variants_dir / "variant_001_momentum_filter").exists()
    assert not (variants_dir / ".variant_001_momentum_filter.tmp").exists()
    assert project_file.read_text(encoding="utf-8") == original_project_text


def test_artifacts_module_does_not_import_planner_validation_or_cli():
    module = sys.modules["quantforge.ai.artifacts"]

    assert "planner" not in module.__dict__
    assert "validate_schema" not in module.__dict__
    assert "validate_against_registry" not in module.__dict__
    assert "cli" not in module.__dict__
