"""Deterministic artifact creation for validated AI modification specs."""

import json
import re
from pathlib import Path


VARIANT_NAME_PATTERN = re.compile(r"^variant_(\d+)_.+")


def create_ai_variant_artifacts(project_file: Path, spec: dict) -> Path:
    """Create local variant artifacts from an already validated AI spec."""
    project_root = project_file.parent
    variants_dir = project_root / "variants"
    baseline_results_path = variants_dir / "baseline" / "backtest_results.csv"

    if not baseline_results_path.exists():
        raise ValueError("ERROR: Baseline analysis not found. Run 'quantforge analyze' first.")

    project = json.loads(project_file.read_text(encoding="utf-8"))
    variants = project.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError("Project metadata 'variants' must be a list.")

    variant_id = _next_variant_id(variants_dir, spec["modification_type"])
    variant_dir = variants_dir / variant_id
    if variant_dir.exists():
        raise ValueError(f"ERROR: Variant {variant_id} already exists. Refusing to overwrite.")

    strategy_config = {
        "variant_id": variant_id,
        "parent_variant_id": "baseline",
        "modification_type": spec["modification_type"],
        "parameters": spec["parameters"],
        "entry_rule": spec["entry_rule_change"],
        "exit_rule": spec["exit_rule_change"],
    }
    change_summary = {
        "variant_id": variant_id,
        "parent_variant_id": "baseline",
        "user_instruction": spec["user_instruction"],
        "summary": f"AI-assisted modification: {spec['modification_type']}",
        "assumptions": spec["assumptions"],
        "warnings": spec["warnings"],
        "status": "created_not_backtested",
        "source": "local_ai_planner",
    }

    variant_dir.mkdir(parents=True)
    (variant_dir / "modification_spec.json").write_text(
        json.dumps(spec, indent=2) + "\n",
        encoding="utf-8",
    )
    (variant_dir / "strategy_config.json").write_text(
        json.dumps(strategy_config, indent=2) + "\n",
        encoding="utf-8",
    )
    (variant_dir / "change_summary.json").write_text(
        json.dumps(change_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (variant_dir / "diff.md").write_text(
        _diff_markdown(variant_id, spec),
        encoding="utf-8",
    )

    variants.append(variant_id)
    project["variants"] = variants
    project_file.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")

    return variant_dir


def _next_variant_id(variants_dir: Path, modification_type: str) -> str:
    max_variant_number = 0
    if variants_dir.exists():
        for path in variants_dir.iterdir():
            if not path.is_dir():
                continue
            match = VARIANT_NAME_PATTERN.match(path.name)
            if match:
                max_variant_number = max(max_variant_number, int(match.group(1)))
    return f"variant_{max_variant_number + 1:03d}_{modification_type}"


def _diff_markdown(variant_id: str, spec: dict) -> str:
    return "\n".join(
        [
            f"# Variant Diff — {variant_id}",
            "",
            "Parent: baseline",
            "",
            "## Summary",
            "",
            "This variant was created from a validated AI-assisted modification specification.",
            "",
            "## Modification Type",
            "",
            spec["modification_type"],
            "",
            "## Entry Rule Change",
            "",
            spec["entry_rule_change"],
            "",
            "## Exit Rule Change",
            "",
            spec["exit_rule_change"],
            "",
            "## Implementation Note",
            "",
            "The AI did not generate executable strategy code.",
            "Strategy logic is implemented by QuantForge's deterministic builder.",
            "",
            "## Backtest Status",
            "",
            "This variant has been created but not backtested yet.",
            "",
        ]
    )
