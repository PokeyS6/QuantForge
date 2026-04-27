"""Project input and output helpers for QuantForge."""

import json
from pathlib import Path

from quantforge.core.models import StrategyProject
from quantforge.core.paths import baseline_config_path, project_metadata_path


def _baseline_variant(project: StrategyProject):
    for variant in project.variants:
        if variant.variant_id == project.baseline_variant_id:
            return variant
    raise ValueError(f"Missing baseline variant: {project.baseline_variant_id}")


def write_project(project_path: Path, project: StrategyProject) -> None:
    """Write project metadata and baseline variant configuration."""
    project_path.mkdir(parents=True, exist_ok=True)
    baseline_path = baseline_config_path(project_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    project_metadata_path(project_path).write_text(
        json.dumps(project.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    baseline_variant = _baseline_variant(project)
    baseline_path.write_text(
        json.dumps(baseline_variant.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def read_project(project_metadata_file: Path) -> dict:
    """Read project metadata from strategy.qf.json."""
    return json.loads(project_metadata_file.read_text(encoding="utf-8"))


def get_baseline_variant(project: dict) -> dict:
    """Return the baseline variant from project metadata."""
    baseline_variant_id = project.get("baseline_variant_id")
    for variant in project.get("variants", []):
        if variant.get("variant_id") == baseline_variant_id:
            return variant
    raise ValueError(f"Missing baseline variant: {baseline_variant_id}")


def project_dir_from_metadata(project_metadata_file: Path) -> Path:
    """Return the project directory for a strategy metadata file."""
    return project_metadata_file.parent
