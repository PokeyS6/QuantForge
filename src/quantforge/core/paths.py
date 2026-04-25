"""Path helpers for local QuantForge projects."""

from pathlib import Path


def project_dir(output_dir: Path, project_name: str) -> Path:
    """Return the local project directory path."""
    slug = project_name.lower().replace(" ", "-")
    return output_dir / slug


def project_metadata_path(project_path: Path) -> Path:
    """Return the strategy metadata path."""
    return project_path / "strategy.qf.json"


def baseline_config_path(project_path: Path) -> Path:
    """Return the baseline variant config path."""
    return project_path / "variants" / "baseline" / "strategy_config.json"
