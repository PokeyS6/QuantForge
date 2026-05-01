"""Comparison report helpers for QuantForge."""

import json
from pathlib import Path


FORBIDDEN_OBSERVATION_WORDS = {
    "improved",
    "better",
    "worse",
    "optimal",
    "recommended",
}


def _format_metric_value(value) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _validate_matching_schema(baseline_metrics: dict, variant_metrics: dict) -> None:
    if list(baseline_metrics) != list(variant_metrics):
        raise ValueError("ERROR: Metrics schema mismatch between baseline and variant.")


def generate_comparison_observations(
    baseline_metrics: dict,
    variant_metrics: dict,
) -> list[str]:
    """Generate factual observations comparing baseline and variant metrics."""
    _validate_matching_schema(baseline_metrics, variant_metrics)
    observations = []
    for metric, baseline_value in baseline_metrics.items():
        variant_value = variant_metrics[metric]
        change = variant_value - baseline_value
        if change > 0:
            observations.append(f"{metric} increased by {_format_metric_value(change)}.")
        elif change < 0:
            observations.append(f"{metric} decreased by {_format_metric_value(abs(change))}.")
        else:
            observations.append(f"{metric} changed by 0.")
    return observations


def build_metrics_comparison_table(
    baseline_metrics: dict,
    variant_metrics: dict,
) -> str:
    """Build a Markdown table comparing baseline and variant metric values."""
    _validate_matching_schema(baseline_metrics, variant_metrics)
    rows = [
        "| Metric | Baseline | Variant | Change |",
        "|--------|----------|---------|--------|",
    ]
    for metric, baseline_value in baseline_metrics.items():
        variant_value = variant_metrics[metric]
        change = variant_value - baseline_value
        rows.append(
            "| "
            f"{metric} | "
            f"{_format_metric_value(baseline_value)} | "
            f"{_format_metric_value(variant_value)} | "
            f"{_format_metric_value(change)} |"
        )
    return "\n".join(rows)


def build_comparison_report(project_root: Path) -> str:
    """Build a Markdown comparison report for baseline and backtested variants."""
    baseline_metrics_path = project_root / "reports" / "baseline_metrics.json"
    variants_dir = project_root / "variants"

    if not baseline_metrics_path.exists():
        raise ValueError("ERROR: Baseline metrics not found. Run 'quantforge analyze' first.")

    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    variant_metrics_paths = sorted(
        path
        for path in variants_dir.glob("*/metrics.json")
        if path.parent.name != "baseline"
    )
    if not variant_metrics_paths:
        raise ValueError("ERROR: No backtested variants available for comparison.")

    sections = ["# Strategy Comparison", ""]
    for metrics_path in variant_metrics_paths:
        variant_id = metrics_path.parent.name
        variant_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        table = build_metrics_comparison_table(baseline_metrics, variant_metrics)
        observations = generate_comparison_observations(baseline_metrics, variant_metrics)

        sections.extend(
            [
                f"## Baseline vs {variant_id}",
                "",
                "### Metrics Comparison",
                "",
                table,
                "",
                "### Observations",
                "",
                *[f"- {observation}" for observation in observations],
                "",
                f"See variants/{variant_id}/diff.md for modification details.",
                "",
                "## Interpretation (Non-Advisory)",
                "",
                "This comparison is based on historical data only.",
                "Changes in metrics do not imply future performance.",
                "This analysis does not constitute financial advice.",
                "",
            ]
        )

    report = "\n".join(sections)
    lower_report = report.lower()
    for word in FORBIDDEN_OBSERVATION_WORDS:
        if word in lower_report:
            raise ValueError(f"Forbidden comparison wording found: {word}")
    return report
