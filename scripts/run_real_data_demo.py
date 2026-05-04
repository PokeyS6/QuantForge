"""Run a one-command QuantForge real-data CLI demo.

This script is demo and validation tooling. It downloads local CSV data with
yfinance, runs the existing QuantForge CLI, and writes a compact markdown
summary. It does not change QuantForge strategy logic or create trading advice.
"""

from __future__ import annotations

import argparse
import json
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_KEYS = [
    "total_return",
    "annualized_return",
    "max_drawdown",
    "exposure",
    "trade_count",
    "win_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QuantForge real-data demo.")
    parser.add_argument("--ticker", default="SPY", help="Ticker to download with yfinance.")
    parser.add_argument("--start", default="2018-01-01", help="Start date for the demo data.")
    parser.add_argument("--end", default=None, help="Optional end date for the demo data.")
    parser.add_argument(
        "--data-csv",
        default=None,
        help="Optional local OHLCV CSV. If provided, yfinance is skipped.",
    )
    parser.add_argument(
        "--output-dir",
        default="demo_real_data_run",
        help="Directory where demo data, project, and reports will be written.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Reuse output-dir instead of deleting it before the run.",
    )
    return parser.parse_args()


def import_yfinance():
    try:
        import yfinance as yf
    except ImportError as error:
        raise SystemExit("Missing optional dependency: pip install yfinance") from error
    return yf


def flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data.columns, pd.MultiIndex):
        return data

    for level in range(data.columns.nlevels):
        candidate = data.copy()
        candidate.columns = candidate.columns.get_level_values(level)
        normalized = {str(column).lower() for column in candidate.columns}
        if {"open", "high", "low", "close", "volume"}.issubset(normalized):
            return candidate

    flattened = data.copy()
    flattened.columns = [
        "_".join(str(part) for part in column if str(part))
        for column in flattened.columns.to_flat_index()
    ]
    return flattened


def download_ohlcv(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    yf = import_yfinance()
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
    except Exception as error:
        raise SystemExit(yfinance_failure_message(ticker, error)) from error
    if data.empty:
        raise SystemExit(yfinance_failure_message(ticker))

    return normalize_ohlcv(data)


def yfinance_failure_message(ticker: str, error: Exception | None = None) -> str:
    details = f" Details: {error}" if error else ""
    return (
        f"Yahoo/yfinance returned no usable rows for {ticker}.{details}\n"
        "Yahoo/yfinance rate limiting can happen, including YFRateLimitError: "
        "Too Many Requests.\n"
        "Retry later or pass --data-csv path/to/file.csv."
    )


def normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    data = flatten_columns(data)
    data = data.reset_index()
    column_lookup = {str(column).lower().replace(" ", "_"): column for column in data.columns}
    date_column = column_lookup.get("date") or column_lookup.get("datetime")
    required = {
        "open": column_lookup.get("open"),
        "high": column_lookup.get("high"),
        "low": column_lookup.get("low"),
        "close": column_lookup.get("close"),
        "volume": column_lookup.get("volume"),
    }
    missing = [name for name, column in required.items() if column is None]
    if date_column is None:
        missing.append("date")
    if missing:
        raise SystemExit(f"OHLCV data missing required columns: {', '.join(missing)}")

    normalized = pd.DataFrame(
        {
            "date": pd.to_datetime(data[date_column]).dt.date.astype(str),
            "open": data[required["open"]],
            "high": data[required["high"]],
            "low": data[required["low"]],
            "close": data[required["close"]],
            "volume": data[required["volume"]],
        }
    )
    normalized = normalized.dropna()
    if normalized.empty:
        raise SystemExit("OHLCV data has no complete rows after normalization.")
    return normalized


def load_local_ohlcv_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    return normalize_ohlcv(pd.read_csv(path))


def validate_ohlcv_csv(path: Path) -> None:
    data = pd.read_csv(path)
    expected_columns = ["date", "open", "high", "low", "close", "volume"]
    if list(data.columns) != expected_columns:
        raise SystemExit(f"CSV columns must be exactly {expected_columns}.")
    if data.empty:
        raise SystemExit("CSV must contain at least one row.")


def run_command(command: list[str], command_outputs: list[dict[str, str | int]]) -> None:
    rendered = shlex.join(command)
    print(f"$ {rendered}")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    command_outputs.append(
        {
            "command": rendered,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    )


def find_project_file(output_dir: Path) -> Path:
    project_files = sorted(output_dir.glob("*/strategy.qf.json"))
    if not project_files:
        raise SystemExit(f"No strategy.qf.json found under {output_dir}.")
    if len(project_files) > 1:
        matches = ", ".join(str(path) for path in project_files)
        raise SystemExit(f"Multiple strategy.qf.json files found under {output_dir}: {matches}")
    return project_files[0]


def read_metrics(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    return {key: metrics[key] for key in METRIC_KEYS}


def format_metric(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_manual_comparison_table(metrics_by_name: dict[str, dict[str, Any]]) -> str:
    rows = ["| Metric | Baseline | Volatility Filter | ML Price Floor |"]
    rows.append("|--------|----------|-------------------|----------------|")
    for metric in METRIC_KEYS:
        rows.append(
            "| "
            f"{metric} | "
            f"{format_metric(metrics_by_name['Baseline'][metric])} | "
            f"{format_metric(metrics_by_name['Volatility Filter'][metric])} | "
            f"{format_metric(metrics_by_name['ML Price Floor'][metric])} |"
        )
    return "\n".join(rows)


def metric_change_sentence(label: str, baseline: dict[str, Any], variant: dict[str, Any]) -> str:
    parts = []
    for metric in ["trade_count", "exposure", "max_drawdown"]:
        change = variant[metric] - baseline[metric]
        if change > 0:
            direction = "increased"
        elif change < 0:
            direction = "decreased"
        else:
            direction = "was unchanged"
        parts.append(f"{metric} {direction} by {format_metric(abs(change))}")
    return f"- {label}: " + "; ".join(parts) + "."


def build_report(
    ticker: str,
    start: str,
    end: str | None,
    csv_path: Path,
    data_source_note: str,
    command_outputs: list[dict[str, str | int]],
    metrics_by_name: dict[str, dict[str, Any]],
    comparison_table: str,
) -> str:
    date_range = f"{start} to {end}" if end else f"{start} onward"
    low_trade_warnings = [
        f"- {name} trade_count is below 10; this may be too low for a compelling demo."
        for name, metrics in metrics_by_name.items()
        if metrics["trade_count"] < 10
    ]
    if not low_trade_warnings:
        low_trade_warnings = ["- No low trade count warning triggered."]

    return "\n".join(
        [
            "# QuantForge Real-Data Demo Report",
            "",
            "## Run Summary",
            f"- Ticker: {ticker}",
            f"- Date range: {date_range}",
            f"- Local CSV: {csv_path}",
            f"- Data source: {data_source_note}",
            "",
            "## Commands Executed",
            "",
            "```bash",
            *[str(output["command"]) for output in command_outputs],
            "```",
            "",
            "## Command Output",
            command_output_block(command_outputs),
            "",
            "## Baseline Metrics",
            metrics_block(metrics_by_name["Baseline"]),
            "",
            "## Volatility Metrics",
            metrics_block(metrics_by_name["Volatility Filter"]),
            "",
            "## ML Metrics",
            metrics_block(metrics_by_name["ML Price Floor"]),
            "",
            "## Manual Comparison Table",
            comparison_table,
            "",
            "## Non-Advisory Interpretation",
            "This demo compares historical backtest outputs for a user-provided RSI hypothesis.",
            metric_change_sentence(
                "Volatility filter vs baseline",
                metrics_by_name["Baseline"],
                metrics_by_name["Volatility Filter"],
            ),
            metric_change_sentence(
                "ML price floor vs baseline",
                metrics_by_name["Baseline"],
                metrics_by_name["ML Price Floor"],
            ),
            "Changes in historical metrics do not imply future performance.",
            "This report does not provide financial advice or a trading recommendation.",
            "",
            "## Demo Warnings",
            *low_trade_warnings,
            "",
        ]
    )


def metrics_block(metrics: dict[str, Any]) -> str:
    return "\n".join(f"- {key}: {format_metric(metrics[key])}" for key in METRIC_KEYS)


def command_output_block(command_outputs: list[dict[str, str | int]]) -> str:
    blocks = []
    for output in command_outputs:
        blocks.extend(
            [
                f"### `{output['command']}`",
                "",
                f"- Exit code: {output['returncode']}",
                "",
                "stdout:",
                "",
                "```text",
                str(output["stdout"]).strip() or "(empty)",
                "```",
                "",
                "stderr:",
                "",
                "```text",
                str(output["stderr"]).strip() or "(empty)",
                "```",
                "",
            ]
        )
    return "\n".join(blocks).rstrip()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()
    output_dir = Path(args.output_dir)

    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{ticker}_ohlcv.csv"

    if args.data_csv:
        prices = load_local_ohlcv_csv(Path(args.data_csv))
        data_source_note = "user-provided CSV (external real historical data)"
    else:
        prices = download_ohlcv(ticker=ticker, start=args.start, end=args.end)
        data_source_note = "yfinance/Yahoo Finance, research/demo use only."
    prices.to_csv(csv_path, index=False)
    validate_ohlcv_csv(csv_path)

    command_outputs: list[dict[str, str | int]] = []

    run_command(
        [
            "quantforge",
            "create",
            "--ticker",
            ticker,
            "--start",
            args.start,
            "--output-dir",
            str(output_dir),
            "RSI mean reversion strategy real-data demo",
        ],
        command_outputs,
    )
    project_file = find_project_file(output_dir)
    project_dir = project_file.parent
    run_command(
        ["quantforge", "analyze", str(project_file), "--data-csv", str(csv_path)],
        command_outputs,
    )
    run_command(
        [
            "quantforge",
            "modify",
            str(project_file),
            "Add a volatility filter that avoids trading during high volatility regimes",
        ],
        command_outputs,
    )
    run_command(
        [
            "quantforge",
            "analyze",
            str(project_file),
            "--data-csv",
            str(csv_path),
            "--variant-id",
            "variant_001_volatility_filter",
        ],
        command_outputs,
    )
    run_command(
        ["quantforge", "modify", str(project_file), "Add an ml price floor"],
        command_outputs,
    )
    run_command(
        [
            "quantforge",
            "analyze",
            str(project_file),
            "--data-csv",
            str(csv_path),
            "--variant-id",
            "variant_002_ml_price_floor",
        ],
        command_outputs,
    )
    run_command(
        ["quantforge", "compare", str(project_file), "--all-variants"],
        command_outputs,
    )

    metrics_by_name = {
        "Baseline": read_metrics(project_dir / "reports" / "baseline_metrics.json"),
        "Volatility Filter": read_metrics(
            project_dir / "variants" / "variant_001_volatility_filter" / "metrics.json"
        ),
        "ML Price Floor": read_metrics(
            project_dir / "variants" / "variant_002_ml_price_floor" / "metrics.json"
        ),
    }
    comparison_table = build_manual_comparison_table(metrics_by_name)
    report = build_report(
        ticker=ticker,
        start=args.start,
        end=args.end,
        csv_path=csv_path,
        data_source_note=data_source_note,
        command_outputs=command_outputs,
        metrics_by_name=metrics_by_name,
        comparison_table=comparison_table,
    )
    report_path = output_dir / "real_data_demo_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Real-data demo report saved to: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"Command failed with exit code {error.returncode}: {shlex.join(error.cmd)}")
        sys.exit(error.returncode)
