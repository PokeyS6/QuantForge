# QuantForge

QuantForge is a local-first, CLI-first system to evaluate strategy behavior and
tradeoffs. It is intended to help technical users build, modify, backtest,
validate, compare, and audit trading strategy variants.
It evaluates how strategy variants behave on historical data; it does not
predict future outcomes.

QuantForge helps test user-provided hypotheses. It does not provide trading advice.

AI-assisted modifications are local-first. They use a local Ollama model to
turn a user prompt into a validated JSON specification, then QuantForge's
deterministic builder creates auditable variant artifacts. The AI does not
generate executable strategy code.

## What QuantForge Is Not

QuantForge is not:

- a financial advisor
- a directional signal service
- a trading bot
- a broker execution tool
- a performance-guarantee engine
- a tool that silently optimizes parameters
- a tool that hides generated code or assumptions

## Current Status

QuantForge currently supports a local RSI baseline workflow, a volatility-filter
variant, an ML price-floor variant, persisted analysis artifacts, and comparison
reports. It remains local-first and CLI-only.

## CLI

Current commands:

```bash
quantforge create
quantforge analyze
quantforge modify
quantforge compare
```

The commands create local projects, run historical analyses from local CSV data,
create supported variants, and compare persisted metrics.

## AI-Assisted Modifications

The `--ai` workflow requires Ollama and a local model name:

```bash
export QUANTFORGE_LOCAL_LLM_MODEL=<model-name>
quantforge modify strategy.qf.json --ai "add a momentum confirmation filter"
```

Pipeline:

```text
user prompt -> local LLM -> validated JSON spec -> deterministic builder -> auditable variant
```

If Ollama or the selected model is unavailable, the planner fails clearly and no
AI-assisted variant is created. Non-AI workflows continue to work without
Ollama.

See [AI Ollama Setup](docs/ai_ollama_setup.md) for setup and validation steps.

## Real-Data Demo

Run the current end-to-end demo with real historical OHLCV data:

```bash
python scripts/run_real_data_demo.py --ticker SPY --start 2018-01-01
```

If Yahoo/yfinance rate-limits the request, provide a local CSV:

```bash
python scripts/run_real_data_demo.py --ticker SPY --start 2018-01-01 --data-csv path/to/spy_ohlcv.csv
```

The runner creates a project, analyzes the baseline, creates and analyzes the
volatility and ML variants, runs comparison, and writes
`real_data_demo_report.md` under the output directory.

## Demo Loop

The intended first demo follows the core product loop:

```text
Build strategy -> Backtest baseline -> Diagnose weaknesses -> Modify strategy
-> Create variant -> Validate variant -> Compare against baseline -> Repeat
```

The current walkthrough compares an RSI baseline against a volatility-filter
variant and an ML price-floor variant. The focus is how filters change
participation, exposure, drawdown, and trade count in a historical test window.

See [Product Boundary](docs/product_boundary.md) and
[Demo Script](docs/demo_script.md) for the practical guardrails and demo plan.

## Development

Install the package with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```
