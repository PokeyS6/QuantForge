# QuantForge

QuantForge is a local-first, CLI-first AI-assisted quantitative strategy workbench.
It is intended to help technical users build, modify, backtest, validate, compare,
and audit trading strategy variants.

QuantForge helps test user-provided hypotheses. It does not provide trading advice.

## What QuantForge Is Not

QuantForge is not:

- a financial advisor
- a buy/sell signal service
- a trading bot
- a broker execution tool
- a guaranteed profit engine
- a tool that silently optimizes parameters
- a tool that hides generated code or assumptions

## Current Status

QuantForge is currently a skeleton only. The repository contains packaging, import
tests, and a placeholder CLI shell. It does not yet implement strategy logic,
backtesting, data fetching, variants, ML, reports, dashboards, cloud services,
accounts, payments, or broker execution.

## CLI

Current placeholder commands:

```bash
quantforge create
quantforge analyze
quantforge modify
quantforge compare
```

Each command currently prints a short non-advisory placeholder message.

## Planned First Demo Loop

The intended first demo follows the core product loop:

```text
Build strategy -> Backtest baseline -> Diagnose weaknesses -> Modify strategy
-> Create variant -> Validate variant -> Compare against baseline -> Repeat
```

The first planned walkthrough uses an RSI strategy on AAPL, compares a baseline
against a volatility-filter variant, and later introduces an ML price-floor
variant with walk-forward validation.

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
