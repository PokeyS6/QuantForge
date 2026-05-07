# Demo Script

The current QuantForge demo is a one-command CLI run that creates a local
project, analyzes a baseline, creates two variants, analyzes both variants, and
writes comparison artifacts.

## Run

```bash
python scripts/run_real_data_demo.py --ticker SPY --start 2018-01-01
```

If yfinance/Yahoo Finance rate-limits the request, use a local OHLCV CSV:

```bash
python scripts/run_real_data_demo.py --ticker SPY --start 2018-01-01 --data-csv path/to/spy_ohlcv.csv
```

## Narrative

The demo compares three historical analyses:

- Baseline: RSI mean-reversion hypothesis.
- Volatility filter: blocks entries during high-volatility regimes.
- ML price floor: blocks entries when walk-forward downside-risk probability is above the configured threshold.

The comparison should focus on tradeoffs: participation, exposure, trade count,
drawdown, and total return changed across variants. These changes are historical
test results, not trading instructions.
In practice, filters can reduce participation without eliminating downside risk.

## Optional AI-Assisted Modification Demo

For student validation demos, AI-assisted modifications require local Ollama
setup. QuantForge sends the user prompt to a local model, validates the returned
JSON spec, and then uses deterministic builders to create auditable variant
artifacts. The AI does not generate executable strategy code.

```bash
export QUANTFORGE_LOCAL_LLM_MODEL=<model-name>
quantforge modify strategy.qf.json --ai "Add a momentum filter with a short lookback and low threshold"
```

Curated demo prompt:

```text
Add a momentum filter with a short lookback and low threshold
```

This prompt is used for demo consistency, not optimization. It is intended to
create an AI-assisted `momentum_filter` variant that filters some RSI entries
based on recent momentum confirmation. On the demo dataset, the created variant
should visibly change trade count, exposure, and/or return/drawdown metrics
relative to the baseline. Those changes describe historical behavior only; they
are not a trading recommendation or a performance guarantee.

After the variant is created, note the created variant id from the CLI output
and run:

```bash
quantforge analyze strategy.qf.json --data-csv ../../real_data_csvs/spy_ohlcv.csv --variant-id <created_variant_id>
quantforge compare strategy.qf.json --all-variants
```

If Ollama or the selected model is unavailable, the AI planner should fail
clearly. Non-AI commands and hardcoded modification workflows should still work.

See [AI Ollama Setup](ai_ollama_setup.md).

## Closing Interpretation

QuantForge evaluates strategy behavior and tradeoffs for a user-provided
hypothesis. Filters can reduce participation and change risk exposure, but they
do not guarantee lower drawdown or future performance. No strategy
recommendation is being made.
