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

## Manual E2E Path

The intended demo CSV is:

```text
real_data_csvs/spy_ohlcv.csv
```

It contains the required OHLCV columns (`date`, `open`, `high`, `low`, `close`,
`volume`) and enough rows for baseline and variant tests.

Create the demo project with:

```bash
quantforge create "SPY RSI reversal" --ticker SPY --start 2020-01-01
```

This creates:

```text
spy-rsi-reversal
```

Outside-project path style:

```bash
quantforge analyze spy-rsi-reversal/strategy.qf.json --data-csv real_data_csvs/spy_ohlcv.csv
```

Inside-project path style:

```bash
cd spy-rsi-reversal
quantforge analyze strategy.qf.json --data-csv ../../real_data_csvs/spy_ohlcv.csv
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
brew install ollama
ollama serve
ollama pull llama3
export QUANTFORGE_LOCAL_LLM_MODEL=llama3
quantforge modify strategy.qf.json --ai "Add a momentum filter using a 5 day lookback and -0.05 threshold"
```

Curated demo prompt:

```text
Add a momentum filter using a 5 day lookback and -0.05 threshold
```

This prompt is used for demo consistency, not optimization. It is intended to
create an AI-assisted `momentum_filter` variant that can filter RSI entries
based on recent momentum confirmation. On the demo dataset, the created variant
may change trade count, exposure, and/or return/drawdown metrics relative to the
baseline. Those changes describe historical behavior only; they are not a
trading recommendation or a performance guarantee.

In validation trials on `real_data_csvs/spy_ohlcv.csv`, this prompt created a
momentum filter with `lookback_days=5` and `threshold=-0.05`. The historical
test changed completed trades from 9 to 6 and changed exposure and
return/drawdown metrics relative to the baseline.

After the variant is created, note the created variant id from the CLI output
and run:

```bash
quantforge analyze strategy.qf.json --data-csv ../../real_data_csvs/spy_ohlcv.csv --variant-id <created_variant_id>
quantforge compare strategy.qf.json --all-variants
```

If `reports/comparison_report.md` already exists, compare refuses to overwrite
it. Before rerunning compare from inside `spy-rsi-reversal`, remove the existing
comparison report:

```bash
rm reports/comparison_report.md
```

The AI momentum variant may produce 0 trades or unchanged metrics on the SPY
demo CSV. This means the filter removed all entries or all persisted entries
passed the filter; it does not mean the workflow failed.

If Ollama or the selected model is unavailable, the AI planner should fail
clearly. Non-AI commands and hardcoded modification workflows should still work.
The expected missing-model message is:

```text
ERROR: Local AI planner unavailable. Configure a local model before using AI-assisted modifications.
```

See [AI Ollama Setup](ai_ollama_setup.md).

## Closing Interpretation

QuantForge evaluates strategy behavior and tradeoffs for a user-provided
hypothesis. Filters can reduce participation and change risk exposure, but they
do not guarantee lower drawdown or future performance. No strategy
recommendation is being made.
