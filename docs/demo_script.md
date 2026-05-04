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

## Closing Interpretation

QuantForge evaluates strategy behavior and tradeoffs for a user-provided
hypothesis. Filters can reduce participation and change risk exposure, but they
do not guarantee lower drawdown or future performance. No strategy
recommendation is being made.
