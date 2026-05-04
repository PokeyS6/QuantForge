# Demo Findings

The SPY real-data demo compares three historical analyses:

- Baseline RSI mean-reversion behavior.
- A volatility-filter variant that reduces participation during high-volatility regimes.
- An ML price-floor variant that blocks entries when walk-forward downside-risk probability is above the configured threshold.

## Key Insight

Filters reduce participation, not guaranteed risk.

In the SPY demo, the variants changed trade count, exposure, drawdown, and total
return relative to the baseline. Those changes describe behavior in the tested
historical window only. A filter may reduce entries or exposure without
guaranteeing lower drawdown or future performance.

The exact metrics should be read from the generated `real_data_demo_report.md`
for the specific run, ticker, date range, and data source.
