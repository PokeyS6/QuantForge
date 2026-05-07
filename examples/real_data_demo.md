# Real-Data Demo Runner

Run the local CLI demo with real historical OHLCV data:

- Default: downloaded via yfinance/Yahoo Finance
- Recommended: user-provided CSV for stability

```bash
python scripts/run_real_data_demo.py --ticker SPY --start 2018-01-01
```

Optional examples:

```bash
python scripts/run_real_data_demo.py --ticker QQQ --start 2020-01-01
python scripts/run_real_data_demo.py --ticker TSLA --start 2019-01-01
```

Yahoo/yfinance may occasionally rate-limit requests with errors such as
`YFRateLimitError: Too Many Requests`. If that happens, retry later or provide a
local OHLCV CSV instead. When `--data-csv` is used, the report labels the data
source as user-provided CSV rather than yfinance:

```bash
python scripts/run_real_data_demo.py --ticker SPY --start 2018-01-01 --data-csv path/to/spy_ohlcv.csv
```

The script writes all generated data, project artifacts, variant analyses, comparison
output, and `real_data_demo_report.md` under the selected output directory. The
output directory is safe to delete after inspection.

This runner is demo/validation tooling. It uses historical data for research and
does not provide financial advice or trading recommendations.

## Optional AI Setup

The real-data runner does not require AI assistance. If you want to demonstrate
`quantforge modify --ai`, configure a local Ollama model first:

```bash
export QUANTFORGE_LOCAL_LLM_MODEL=<model-name>
quantforge modify strategy.qf.json --ai "Add a momentum filter with a short lookback and low threshold"
```

For controlled demos, use this curated prompt for consistency:

```text
Add a momentum filter with a short lookback and low threshold
```

It is intended to create an AI-assisted momentum filter variant that changes
historical participation metrics on the demo dataset. It is not an optimization
request, trading recommendation, or performance guarantee.

AI-assisted modification planning follows this local boundary:

```text
user prompt -> local LLM -> validated JSON spec -> deterministic builder -> auditable variant
```

If Ollama or the model is unavailable, the AI planner should fail clearly.
Non-AI workflows should still work. The AI does not generate executable strategy
code, recommend trades, or optimize strategies.
