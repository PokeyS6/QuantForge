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
