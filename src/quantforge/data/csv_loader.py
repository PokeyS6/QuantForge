"""CSV data loader placeholder for QuantForge."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Load local OHLCV data from a CSV file."""
    data = pd.read_csv(path)
    data.columns = [column.lower() for column in data.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing_columns)}")

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").set_index("date")
    return data[OHLCV_COLUMNS]
