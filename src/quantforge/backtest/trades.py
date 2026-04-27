"""Trade record placeholders for QuantForge backtests."""

import pandas as pd


def build_long_positions(signals: pd.DataFrame) -> pd.Series:
    """Build a deterministic long/flat position series from entry and exit signals."""
    required_columns = ["entry_signal", "exit_signal"]
    missing_columns = [column for column in required_columns if column not in signals.columns]
    if missing_columns:
        raise ValueError(f"Missing required signal columns: {', '.join(missing_columns)}")

    positions = []
    current_position = 0
    for _, row in signals.iterrows():
        if bool(row["exit_signal"]):
            current_position = 0
        elif bool(row["entry_signal"]):
            current_position = 1
        positions.append(current_position)

    return pd.Series(positions, index=signals.index, name="position")
