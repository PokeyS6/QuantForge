from pathlib import Path

import pandas as pd
import pytest

from quantforge.data.csv_loader import load_ohlcv_csv


def test_load_ohlcv_csv_sorts_and_returns_expected_columns(tmp_path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date,Open,High,Low,Close,Volume",
                "2020-01-03,102,103,101,102.5,3000",
                "2020-01-01,100,101,99,100.5,1000",
                "2020-01-02,101,102,100,101.5,2000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    data = load_ohlcv_csv(csv_path)

    assert list(data.columns) == ["open", "high", "low", "close", "volume"]
    assert list(data.index) == list(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    assert data.loc[pd.Timestamp("2020-01-01"), "close"] == 100.5


def test_load_ohlcv_csv_raises_for_missing_required_columns(tmp_path):
    csv_path = Path(tmp_path / "missing.csv")
    csv_path.write_text(
        "date,open,high,low,close\n2020-01-01,100,101,99,100.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="volume"):
        load_ohlcv_csv(csv_path)
