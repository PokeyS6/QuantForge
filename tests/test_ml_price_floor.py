import pandas as pd
import pytest

from quantforge.ml.price_floor import (
    build_downside_risk_labels,
    build_ml_price_floor_features,
)


def test_build_ml_price_floor_features_columns_and_index_are_preserved():
    index = pd.date_range("2020-01-01", periods=40)
    prices = pd.DataFrame({"close": range(100, 140)}, index=index)

    features = build_ml_price_floor_features(prices)

    assert list(features.columns) == [
        "return_1d",
        "return_5d",
        "return_10d",
        "rolling_volatility_30d",
        "distance_from_sma_20",
    ]
    assert features.index.equals(prices.index)


def test_build_ml_price_floor_features_values_are_deterministic():
    index = pd.date_range("2020-01-01", periods=40)
    close = pd.Series(range(100, 140), index=index)
    prices = pd.DataFrame({"close": close})

    features = build_ml_price_floor_features(prices)

    expected = pd.DataFrame(index=index)
    expected["return_1d"] = close.pct_change(1)
    expected["return_5d"] = close.pct_change(5)
    expected["return_10d"] = close.pct_change(10)
    expected["rolling_volatility_30d"] = close.pct_change().rolling(30).std()
    expected["distance_from_sma_20"] = close / close.rolling(20).mean() - 1

    pd.testing.assert_frame_equal(features, expected)


def test_build_downside_risk_labels_values_and_tail_nans_are_correct():
    index = pd.date_range("2020-01-01", periods=7)
    prices = pd.DataFrame(
        {"close": [100, 98, 97, 101, 96, 95, 110]},
        index=index,
    )

    labels = build_downside_risk_labels(
        prices,
        forecast_horizon_days=3,
        downside_threshold=-0.03,
    )

    expected = pd.Series(
        [1.0, 0.0, 0.0, 1.0, float("nan"), float("nan"), float("nan")],
        index=index,
        name="downside_risk",
        dtype="float64",
    )
    pd.testing.assert_series_equal(labels, expected)


def test_ml_price_floor_helpers_missing_close_raise():
    prices = pd.DataFrame({"open": [100, 101]})

    with pytest.raises(ValueError, match="Prices must include a close column."):
        build_ml_price_floor_features(prices)
    with pytest.raises(ValueError, match="Prices must include a close column."):
        build_downside_risk_labels(prices)


def test_ml_price_floor_helpers_empty_prices_raise():
    prices = pd.DataFrame({"close": []})

    with pytest.raises(ValueError, match="Prices cannot be empty."):
        build_ml_price_floor_features(prices)
    with pytest.raises(ValueError, match="Prices cannot be empty."):
        build_downside_risk_labels(prices)


def test_build_downside_risk_labels_invalid_horizon_raises():
    prices = pd.DataFrame({"close": [100, 101]})

    with pytest.raises(ValueError, match="forecast_horizon_days must be greater than 0."):
        build_downside_risk_labels(prices, forecast_horizon_days=0)
