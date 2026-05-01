import pandas as pd
import pytest

from quantforge.ml.price_floor import (
    build_downside_risk_labels,
    build_ml_price_floor_features,
    compute_walk_forward_downside_risk_predictions,
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


def _prediction_inputs(periods=12):
    index = pd.date_range("2020-01-01", periods=periods)
    features = pd.DataFrame(
        {
            "return_1d": [value / 100 for value in range(periods)],
            "return_5d": [value / 90 for value in range(periods)],
            "return_10d": [value / 80 for value in range(periods)],
            "rolling_volatility_30d": [0.01 + value / 1000 for value in range(periods)],
            "distance_from_sma_20": [value / 70 for value in range(periods)],
        },
        index=index,
    )
    labels = pd.Series(
        [float(value % 2) for value in range(periods)],
        index=index,
        name="downside_risk",
    )
    return features, labels


def test_walk_forward_predictions_index_name_and_probability_range():
    features, labels = _prediction_inputs()

    predictions = compute_walk_forward_downside_risk_predictions(
        features,
        labels,
        min_training_rows=4,
    )

    assert predictions.index.equals(features.index)
    assert predictions.name == "p_downside_risk"
    assert predictions.iloc[:4].isna().all()
    available = predictions.dropna()
    assert not available.empty
    assert ((available >= 0) & (available <= 1)).all()


def test_walk_forward_predictions_one_class_training_window_produces_nan():
    features, labels = _prediction_inputs(periods=8)
    labels.iloc[:4] = 0.0

    predictions = compute_walk_forward_downside_risk_predictions(
        features,
        labels,
        min_training_rows=4,
    )

    assert pd.isna(predictions.iloc[4])


def test_walk_forward_predictions_invalid_inputs_raise():
    features, labels = _prediction_inputs()

    with pytest.raises(ValueError, match="Features cannot be empty."):
        compute_walk_forward_downside_risk_predictions(features.iloc[0:0], labels)
    with pytest.raises(ValueError, match="Labels cannot be empty."):
        compute_walk_forward_downside_risk_predictions(features, labels.iloc[0:0])
    with pytest.raises(ValueError, match="Features and labels indexes must match."):
        compute_walk_forward_downside_risk_predictions(
            features,
            labels.rename(index={labels.index[0]: pd.Timestamp("1999-01-01")}),
        )
    with pytest.raises(ValueError, match="min_training_rows must be greater than 0."):
        compute_walk_forward_downside_risk_predictions(
            features,
            labels,
            min_training_rows=0,
        )


def test_walk_forward_predictions_do_not_train_on_current_or_future_rows(monkeypatch):
    features, labels = _prediction_inputs(periods=7)
    fit_indexes = []

    class FakeLogisticRegression:
        def __init__(self, max_iter, random_state):
            self.classes_ = [0.0, 1.0]

        def fit(self, x_train, y_train):
            fit_indexes.append(x_train.index.copy())
            return self

        def predict_proba(self, x_current):
            return [[0.25, 0.75]]

    monkeypatch.setattr(
        "quantforge.ml.price_floor.LogisticRegression",
        FakeLogisticRegression,
    )

    predictions = compute_walk_forward_downside_risk_predictions(
        features,
        labels,
        min_training_rows=2,
    )

    predicted_indexes = list(predictions.dropna().index)
    assert predicted_indexes
    assert len(fit_indexes) == len(predicted_indexes)
    for training_index, prediction_index in zip(fit_indexes, predicted_indexes, strict=True):
        assert training_index.max() < prediction_index
