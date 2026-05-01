"""Feature and label helpers for the ML price-floor variant."""

import pandas as pd
from sklearn.linear_model import LogisticRegression


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "rolling_volatility_30d",
    "distance_from_sma_20",
]
SIGNAL_COLUMNS = ["close", "rsi", "entry_signal", "exit_signal"]


def _validate_prices(prices: pd.DataFrame) -> None:
    if prices.empty:
        raise ValueError("Prices cannot be empty.")
    if "close" not in prices.columns:
        raise ValueError("Prices must include a close column.")


def build_ml_price_floor_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic price-floor features from historical close prices."""
    _validate_prices(prices)
    close = prices["close"]

    features = pd.DataFrame(index=prices.index)
    features["return_1d"] = close.pct_change(1)
    features["return_5d"] = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["rolling_volatility_30d"] = close.pct_change().rolling(30).std()
    features["distance_from_sma_20"] = close / close.rolling(20).mean() - 1
    return features[FEATURE_COLUMNS]


def build_downside_risk_labels(
    prices: pd.DataFrame,
    forecast_horizon_days: int = 5,
    downside_threshold: float = -0.03,
) -> pd.Series:
    """Build downside-risk labels using future close-price drawdowns."""
    _validate_prices(prices)
    if forecast_horizon_days <= 0:
        raise ValueError("forecast_horizon_days must be greater than 0.")

    close = prices["close"]
    future_closes = pd.concat(
        [close.shift(-offset) for offset in range(1, forecast_horizon_days + 1)],
        axis=1,
    )
    future_min_return = future_closes.min(axis=1) / close - 1
    labels = (future_min_return <= downside_threshold).astype(float)
    labels[~future_closes.notna().all(axis=1)] = pd.NA
    labels.name = "downside_risk"
    return labels


def compute_walk_forward_downside_risk_predictions(
    features: pd.DataFrame,
    labels: pd.Series,
    min_training_rows: int = 252,
) -> pd.Series:
    """Compute walk-forward downside-risk probabilities without lookahead."""
    if features.empty:
        raise ValueError("Features cannot be empty.")
    if labels.empty:
        raise ValueError("Labels cannot be empty.")
    if not features.index.equals(labels.index):
        raise ValueError("Features and labels indexes must match.")
    if min_training_rows <= 0:
        raise ValueError("min_training_rows must be greater than 0.")

    predictions = pd.Series(float("nan"), index=features.index, name="p_downside_risk")
    for row_number, row_label in enumerate(features.index):
        training_features = features.iloc[:row_number]
        training_labels = labels.iloc[:row_number]
        training_data = training_features.copy()
        training_data["downside_risk"] = training_labels
        training_data = training_data.dropna()

        if len(training_data) < min_training_rows:
            continue

        y_train = training_data["downside_risk"]
        if y_train.nunique() < 2:
            continue

        current_features = features.loc[[row_label]]
        if current_features.isna().any(axis=None):
            continue

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(training_data[features.columns], y_train)
        class_index = list(model.classes_).index(1.0)
        predictions.loc[row_label] = model.predict_proba(current_features)[0][class_index]

    return predictions


def apply_ml_price_floor_filter_to_signals(
    signals: pd.DataFrame,
    p_downside_risk: pd.Series,
    risk_probability_threshold: float = 0.5,
) -> pd.DataFrame:
    """Apply ML downside-risk entry filtering to RSI signals."""
    if signals.empty:
        raise ValueError("Signals cannot be empty.")
    if p_downside_risk.empty:
        raise ValueError("Predictions cannot be empty.")
    missing_signal_columns = set(SIGNAL_COLUMNS) - set(signals.columns)
    if missing_signal_columns:
        raise ValueError("Signals must include close, rsi, entry_signal, and exit_signal columns.")
    if not signals.index.equals(p_downside_risk.index):
        raise ValueError("Signals and predictions indexes must match.")
    if not 0 <= risk_probability_threshold <= 1:
        raise ValueError("risk_probability_threshold must be between 0 and 1.")

    ml_entry_allowed = (p_downside_risk <= risk_probability_threshold).fillna(False)
    filtered = signals[SIGNAL_COLUMNS].copy()
    filtered["entry_signal"] = filtered["entry_signal"] & ml_entry_allowed
    filtered["p_downside_risk"] = p_downside_risk
    filtered["ml_entry_allowed"] = ml_entry_allowed
    return filtered
