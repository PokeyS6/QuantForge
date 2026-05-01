"""Feature and label helpers for the ML price-floor variant."""

import pandas as pd


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "rolling_volatility_30d",
    "distance_from_sma_20",
]


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
