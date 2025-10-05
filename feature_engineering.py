import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict


def create_time_features(date: datetime) -> Dict:
    """Create time-based features from a date (16 features)"""
    return {
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "day_of_week": date.weekday(),
        "day_of_year": date.timetuple().tm_yday,
        "week_of_year": date.isocalendar()[1],
        "quarter": (date.month - 1) // 3 + 1,
        "month_sin": np.sin(2 * np.pi * date.month / 12),
        "month_cos": np.cos(2 * np.pi * date.month / 12),
        "day_sin": np.sin(2 * np.pi * date.weekday() / 7),
        "day_cos": np.cos(2 * np.pi * date.weekday() / 7),
        "is_weekend": int(date.weekday() >= 5),
        "is_month_start": int(date.day == 1),
        "is_month_end": int(date.day >= 28),
        "is_quarter_start": int(date.month in [1, 4, 7, 10] and date.day == 1),
        "is_quarter_end": int(date.month in [3, 6, 9, 12] and date.day >= 28),
    }


def create_lag_features(
    recent_history: pd.DataFrame, lags: list = [1, 7, 14, 30]
) -> Dict:
    """
    Create lag features from historical predictions (4 features)
    """
    if recent_history.empty:
        return {f"ndvi_mean_lag_{lag}": 0.3 for lag in lags}

    recent_history = recent_history.sort_values("date")
    ndvi_values = recent_history["ndvi_score"].values

    lag_features = {}
    for lag in lags:
        if len(ndvi_values) >= lag:
            lag_features[f"ndvi_mean_lag_{lag}"] = ndvi_values[-lag]
        else:
            lag_features[f"ndvi_mean_lag_{lag}"] = np.mean(ndvi_values)

    return lag_features


def create_rolling_features(
    recent_history: pd.DataFrame, windows: list = [7, 14, 30]
) -> Dict:
    """
    Create rolling window features from historical predictions (12 features)
    """
    if recent_history.empty:
        rolling_features = {}
        for window in windows:
            rolling_features[f"ndvi_mean_rolling_mean_{window}"] = 0.3
            rolling_features[f"ndvi_mean_rolling_std_{window}"] = 0.1
            rolling_features[f"ndvi_mean_rolling_min_{window}"] = 0.2
            rolling_features[f"ndvi_mean_rolling_max_{window}"] = 0.4
        return rolling_features

    recent_history = recent_history.sort_values("date")
    ndvi_series = recent_history["ndvi_score"]

    rolling_features = {}
    for window in windows:
        if len(ndvi_series) >= window:
            window_data = ndvi_series.iloc[-window:]
            rolling_features[f"ndvi_mean_rolling_mean_{window}"] = window_data.mean()
            rolling_features[f"ndvi_mean_rolling_std_{window}"] = window_data.std()
            rolling_features[f"ndvi_mean_rolling_min_{window}"] = window_data.min()
            rolling_features[f"ndvi_mean_rolling_max_{window}"] = window_data.max()
        else:
            rolling_features[f"ndvi_mean_rolling_mean_{window}"] = ndvi_series.mean()
            rolling_features[f"ndvi_mean_rolling_std_{window}"] = ndvi_series.std()
            rolling_features[f"ndvi_mean_rolling_min_{window}"] = ndvi_series.min()
            rolling_features[f"ndvi_mean_rolling_max_{window}"] = ndvi_series.max()

    return rolling_features


def create_satellite_features(recent_history: pd.DataFrame) -> Dict:
    """
    Create satellite-derived features (5 features)

    These were in the original CSV but not created in the API.
    We'll use default values or compute from recent history.

    Args:
        recent_history: DataFrame with recent NDVI predictions
                       (may contain ndvi_median, ndvi_std, etc. if available)

    Returns:
        Dictionary with 5 satellite features
    """
    if recent_history.empty or "ndvi_median" not in recent_history.columns:
        # Use defaults based on typical NDVI statistics
        return {
            "ndvi_median": 0.3,  # Typical median NDVI
            "ndvi_std": 0.1,  # Standard deviation
            "ndvi_min": 0.15,  # Minimum NDVI
            "ndvi_max": 0.45,  # Maximum NDVI
            "valid_pixels": 1000,  # Default pixel count
        }

    # If history has these columns, use most recent values
    recent_history = recent_history.sort_values("date")
    return {
        "ndvi_median": (
            recent_history["ndvi_median"].iloc[-1]
            if "ndvi_median" in recent_history.columns
            else 0.3
        ),
        "ndvi_std": (
            recent_history["ndvi_std"].iloc[-1]
            if "ndvi_std" in recent_history.columns
            else 0.1
        ),
        "ndvi_min": (
            recent_history["ndvi_min"].iloc[-1]
            if "ndvi_min" in recent_history.columns
            else 0.15
        ),
        "ndvi_max": (
            recent_history["ndvi_max"].iloc[-1]
            if "ndvi_max" in recent_history.columns
            else 0.45
        ),
        "valid_pixels": (
            recent_history["valid_pixels"].iloc[-1]
            if "valid_pixels" in recent_history.columns
            else 1000
        ),
    }


def create_all_features(date: datetime, recent_history: pd.DataFrame) -> pd.DataFrame:
    """
    Create all 37 features for a prediction

    Feature breakdown:
    - 5 satellite features (ndvi_median, ndvi_std, ndvi_min, ndvi_max, valid_pixels)
    - 16 time features (year, month, day, cyclical encodings, etc.)
    - 4 lag features (lag_1, lag_7, lag_14, lag_30)
    - 12 rolling features (mean, std, min, max for windows 7, 14, 30)

    Total: 5 + 16 + 4 + 12 = 37 features

    Args:
        date: Prediction date
        recent_history: DataFrame with recent NDVI predictions

    Returns:
        DataFrame with 1 row and 37 feature columns IN CORRECT ORDER
    """
    #  1. Satellite features (5) - MUST BE FIRST
    features = create_satellite_features(recent_history)

    #  2. Time features (16)
    time_features = create_time_features(date)
    features.update(time_features)

    #  3. Lag features (4)
    lag_features = create_lag_features(recent_history)
    features.update(lag_features)

    #  4. Rolling features (12)
    rolling_features = create_rolling_features(recent_history)
    features.update(rolling_features)

    # Convert to DataFrame
    feature_df = pd.DataFrame([features])

    #  CRITICAL: Ensure features are in the SAME ORDER as training
    expected_order = [
        # Satellite features (5)
        "ndvi_median",
        "ndvi_std",
        "ndvi_min",
        "ndvi_max",
        "valid_pixels",
        # Time features (16)
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "quarter",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        # Lag features (4)
        "ndvi_mean_lag_1",
        "ndvi_mean_lag_7",
        "ndvi_mean_lag_14",
        "ndvi_mean_lag_30",
        # Rolling features (12)
        "ndvi_mean_rolling_mean_7",
        "ndvi_mean_rolling_std_7",
        "ndvi_mean_rolling_min_7",
        "ndvi_mean_rolling_max_7",
        "ndvi_mean_rolling_mean_14",
        "ndvi_mean_rolling_std_14",
        "ndvi_mean_rolling_min_14",
        "ndvi_mean_rolling_max_14",
        "ndvi_mean_rolling_mean_30",
        "ndvi_mean_rolling_std_30",
        "ndvi_mean_rolling_min_30",
        "ndvi_mean_rolling_max_30",
    ]

    # Reorder columns to match training
    feature_df = feature_df[expected_order]

    # Debug print
    print(f"API Generated {len(feature_df.columns)} features")
    if len(feature_df.columns) != 37:
        print(f"WARNING: Expected 37 features, got {len(feature_df.columns)}")

    return feature_df
