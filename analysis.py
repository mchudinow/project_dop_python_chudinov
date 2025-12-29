import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL


def load_data(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df = df.sort_values("timestamp")

    df["rolling_mean"] = (
        df.groupby("city")["temperature"]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_std"] = (
        df.groupby("city")["temperature"]
        .rolling(window, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    return df


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df["anomaly"] = (
        (df["temperature"] > df["rolling_mean"] + 2 * df["rolling_std"]) |
        (df["temperature"] < df["rolling_mean"] - 2 * df["rolling_std"])
    )
    return df


def seasonal_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["city", "season"])["temperature"]
        .agg(mean="mean", std="std")
        .reset_index()
    )


def stl_decomposition(df_city: pd.DataFrame) -> pd.DataFrame:
    """
    STL-декомпозиция: trend / seasonality / residuals
    """
    ts = df_city.set_index("timestamp")["temperature"]

    stl = STL(ts, period=365)
    result = stl.fit()

    return pd.DataFrame({
        "timestamp": ts.index,
        "trend": result.trend,
        "seasonal": result.seasonal,
        "resid": result.resid
    })
