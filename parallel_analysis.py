import pandas as pd
from joblib import Parallel, delayed
from analysis import add_rolling_features, detect_anomalies


def _process_city(df_city: pd.DataFrame) -> pd.DataFrame:
    """
    Анализ одного города
    """
    df_city = add_rolling_features(df_city)
    df_city = detect_anomalies(df_city)
    return df_city


def run_parallel_analysis(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Параллельный анализ по городам
    """
    cities = df["city"].unique()

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_city)(df[df["city"] == city]) for city in cities
    )

    return pd.concat(results).sort_values(["city", "timestamp"])
