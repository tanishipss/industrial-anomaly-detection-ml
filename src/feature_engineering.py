import pandas as pd
from src.config import SENSOR_COLS, ROLL_WINDOWS, LAG_STEPS, ZSCORE_WINDOW

def create_features(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df["hour"] = df["Date"].dt.hour
    df["day"] = df["Date"].dt.day
    df["weekday"] = df["Date"].dt.weekday

    for col in SENSOR_COLS:
        for w in ROLL_WINDOWS:
            df[f"{col}_mean_{w}"] = df[col].rolling(w).mean()
            df[f"{col}_std_{w}"]  = df[col].rolling(w).std()
            df[f"{col}_max_{w}"]  = df[col].rolling(w).max()
            df[f"{col}_min_{w}"]  = df[col].rolling(w).min()

        for lag in LAG_STEPS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

        df[f"{col}_abs_diff1"] = (df[col] - df[col].shift(1)).abs()

        df[f"{col}_z"] = (
            df[col] - df[col].rolling(ZSCORE_WINDOW).mean()
        ) / (df[col].rolling(ZSCORE_WINDOW).std() + 1e-6)

    df.fillna(method="bfill", inplace=True)
    return df
