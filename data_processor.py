import io
import numpy as np
import pandas as pd


COLUMN_MAP = {
    "activity date":        "date",
    "activity name":        "name",
    "activity type":        "type",
    "sport type":           "type",
    "distance":             "distance",
    "moving time":          "moving_time",
    "average heart rate":   "average_heartrate",
    "max heart rate":       "max_heartrate",
    "elevation gain":       "elevation_gain",
    "total elevation gain": "elevation_gain",
}


def load_csv(raw_bytes):
    df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    return _clean(df)


def _clean(df):
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    if "type" not in df.columns:
        raise ValueError("Could not find an activity type column in this file.")
    df = df[df["type"].str.lower().isin(["run", "virtualrun"])].copy()

    if df.empty:
        raise ValueError("No running activities found in this file.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["run_number"] = range(1, len(df) + 1)

    df["distance"] = pd.to_numeric(df.get("distance", np.nan), errors="coerce")
    if df["distance"].median() > 200:
        df["distance"] = df["distance"] / 1000
    df = df.rename(columns={"distance": "distance_km"})

    df["moving_time"] = df["moving_time"].apply(_parse_time_to_seconds)
    df["duration_min"] = df["moving_time"] / 60

    df["pace_min_per_km"] = np.where(
        df["distance_km"] > 0,
        df["moving_time"] / 60 / df["distance_km"],
        np.nan,
    )
    df["pace_min_per_km"] = df["pace_min_per_km"].where(
        df["pace_min_per_km"].between(3.0, 15.0)
    )
    df["pace_label"] = df["pace_min_per_km"].apply(_format_pace)

    df["average_heartrate"] = pd.to_numeric(df.get("average_heartrate", np.nan), errors="coerce")
    df["max_heartrate"]     = pd.to_numeric(df.get("max_heartrate",     np.nan), errors="coerce")
    df["elevation_gain"]    = pd.to_numeric(df.get("elevation_gain",    np.nan), errors="coerce")

    df["week_start"]  = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    df["month_start"] = df["date"].dt.to_period("M").apply(lambda p: p.start_time)

    if "name" not in df.columns:
        df["name"] = "Run"

    keep = [
        "run_number", "date", "name",
        "distance_km", "pace_min_per_km", "pace_label",
        "duration_min", "average_heartrate", "max_heartrate",
        "elevation_gain", "week_start", "month_start",
    ]
    return df[[c for c in keep if c in df.columns]]


def _parse_time_to_seconds(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _format_pace(decimal_minutes):
    if pd.isna(decimal_minutes):
        return "–"
    m = int(decimal_minutes)
    s = int(round((decimal_minutes - m) * 60))
    return f"{m}:{s:02d} /km"
