"""
Reads a Strava activities.csv export and returns a clean pandas DataFrame
of running activities only.

How to get your export:
  Strava → Settings → My Account → Download or Delete Your Account → Request Your Archive
  Then open the ZIP and upload the activities.csv file inside it.
"""

import io
import numpy as np
import pandas as pd


# Strava has changed export column names over the years.
# This maps every known variant to one consistent name.
COLUMN_MAP = {
    "activity date":      "date",
    "activity name":      "name",
    "activity type":      "type",
    "sport type":         "type",
    "distance":           "distance",
    "moving time":        "moving_time",
    "average heart rate": "average_heartrate",
    "max heart rate":     "max_heartrate",
    "elevation gain":     "elevation_gain",
    "total elevation gain": "elevation_gain",
}


def load_csv(raw_bytes):
    """
    raw_bytes – the CSV file contents as bytes (from st.file_uploader)
    Returns a pandas DataFrame with one row per run.
    """
    df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    return _clean(df)


def _clean(df):
    """Rename columns, filter to runs, and compute useful new columns."""

    # Lowercase all column names so the mapping works regardless of capitalisation
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)

    # Strava's CSV has duplicate column names (e.g. two "Distance" columns).
    # Keep only the first of each duplicate.
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Keep only running activities
    if "type" not in df.columns:
        raise ValueError("Could not find an activity type column in this file.")
    df = df[df["type"].str.lower().isin(["run", "virtualrun"])].copy()

    if df.empty:
        raise ValueError("No running activities found in this file.")

    # Parse and sort by date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Sequential run number (1 = first run ever)
    df["run_number"] = range(1, len(df) + 1)

    # Distance → km
    df["distance"] = pd.to_numeric(df.get("distance", np.nan), errors="coerce")
    # Strava sometimes exports in metres instead of km — a median above 200 means metres
    if df["distance"].median() > 200:
        df["distance"] = df["distance"] / 1000
    df = df.rename(columns={"distance": "distance_km"})

    # Moving time → seconds, then minutes
    df["moving_time"] = df["moving_time"].apply(_parse_time_to_seconds)
    df["duration_min"] = df["moving_time"] / 60

    # Pace in min/km = time_in_minutes / distance_km
    df["pace_min_per_km"] = np.where(
        df["distance_km"] > 0,
        df["moving_time"] / 60 / df["distance_km"],
        np.nan,
    )
    # Drop impossible paces (faster than 3 min/km or slower than 15 min/km)
    df["pace_min_per_km"] = df["pace_min_per_km"].where(
        df["pace_min_per_km"].between(3.0, 15.0)
    )
    df["pace_label"] = df["pace_min_per_km"].apply(_format_pace)

    # Heart rate and elevation (may not exist in older exports)
    df["average_heartrate"] = pd.to_numeric(df.get("average_heartrate", np.nan), errors="coerce")
    df["max_heartrate"]     = pd.to_numeric(df.get("max_heartrate",     np.nan), errors="coerce")
    df["elevation_gain"]    = pd.to_numeric(df.get("elevation_gain",    np.nan), errors="coerce")

    # Week/month groupings used by the charts
    df["week_start"]  = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    df["month_start"] = df["date"].dt.to_period("M").apply(lambda p: p.start_time)

    if "name" not in df.columns:
        df["name"] = "Run"

    # Return only the columns the dashboard uses
    keep = [
        "run_number", "date", "name",
        "distance_km", "pace_min_per_km", "pace_label",
        "duration_min", "average_heartrate", "max_heartrate",
        "elevation_gain", "week_start", "month_start",
    ]
    return df[[c for c in keep if c in df.columns]]


def _parse_time_to_seconds(value):
    """Convert a time value to seconds. Handles '1:00:00', '5:30', or plain numbers."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:   # MM:SS
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:   # HH:MM:SS
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _format_pace(decimal_minutes):
    """Convert 6.5 → '6:30 /km'"""
    if pd.isna(decimal_minutes):
        return "–"
    m = int(decimal_minutes)
    s = int(round((decimal_minutes - m) * 60))
    return f"{m}:{s:02d} /km"
