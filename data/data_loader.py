"""Load and clean the accident dataset for the dashboard.

Mirrors the cleaning steps in notebooks/01_eda_and_baseline.ipynb so the
dashboard and the notebook operate on identical data.
"""

import pathlib
import pandas as pd

_PDF_CACHE = None

_DATA_PATH = pathlib.Path(__file__).parent / "raw" / "accidentsData.csv"

_REQUIRED_COLS = [
    "Severity", "Start_Time", "Temperature(F)", "Humidity(%)",
    "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
    "Weather_Condition", "Traffic_Signal", "Junction", "Crossing",
]

_BOOL_ROAD_COLS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal", "Turning_Loop",
]

_DAYS_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


def load_accident_data() -> pd.DataFrame:
    """Load, clean, and cache the balanced accident CSV.

    Returns the same DataFrame structure produced by the notebook's
    cleaning section (section 2 + section 3.1).
    """
    global _PDF_CACHE
    if _PDF_CACHE is not None:
        return _PDF_CACHE

    print(f"Loading {_DATA_PATH} ...")
    df = pd.read_csv(_DATA_PATH)
    df = df.dropna(subset=_REQUIRED_COLS)

    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["Hour"] = df["Start_Time"].dt.hour
    df["DayOfWeek"] = df["Start_Time"].dt.day_name()

    for col in _BOOL_ROAD_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    print(f"Loaded {len(df):,} rows after cleaning.")
    _PDF_CACHE = df
    return df


def get_weather_options(pdf: pd.DataFrame) -> list[dict]:
    """Return sorted weather condition options for a Dash Dropdown."""
    conditions = sorted(pdf["Weather_Condition"].dropna().unique().tolist())
    return [{"label": c, "value": c} for c in conditions]


def get_slider_bounds(pdf: pd.DataFrame) -> dict:
    """Return (min, max) tuples for each numerical predictor feature."""
    return {
        "temperature": (round(float(pdf["Temperature(F)"].min()), 1),
                        round(float(pdf["Temperature(F)"].max()), 1)),
        "humidity":    (round(float(pdf["Humidity(%)"].min()), 1),
                        round(float(pdf["Humidity(%)"].max()), 1)),
        "pressure":    (round(float(pdf["Pressure(in)"].min()), 2),
                        round(float(pdf["Pressure(in)"].max()), 2)),
        "visibility":  (round(float(pdf["Visibility(mi)"].min()), 1),
                        round(float(pdf["Visibility(mi)"].max()), 1)),
        "wind_speed":  (0.0, 60.0),  # cap at 60 mph; raw data has sensor outliers up to 822 mph
    }
