from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_ZIP = ROOT / "Dataset.zip"
OUT_DIR = ROOT / "data"
RAW_SOLAR_NAME = "Solar station site 1 (Nominal capacity-50MW).xlsx"


RENAME = {
    "Time(year-month-day h:m:s)": "timestamp",
    "Total solar irradiance (W/m2)": "total_irradiance",
    "Direct normal irradiance (W/m2)": "direct_normal_irradiance",
    "Global horizontal irradiance (W/m2)": "global_horizontal_irradiance",
    "Air temperature  (°C) ": "air_temperature",
    "Atmosphere (hpa)": "pressure",
    "Relative humidity (%)": "relative_humidity",
    "Power (MW)": "power",
}


def cyclical(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * series.astype(float) / period
    return np.sin(angle), np.cos(angle)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATASET_ZIP) as zf:
        with zf.open(RAW_SOLAR_NAME) as fh:
            df = pd.read_excel(fh, sheet_name="sheet1")

    original_rows = int(len(df))
    df = df.rename(columns=RENAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in [c for c in df.columns if c != "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_timestamps = int(df["timestamp"].isna().sum())
    duplicate_timestamps = int(df.duplicated("timestamp").sum())
    missing_before = df.isna().sum().to_dict()

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.ffill().bfill()

    dt = df["timestamp"]
    hour_float = dt.dt.hour + dt.dt.minute / 60.0
    day_of_year = dt.dt.dayofyear
    month = dt.dt.month
    df["hour_sin"], df["hour_cos"] = cyclical(hour_float, 24)
    df["doy_sin"], df["doy_cos"] = cyclical(day_of_year, 366)
    df["month_sin"], df["month_cos"] = cyclical(month, 12)

    engineered_cols = [
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "month_sin",
        "month_cos",
    ]
    real_cols = [
        "total_irradiance",
        "direct_normal_irradiance",
        "global_horizontal_irradiance",
        "air_temperature",
        "pressure",
        "relative_humidity",
        "power",
    ]
    df = df[["timestamp", *real_cols, *engineered_cols]]

    processed_path = OUT_DIR / "solar_processed.csv"
    df.to_csv(processed_path, index=False)

    summary = {
        "source_file": RAW_SOLAR_NAME,
        "processed_file": str(processed_path.relative_to(ROOT)),
        "original_rows": original_rows,
        "processed_rows": int(len(df)),
        "invalid_timestamps_removed": invalid_timestamps,
        "duplicate_timestamps_removed": duplicate_timestamps,
        "missing_values_before_interpolation": {
            str(k): int(v) for k, v in missing_before.items()
        },
        "missing_values_after_processing": {
            str(k): int(v) for k, v in df.isna().sum().to_dict().items()
        },
        "start_timestamp": str(df["timestamp"].min()),
        "end_timestamp": str(df["timestamp"].max()),
        "frequency_minutes": 15,
        "nominal_capacity_mw": 50,
        "columns": list(df.columns),
    }
    (OUT_DIR / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
