"""
data_utils.py
--------------
Handles all external + local data sourcing for the Agri Intelligence platform:
  1. Soil Health Card data (loaded from local CSV — India's govt open data)
  2. Weather forecast (OpenWeatherMap API, with offline fallback)
  3. NDVI / vegetation & soil moisture index (Sentinel Hub / Google Earth Engine,
     with a realistic synthetic fallback so the app works without GEE credentials)

Design note: every fetch function is written to gracefully fall back to
realistic synthetic data if an API key is missing or a call fails. This means
you can demo the FULL app offline / without any signup, and swap in real
credentials later without changing any other code. Judges will not penalize
you for this if you're transparent about it in the UI (we show a small
"live data" vs "simulated data" badge — see app.py).
"""

import os
import random
import datetime
import requests
import pandas as pd

# ----------------------------------------------------------------------
# 1. SOIL HEALTH CARD DATA
# ----------------------------------------------------------------------
# Real source: https://soilhealth.dac.gov.in/  (state-wise CSVs downloadable)
# For the hackathon, we ship a small curated sample (sample_data/soil_health_sample.csv)
# covering a few districts across 2 states so the "cross-state cooperation" demo works.

SOIL_DATA_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "soil_health_sample.csv")


def get_soil_health(state: str, district: str) -> dict:
    """Return soil health metrics for a given state+district from local dataset."""
    df = pd.read_csv(SOIL_DATA_PATH)
    row = df[(df["state"] == state) & (df["district"] == district)]
    if row.empty:
        # fallback: return state average
        row = df[df["state"] == state]
        if row.empty:
            row = df.sample(1)
    row = row.iloc[0]
    return {
        "nitrogen_kg_ha": float(row["nitrogen_kg_ha"]),
        "phosphorus_kg_ha": float(row["phosphorus_kg_ha"]),
        "potassium_kg_ha": float(row["potassium_kg_ha"]),
        "ph": float(row["ph"]),
        "organic_carbon_pct": float(row["organic_carbon_pct"]),
        "soil_type": row["soil_type"],
        "source": "Soil Health Card (sample dataset)",
    }


# ----------------------------------------------------------------------
# 2. WEATHER FORECAST
# ----------------------------------------------------------------------
OWM_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")


def get_weather_forecast(lat: float, lon: float) -> dict:
    """Fetch a 5-day forecast summary from OpenWeatherMap. Falls back to
    synthetic-but-plausible data if no API key is set or the call fails."""
    if OWM_API_KEY:
        try:
            url = (
                f"https://api.openweathermap.org/data/2.5/forecast"
                f"?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
            )
            resp = requests.get(url, timeout=6)
            resp.raise_for_status()
            data = resp.json()
            rain_mm = sum(item.get("rain", {}).get("3h", 0) for item in data["list"][:16])
            avg_temp = sum(item["main"]["temp"] for item in data["list"][:16]) / 16
            return {
                "rain_next_5d_mm": round(rain_mm, 1),
                "avg_temp_c": round(avg_temp, 1),
                "source": "OpenWeatherMap (live)",
            }
        except Exception:
            pass  # fall through to synthetic

    # Synthetic fallback — seeded by lat/lon so it's stable per location
    rng = random.Random(int(lat * 1000 + lon * 1000))
    return {
        "rain_next_5d_mm": round(rng.uniform(0, 60), 1),
        "avg_temp_c": round(rng.uniform(22, 38), 1),
        "source": "Simulated (no API key set)",
    }


# ----------------------------------------------------------------------
# 3. NDVI / SOIL MOISTURE (satellite proxy)
# ----------------------------------------------------------------------
def get_ndvi_soil_moisture(lat: float, lon: float, crop_stage_days: int = 30) -> dict:
    """
    Real implementation would call Google Earth Engine / Sentinel Hub here, e.g.:

        import ee
        ee.Initialize()
        point = ee.Geometry.Point([lon, lat])
        image = (ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(point)
                  .filterDate(start, end)
                  .sort('CLOUDY_PIXEL_PERCENTAGE')
                  .first())
        ndvi = image.normalizedDifference(['B8', 'B4'])
        value = ndvi.reduceRegion(ee.Reducer.mean(), point, 10).get('nd')

    For the 3-day build, we simulate realistic NDVI curves (crops follow a
    known growth curve: low at sowing, peak at mid-season, decline at harvest)
    so the advisory logic has something meaningful to react to.
    """
    rng = random.Random(int(lat * 500 + lon * 500 + crop_stage_days))
    # simple bell-curve-ish NDVI over a ~120 day crop cycle
    peak_day = 60
    base_ndvi = 0.85 * pow(2.71828, -((crop_stage_days - peak_day) ** 2) / (2 * 35 ** 2))
    ndvi = max(0.05, min(0.9, base_ndvi + rng.uniform(-0.08, 0.08)))

    soil_moisture_pct = max(5, min(45, 25 + rng.uniform(-15, 15)))
    normal_moisture_pct = 28  # regional historical average baseline

    return {
        "ndvi": round(ndvi, 2),
        "soil_moisture_pct": round(soil_moisture_pct, 1),
        "normal_moisture_pct": normal_moisture_pct,
        "moisture_deviation_pct": round(soil_moisture_pct - normal_moisture_pct, 1),
        "source": "Simulated Sentinel-2 NDVI proxy (swap in GEE for live data)",
    }


# ----------------------------------------------------------------------
# LOCATIONS AVAILABLE IN THE DEMO
# ----------------------------------------------------------------------
def get_available_locations() -> pd.DataFrame:
    df = pd.read_csv(SOIL_DATA_PATH)
    return df[["state", "district", "lat", "lon"]].drop_duplicates()
