"""
L0 - STATE-LEVEL DAILY WEATHER, 1966-2024 (long-record, weather-only pipeline)
==============================================================================

Purpose
-------
ICRISAT gives wheat yields for 1966-2017 on FIXED 1966 districts (later splits are merged), so the
clean analysis unit for the long record is the STATE. This script exports, for each harvest year Y,
daily state-mean weather for Oct 1 (Y-1) .. May 15 (Y), from ERA5-Land (the only source covering
1966-2024 consistently; CHIRPS starts in 1981, MODIS in 2000).

    bands: Tmax, Tmin, dew point, solar radiation, soil moisture (layer 1), precipitation (all daily)
    units: states as in FAO GAUL 2015 level 1 (Punjab, Haryana, Uttar Pradesh, Madhya Pradesh,
           Chhattisgarh, Rajasthan).  Chhattisgarh is exported separately because ICRISAT's
           undivided Madhya Pradesh may include its districts - decide when you see the MP file.
    spatial mean over the WHEAT REGION only: the state's ERA5 cells are kept where the multi-year
    winter-crop fraction (MODIS cropland x winter-green signature, built as in A0) exceeds MIN_FRAC.
    This stops e.g. the Thar desert from dominating the Rajasthan mean. The winter-crop mask is
    static (2003-2024), a modelling assumption for earlier decades.

One CSV per year:  lt_era5state_YYYY.csv   (columns: ADM1_NAME, date, <bands>)
Download them into  data/raw/gee_state/

Run (next to extract_gee_features.py, from which the Earth Engine login is reused)
---------------------------------------------------------------------------------
    python L0_gee_state_weather_longrecord.py --sanity-check     # are the state names matched?
    python L0_gee_state_weather_longrecord.py --test             # one day: values + wheat fraction per state
    python L0_gee_state_weather_longrecord.py --run              # 59 export tasks
    python L0_gee_state_weather_longrecord.py --task-status
    python L0_gee_state_weather_longrecord.py --run --no-wheat-weight   # plain state mean (fallback)
NOTE: written without Earth Engine access - run --sanity-check and --test first and send me the output.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import ee

from extract_gee_features import initialize_earth_engine

PREFIX = "lt_era5state"
EXPORT_FOLDER = "wheat_longrecord_state_weather"
START_YEAR, END_YEAR = 1966, 2024
STATE_NAMES = ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Chhattisgarh", "Chattisgarh", "Rajasthan"]
BANDS = ["temperature_2m_max", "temperature_2m_min", "dewpoint_temperature_2m",
         "surface_solar_radiation_downwards_sum", "volumetric_soil_water_layer_1", "total_precipitation_sum"]
ERA5 = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_SCALE = 11132
MIN_FRAC = 0.10                 # keep ERA5 cells with >= 10% winter-crop cover (set with --min-frac)
TASK_DELAY_SECONDS = 2


def window(year):
    return f"{year - 1}-10-01", f"{year}-05-16"


def states_fc():
    return (ee.FeatureCollection("FAO/GAUL/2015/level1")
            .filter(ee.Filter.eq("ADM0_NAME", "India"))
            .filter(ee.Filter.inList("ADM1_NAME", STATE_NAMES)))


# ---- winter-crop mask (same rule as A0) resampled to the ERA5-Land grid ----
def winter_crop_mask(start_year=2003, end_year=2024, min_crop_share=0.5, peak_min=0.45, seasonality_min=0.20):
    ndvi = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI")
    lc = (ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1")
          .filterDate(f"{start_year}-01-01", f"{min(end_year, 2023)}-12-31"))
    cropland = lc.map(lambda i: i.eq(12).rename("c")).mean().gte(min_crop_share)
    years = ee.List.sequence(start_year, end_year)

    def peak(y):
        y = ee.Number(y)
        return ndvi.filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 3, 31)).max().multiply(0.0001).rename("p")

    def low(y):
        y = ee.Number(y)
        return ndvi.filterDate(ee.Date.fromYMD(y, 5, 20), ee.Date.fromYMD(y, 6, 20)).mean().multiply(0.0001).rename("l")

    pm = ee.ImageCollection.fromImages(years.map(peak)).median()
    lm = ee.ImageCollection.fromImages(years.map(low)).median()
    return cropland.And(pm.gte(peak_min)).And(pm.subtract(lm).gte(seasonality_min)).rename("mask")


def wheat_fraction():
    proj = ee.ImageCollection(ERA5).first().select("temperature_2m_max").projection()
    m500 = winter_crop_mask().unmask(0).reproject("EPSG:4326", None, 500)
    return m500.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024).reproject(proj).rename("wheat_frac")


def state_table(year, states, frac, weight):
    start, end = window(year)
    col = ee.ImageCollection(ERA5).filterDate(start, end).select(BANDS)

    def per_image(img):
        img = ee.Image(img)
        date = img.date().format("YYYY-MM-dd")
        if weight:
            img = img.updateMask(frac.gt(MIN_FRAC))
        fc = img.reduceRegions(collection=states, reducer=ee.Reducer.mean(), scale=ERA5_SCALE, tileScale=4)
        return fc.map(lambda f: f.set({"date": date}))

    return ee.FeatureCollection(col.map(per_image)).flatten()


def export_year(year, states, frac, weight):
    fc = state_table(year, states, frac, weight).map(lambda f: ee.Feature(None, f.toDictionary()))
    name = f"{PREFIX}_{year}"
    ee.batch.Export.table.toDrive(collection=fc, description=name, folder=EXPORT_FOLDER, fileNamePrefix=name,
                                  fileFormat="CSV", selectors=["ADM1_NAME", "date"] + BANDS).start()
    print("submitted", name)


def main():
    global MIN_FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-check", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--task-status", action="store_true")
    ap.add_argument("--no-wheat-weight", action="store_true", help="plain state mean instead of wheat-region mean")
    ap.add_argument("--min-frac", type=float, default=MIN_FRAC, help="minimum winter-crop fraction of an ERA5 cell to be kept")
    ap.add_argument("--test-date", default="2003-02-01", help="day used by --test (use e.g. 1966-02-01 to check early data)")
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--end-year", type=int, default=END_YEAR)
    a = ap.parse_args()
    if not (a.sanity_check or a.test or a.run or a.task_status):
        ap.error("choose --sanity-check, --test, --run or --task-status")
    MIN_FRAC = a.min_frac
    initialize_earth_engine()

    if a.task_status:
        rows = [t.status() for t in ee.batch.Task.list() if str(t.status().get("description", "")).startswith(PREFIX)]
        counts = {}
        for s in sorted(rows, key=lambda s: s.get("description", "")):
            counts[s["state"]] = counts.get(s["state"], 0) + 1
            if s["state"] == "FAILED":
                print(s["description"], "FAILED:", s.get("error_message"))
        print("summary:", counts)
        return

    states = states_fc()
    if a.sanity_check:
        names = sorted(states.aggregate_array("ADM1_NAME").getInfo())
        print("matched GAUL states:", names)
        print("expected (any spelling of Chhattisgarh counts once):", STATE_NAMES)
        return

    weight = not a.no_wheat_weight
    frac = wheat_fraction() if weight else None
    if a.test:
        d0 = dt.date.fromisoformat(a.test_date)
        d1 = d0 + dt.timedelta(days=1)
        print(f"test day {d0}, MIN_FRAC {MIN_FRAC}")
        img = ee.ImageCollection(ERA5).filterDate(str(d0), str(d1)).first().select(BANDS)
        if weight:
            img = img.addBands(frac)
            print("wheat fraction per state (mean over the state's ERA5 cells):")
            for f in frac.reduceRegions(states, ee.Reducer.mean(), ERA5_SCALE).getInfo()["features"]:
                print("  ", f["properties"]["ADM1_NAME"], round(f["properties"].get("mean", float("nan")), 3))
            img = ee.Image(img).select(BANDS).updateMask(frac.gt(MIN_FRAC))
        res = img.reduceRegions(states, ee.Reducer.mean(), ERA5_SCALE).getInfo()["features"]
        print(f"\n{a.test_date} state means (Kelvin for temperatures, J/m2 solar, m precipitation):")
        for f in res:
            p = f["properties"]
            print("  ", p["ADM1_NAME"], {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in p.items() if k in BANDS})
        return

    for y in range(a.start_year, a.end_year + 1):
        export_year(y, states, frac, weight)
        time.sleep(TASK_DELAY_SECONDS)
    print(f"submitted {a.end_year - a.start_year + 1} tasks -> Drive/{EXPORT_FOLDER}; download into data/raw/gee_state/")


if __name__ == "__main__":
    main()