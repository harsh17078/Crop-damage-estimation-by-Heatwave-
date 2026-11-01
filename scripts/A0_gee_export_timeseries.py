"""
OPTION A — STEP A0: RAW TIME-SERIES EXPORT FROM GOOGLE EARTH ENGINE
===================================================================

Why time series instead of ready-made window features?
------------------------------------------------------
Wheat does not follow one calendar. MP / Rajasthan wheat peaks and is harvested
weeks before Punjab / Haryana wheat. A fixed "before = Dec-Jan, label = Mar 16-Apr 30"
window would put the whole label window AFTER harvest for MP/Rajasthan.
So this script exports only raw per-district series; A1 builds district-specific,
phenology-anchored windows in Python (where they can be tested and re-tuned).

Per harvest year Y the export covers Oct 1 (Y-1) .. May 15 (Y) and writes 3 CSVs:

    ts_era5_Y   daily : Tmax, Tmin, dew point, solar radiation, soil moisture (Kelvin / raw units)
    ts_chirps_Y daily : precipitation (mm)
    ts_modis_Y  composites (column `source`):
                  veg  = MOD13Q1  NDVI, EVI (SummaryQA <= 1 only)  + pixel counts
                  ndmi = MOD09A1  NDMI
                  lst  = MOD11A2  LST day / night (deg C)
                  gpp  = MOD17A2H GPP (g C m-2 per 8 days)
plus one `ts_mask_coverage` table (share of every district inside the winter-crop mask).

Exposure vs response (the leakage rule)
---------------------------------------
Weather (ERA5, CHIRPS) = EXPOSURE. Vegetation / LST / GPP (MODIS) = RESPONSE.
The damage label is built from the response AFTER the district's peak-greenness date,
so A1 only lets response variables up to that date into the features.

Mask
----
Multi-year winter-crop mask (cropland >= 50% of years AND median Jan-Mar peak NDVI >= 0.45
AND peak - median(late May-mid Jun) >= 0.20). Deliberately NOT a per-year mask and NOT
based on senescence: a per-year senescence rule selects exactly the stressed pixels.
It is winter-crop, not wheat-only (mustard, gram, barley pass).

Run (must sit next to extract_gee_features.py, which supplies the district list)
-------------------------------------------------------------------------------
    python A0_gee_export_timeseries.py --sanity-check
    python A0_gee_export_timeseries.py --run --start-year 2003 --end-year 2024
    python A0_gee_export_timeseries.py --task-status
    python A0_gee_export_timeseries.py --run --only modis      # repair run: MODIS tables only
Download the CSVs into  data/raw/gee_ts/   (NOTE: this file could not be executed where it
was written - run --sanity-check and one year first.)
"""

from __future__ import annotations

import argparse
import time

import ee

from extract_gee_features import get_districts, initialize_earth_engine, sanity_check, TILE_SCALE

EXPORT_FOLDER = "wheat_proxy_timeseries"
PREFIX = "ts"
START_YEAR, END_YEAR = 2003, 2024
TASK_DELAY_SECONDS = 2

ERA5_BANDS = ["temperature_2m_max", "temperature_2m_min", "dewpoint_temperature_2m",
              "surface_solar_radiation_downwards_sum", "volumetric_soil_water_layer_1"]


def window(year):
    return f"{year - 1}-10-01", f"{year}-05-16"


# ----------------------------------------------------------------------
# mask
# ----------------------------------------------------------------------
def winter_crop_mask(start_year=START_YEAR, end_year=END_YEAR, min_crop_share=0.5,
                     peak_min=0.45, seasonality_min=0.20):
    ndvi = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI")
    lc = (ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1")
          .filterDate(f"{start_year}-01-01", f"{min(end_year, 2023)}-12-31"))
    cropland = lc.map(lambda i: i.eq(12).rename("c")).mean().gte(min_crop_share)

    years = ee.List.sequence(start_year, end_year)

    def peak(y):
        y = ee.Number(y)
        return (ndvi.filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 3, 31))
                .max().multiply(0.0001).rename("p"))

    def low(y):
        y = ee.Number(y)
        return (ndvi.filterDate(ee.Date.fromYMD(y, 5, 20), ee.Date.fromYMD(y, 6, 20))
                .mean().multiply(0.0001).rename("l"))

    peak_med = ee.ImageCollection.fromImages(years.map(peak)).median()
    low_med = ee.ImageCollection.fromImages(years.map(low)).median()
    return (cropland.And(peak_med.gte(peak_min))
            .And(peak_med.subtract(low_med).gte(seasonality_min)).rename("mask"))


# ----------------------------------------------------------------------
# generic "one row per district x image" table
# ----------------------------------------------------------------------
def table(col, fn, bands, source, districts, scale, reducer, start, end):
    def per_image(img):
        img = ee.Image(img)
        date = img.date().format("YYYY-MM-dd")
        out = ee.Image(fn(img)).select(bands)
        fc = out.reduceRegions(collection=districts, reducer=reducer, scale=scale,
                               tileScale=TILE_SCALE)
        return fc.map(lambda f: f.set({"date": date, "source": source}))

    return ee.FeatureCollection(col.filterDate(start, end).map(per_image)).flatten()


def strip_geometry(fc):
    return fc.map(lambda f: ee.Feature(None, f.toDictionary()))


ID_COLS = ["PROJECT_STATE", "PROJECT_DISTRICT", "date", "source"]
SELECTORS = {
    "era5": ID_COLS + ERA5_BANDS,
    "chirps": ID_COLS + ["precipitation"],
    # MODIS mixes four sources in one table. Earth Engine builds the CSV header from the FIRST
    # feature only, so without explicit selectors the NDMI / LST / GPP columns were silently lost.
    "modis": ID_COLS + ["NDVI_mean", "NDVI_count", "EVI_mean", "EVI_count", "NDMI",
                        "LST_Day_1km", "LST_Night_1km", "Gpp"],
}


def export(fc, name, kind):
    task = ee.batch.Export.table.toDrive(collection=strip_geometry(fc), description=name,
                                         folder=EXPORT_FOLDER, fileNamePrefix=name,
                                         fileFormat="CSV", selectors=SELECTORS[kind])
    task.start()
    print(f"submitted {name}")


def export_year(year, districts, mask, kinds=("era5", "chirps", "modis")):
    start, end = window(year)
    mean_count = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)   # -> NDVI_mean, NDVI_count, ...
    mean_only = ee.Reducer.mean()            # multi-band images: columns are the band names
    # single-band images would otherwise be named "mean": names are pinned with setOutputs below

    # ---- weather (exposure) : native resolution, no mask ----
    era5 = table(ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR"), lambda i: i, ERA5_BANDS,
                 "era5", districts, 11132, mean_only, start, end)
    chirps = table(ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY"), lambda i: i, ["precipitation"],
                   "chirps", districts, 5566, ee.Reducer.mean().setOutputs(["precipitation"]), start, end)

    # ---- vegetation / thermal / productivity (response) : masked, 500 m ----
    veg = table(ee.ImageCollection("MODIS/061/MOD13Q1"),
                lambda i: i.updateMask(i.select("SummaryQA").lte(1)).select(["NDVI", "EVI"])
                .multiply(0.0001).updateMask(mask),
                ["NDVI", "EVI"], "veg", districts, 500, mean_count, start, end)
    ndmi = table(ee.ImageCollection("MODIS/061/MOD09A1"),
                 lambda i: i.normalizedDifference(["sur_refl_b02", "sur_refl_b06"])
                 .rename("NDMI").updateMask(mask),
                 ["NDMI"], "ndmi", districts, 500, ee.Reducer.mean().setOutputs(["NDMI"]), start, end)
    lst = table(ee.ImageCollection("MODIS/061/MOD11A2"),
                lambda i: i.select(["LST_Day_1km", "LST_Night_1km"]).multiply(0.02)
                .subtract(273.15).updateMask(mask),
                ["LST_Day_1km", "LST_Night_1km"], "lst", districts, 1000, mean_only, start, end)
    gpp = table(ee.ImageCollection("MODIS/061/MOD17A2H"),
                lambda i: i.select("Gpp").multiply(0.1).updateMask(mask),   # g C m-2 / 8 d
                ["Gpp"], "gpp", districts, 500, ee.Reducer.mean().setOutputs(["Gpp"]), start, end)
    modis = veg.merge(ndmi).merge(lst).merge(gpp)

    if "era5" in kinds:
        export(era5, f"{PREFIX}_era5_{year}", "era5")
    if "chirps" in kinds:
        export(chirps, f"{PREFIX}_chirps_{year}", "chirps")
    if "modis" in kinds:
        export(modis, f"{PREFIX}_modis_{year}", "modis")


def export_mask_coverage(districts, mask, start_year, end_year):
    cov = (mask.unmask(0).rename("mask_frac")
           .reduceRegions(collection=districts, reducer=ee.Reducer.mean().setOutputs(["mask_frac"]),
                          scale=500, tileScale=TILE_SCALE))
    task = ee.batch.Export.table.toDrive(collection=strip_geometry(cov),
                                         description=f"{PREFIX}_mask_coverage",
                                         folder=EXPORT_FOLDER, fileNamePrefix=f"{PREFIX}_mask_coverage",
                                         fileFormat="CSV")
    task.start()
    print("submitted ts_mask_coverage")


def task_status():
    rows = [t.status() for t in ee.batch.Task.list() if str(t.status().get("description", "")).startswith(PREFIX)]
    counts = {}
    for s in sorted(rows, key=lambda s: s.get("description", "")):
        counts[s["state"]] = counts.get(s["state"], 0) + 1
        if s["state"] == "FAILED":
            print(s["description"], "FAILED:", s.get("error_message"))
    print("summary:", counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--sanity-check", action="store_true")
    ap.add_argument("--task-status", action="store_true")
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--end-year", type=int, default=END_YEAR)
    ap.add_argument("--only", nargs="+", choices=["era5", "chirps", "modis"],
                    help="re-submit only these tables (e.g. --only modis after the selectors fix)")
    a = ap.parse_args()
    if not (a.run or a.sanity_check or a.task_status):
        ap.error("choose --sanity-check, --task-status or --run")

    initialize_earth_engine()
    if a.task_status:
        task_status()
        return
    districts = get_districts().select(["PROJECT_STATE", "PROJECT_DISTRICT"])
    if a.sanity_check:
        sanity_check(get_districts())
        return

    kinds = tuple(a.only) if a.only else ("era5", "chirps", "modis")
    mask = winter_crop_mask(a.start_year, a.end_year)
    if not a.only:
        export_mask_coverage(districts, mask, a.start_year, a.end_year)
    for y in range(a.start_year, a.end_year + 1):
        export_year(y, districts, mask, kinds)
        time.sleep(TASK_DELAY_SECONDS)
    print(f"All tasks submitted -> Drive/{EXPORT_FOLDER}. Download into data/raw/gee_ts/")


if __name__ == "__main__":
    main()