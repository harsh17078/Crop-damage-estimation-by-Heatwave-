"""
CHECK THE EXPORTED ts_modis_YYYY.csv FILES  (run on ALL years before using them)

Usage:
    python check_modis_exports.py                     # checks data/raw/gee_ts/ts_modis_*.csv
    python check_modis_exports.py path/to/folder_or_file

For every file it checks
  1. header has all expected columns
  2. each source (veg, ndmi, lst, gpp) exists and has the expected number of composite dates
  3. every composite date has every district (no missing district-dates), no duplicates
  4. values are filled (valid share) and physically plausible
  5. dates lie inside Oct 1 (Y-1) .. May 16 (Y) for the year in the file name
  6. composites built from very few pixels (veg count < 20% of the district's median),
     typically fog days in December
Output: one PASS / WARN / FAIL line per check and a final table over all files.
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

DEFAULT_DIR = "data/raw/gee_ts"
ID = ["PROJECT_STATE", "PROJECT_DISTRICT", "date", "source"]
COLS = ["NDVI_mean", "NDVI_count", "EVI_mean", "EVI_count", "NDMI", "LST_Day_1km", "LST_Night_1km", "Gpp"]
SOURCES = {          # source -> (value columns, expected composites per season window, plausible range)
    "veg": (["NDVI_mean", "EVI_mean"], (13, 15), {"NDVI_mean": (-0.2, 1.0), "EVI_mean": (-0.2, 1.0)}),
    "ndmi": (["NDMI"], (27, 30), {"NDMI": (-1.0, 1.0)}),
    "lst": (["LST_Day_1km", "LST_Night_1km"], (27, 30), {"LST_Day_1km": (-5, 60), "LST_Night_1km": (-15, 40)}),
    "gpp": (["Gpp"], (27, 30), {"Gpp": (0, 120)}),
}


def check_file(path):
    year = int(re.search(r"_(\d{4})\.csv$", path).group(1))
    d = pd.read_csv(path)
    msgs, status = [], "PASS"

    def add(level, text):
        nonlocal status
        msgs.append(f"  [{level}] {text}")
        if level == "FAIL" or (level == "WARN" and status == "PASS"):
            status = level

    missing = [c for c in ID + COLS if c not in d.columns]
    if missing:
        add("FAIL", f"missing columns {missing}")
        return year, status, msgs, {}
    add("PASS", "header complete (12 expected columns)")

    d["date"] = pd.to_datetime(d["date"])
    n_dist = d["PROJECT_DISTRICT"].nunique()
    lo, hi = pd.Timestamp(year - 1, 10, 1), pd.Timestamp(year, 5, 16)
    if d["date"].min() < lo or d["date"].max() > hi:
        add("FAIL", f"dates outside {lo.date()}..{hi.date()}: {d['date'].min().date()}..{d['date'].max().date()}")
    else:
        add("PASS", f"dates inside the season window ({d['date'].min().date()}..{d['date'].max().date()})")
    add("PASS" if not d.duplicated(["PROJECT_DISTRICT", "date", "source"]).any() else "FAIL", "no duplicate district-date-source rows")

    summary = {"year": year, "districts": n_dist}
    for src, (vcols, (n_lo, n_hi), ranges) in SOURCES.items():
        g = d[d["source"] == src]
        if g.empty:
            if src == "gpp":
                add("PASS", "source 'gpp': no rows - expected (GEE MOD17A2H v061 starts 2021-01-01; GPP is optional)")
            else:
                add("FAIL" if src == "veg" else "WARN", f"source '{src}': NO ROWS (variables {vcols} unavailable for this year)")
            summary[src] = 0
            continue
        nd = g["date"].nunique()
        per_date = g.groupby("date")["PROJECT_DISTRICT"].nunique()
        ok_dates = n_lo <= nd <= n_hi
        add("PASS" if ok_dates else "WARN", f"source '{src}': {len(g)} rows, {nd} composite dates (expected {n_lo}-{n_hi})")
        if per_date.min() < n_dist:
            add("WARN", f"source '{src}': {int((per_date < n_dist).sum())} dates lack some districts (min {per_date.min()}/{n_dist})")
        for c in vcols:
            valid = g[c].notna().mean()
            lo_r, hi_r = ranges[c]
            out = ((g[c] < lo_r) | (g[c] > hi_r)).sum()
            lvl = "PASS" if valid >= 0.9 and out == 0 else ("FAIL" if valid < 0.5 else "WARN")
            add(lvl, f"{c}: valid {valid:.1%}, range {g[c].min():.3f}..{g[c].max():.3f}, outside plausible {lo_r}..{hi_r}: {int(out)}")
        summary[src] = len(g)

    v = d[d["source"] == "veg"]
    if len(v):
        med = v.groupby("PROJECT_DISTRICT")["NDVI_count"].transform("median")
        low = v[v["NDVI_count"] < 0.2 * med]
        if len(low):
            by = low["date"].dt.strftime("%Y-%m-%d").value_counts().to_dict()
            add("WARN", f"{len(low)} veg composites built from < 20% of the usual pixels (fog/cloud) on {by}; "
                        "their NDVI means are noisy - filter them before interpolation")
        else:
            add("PASS", "no veg composite built from unusually few pixels")
        summary["min_pixels"] = int(v["NDVI_count"].min())
        summary["median_pixels_min_district"] = int(v.groupby("PROJECT_DISTRICT")["NDVI_count"].median().min())
    return year, status, msgs, summary


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    files = [target] if os.path.isfile(target) else sorted(glob.glob(os.path.join(target, "ts_modis_*.csv")))
    if not files:
        sys.exit(f"No ts_modis_*.csv found in {target}")
    rows = []
    for f in files:
        year, status, msgs, summ = check_file(f)
        print(f"\n=== {os.path.basename(f)}  ->  {status}")
        print("\n".join(msgs))
        rows.append({"year": year, "status": status, **summ})
    t = pd.DataFrame(rows)
    print("\n" + "=" * 70 + "\nSUMMARY OVER ALL FILES\n" + "=" * 70)
    print(t.to_string(index=False))
    miss = sorted(set(range(2003, 2025)) - set(t["year"]))
    if len(files) > 1 and miss:
        print(f"\n[WARNING] years with no file: {miss}")


if __name__ == "__main__":
    main()