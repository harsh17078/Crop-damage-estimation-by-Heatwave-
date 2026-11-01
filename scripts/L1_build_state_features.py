"""
L1 - STATE-LEVEL LONG-RECORD TABLE:  ERA5 weather features  +  ICRISAT wheat yields  (1966-2017)
===============================================================================================

Builds ONE row per State x harvest Year with
   * the SAME weather feature names and definitions as A1 (functions are imported from A1, so
     wx_pre_*, wx_anth_*, wx_fill_* mean exactly what they meant before),
   * the area-weighted state wheat yield from ICRISAT (sum production / sum area; the fixed 1966
     districts make state totals immune to later district splits).

Differences from the district pipeline (be explicit about them in the report)
   * weather = ERA5-Land only (precipitation from ERA5-Land, not CHIRPS) so that ALL decades use one
     source; spatial mean over the wheat region of the state (L0);
   * windows are FIXED calendar windows around a per-state anchor P (median of the district NDVI-peak
     anchors from A1 for Punjab/Haryana/UP). For states without an NDVI-derived anchor a placeholder
     is used and printed as ASSUMED - replace it (export MODIS for those states with A0 and run A1);
   * the hot-day anomaly uses a state Tmax climatology from 1981-2010 (a frozen WMO-style normal;
     it uses no yield information).

ERA5 HOMOGENEITY CHECK (printed): mean Feb-Mar Tmax / Tmin by era and state. ERA5-Land is less
constrained before 1979 (pre-satellite era). A step change at 1979 larger than the expected warming
trend would mean the early weather cannot be trusted.

Input : data/raw/gee_state/lt_era5state_YYYY.csv   (from L0)
        data/raw/icrisat/*.csv                      (ICRISAT wheat file(s): year, state, district, area, production, yield)
        data/processed/A_phenology_reference.csv    (from A1; optional, gives P for Punjab/Haryana/UP)
Output: data/processed/L_state_features.csv   (State, District(=State), Year, P_doy, wx_*, Yield [t/ha], Area_kha)
Run   : python scripts/L1_build_state_features.py
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import A1_build_features_and_signals as a1          # reuse the exact feature definitions

RAW_DIR = "data/raw/gee_state"
ICRISAT_GLOB = "data/raw/icrisat/*.csv"
PHENO_REF = "data/processed/A_phenology_reference.csv"
OUT_PATH = "data/processed/L_state_features.csv"
CLIM_REF = (1981, 2010)

# placeholders ONLY for states with no NDVI-derived anchor (printed as ASSUMED)
ASSUMED_P = {"Madhya Pradesh": 46, "Rajasthan": 52, "Chhattisgarh": 46}
ERAS = [("1966-78", 1966, 1978), ("1979-2002", 1979, 2002), ("2003-17", 2003, 2017), ("2018+", 2018, 2100)]


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def find_col(df, *keys, exclude=()):
    for c in df.columns:
        l = c.lower()
        if all(k in l for k in keys) and not any(e in l for e in exclude):
            return c
    return None


def read_icrisat():
    files = sorted(glob.glob(ICRISAT_GLOB))
    if not files:
        raise FileNotFoundError(f"No ICRISAT file in {ICRISAT_GLOB}")
    out = []
    for f in files:
        d = pd.read_csv(f, encoding="latin-1")
        cols = {"Year": find_col(d, "year"), "State": find_col(d, "state", exclude=("code",)),
                "District": find_col(d, "dist", exclude=("code",)), "Area": find_col(d, "area"),
                "Prod": find_col(d, "prod"), "Yield": find_col(d, "yield")}
        if None in cols.values():
            raise ValueError(f"{f}: could not identify columns {cols}; columns are {list(d.columns)}")
        d = d[[cols[k] for k in cols]].copy()
        d.columns = list(cols)
        out.append(d)
        print(f"ICRISAT {os.path.basename(f)}: {len(d)} rows, states {sorted(d['State'].unique())}, "
              f"years {int(d['Year'].min())}-{int(d['Year'].max())}")
    d = pd.concat(out, ignore_index=True)
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    for c in ("Area", "Prod", "Yield"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["Year", "Area", "Prod"])


def state_yields(d):
    g = d[d["Area"] > 0].groupby(["State", "Year"]).agg(Area_kha=("Area", "sum"), Prod=("Prod", "sum"),
                                                        n_districts=("District", "nunique")).reset_index()
    g["Yield"] = g["Prod"] / g["Area_kha"]            # '000 tonnes / '000 ha = t/ha
    g["Year"] = g["Year"].astype(int)
    return g[["State", "Year", "Yield", "Area_kha", "n_districts"]]


def read_weather():
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW_DIR, "lt_era5state_*.csv"))):
        m = re.search(r"_(\d{4})\.csv$", f)
        if not m:
            continue
        w = pd.read_csv(f)
        w["HYear"] = int(m.group(1))
        frames.append(w)
    if not frames:
        raise FileNotFoundError(f"No lt_era5state_YYYY.csv in {RAW_DIR}")
    w = pd.concat(frames, ignore_index=True)
    w["date"] = pd.to_datetime(w["date"])
    w = w.rename(columns={"ADM1_NAME": "State"})
    w["State"] = w["State"].replace({"Chattisgarh": "Chhattisgarh"})
    w["tmax"] = w["temperature_2m_max"] - 273.15
    w["tmin"] = w["temperature_2m_min"] - 273.15
    w["dew"] = w["dewpoint_temperature_2m"] - 273.15
    w["solar"] = w["surface_solar_radiation_downwards_sum"] / 1e6
    w["sm"] = w["volumetric_soil_water_layer_1"]
    w["rain"] = w["total_precipitation_sum"] * 1000.0          # m -> mm
    w["dos"] = (w["date"] - w["HYear"].map(lambda y: pd.Timestamp(int(y) - 1, 10, 1))).dt.days
    return w[["State", "HYear", "date", "dos", "tmax", "tmin", "dew", "solar", "sm", "rain"]]


def anchors(states):
    P = {}
    if os.path.exists(PHENO_REF):
        ref = pd.read_csv(PHENO_REF)
        if "P_fallback" in ref.columns:
            ref = ref[ref["P_fallback"] == 0]
        for st, g in ref.groupby("State"):
            P[st] = int(round(g["P_doy"].median()))
    for st in states:
        if st not in P:
            P[st] = ASSUMED_P.get(st, 52)
            print(f"[ASSUMED] phenology anchor P for {st} = {P[st]} (no NDVI-derived value). Replace it: export MODIS for "
                  f"this state (A0) and run A1, then re-run this script.")
        else:
            print(f"anchor P for {st} = {P[st]} (median of district NDVI-peak anchors)")
    return P


def homogeneity(w):
    section("ERA5 HOMOGENEITY CHECK: mean Feb 1 - Mar 31 temperature by era (deg C)")
    m = w[(w["date"].dt.month.isin([2, 3]))].groupby(["State", "HYear"])[["tmax", "tmin"]].mean().reset_index()
    rows = []
    for st, g in m.groupby("State"):
        r = {"State": st}
        for name, lo, hi in ERAS:
            s = g[(g["HYear"] >= lo) & (g["HYear"] <= hi)]
            r[f"Tmax {name}"] = s["tmax"].mean() if len(s) else np.nan
        # step at 1979 beyond a linear trend: fit trend on all years, compare residual means 1966-78 vs 1979-90
        x, y = g["HYear"].to_numpy(float), g["tmax"].to_numpy(float)
        b, a = np.polyfit(x, y, 1)
        res = y - (a + b * x)
        pre, post = res[(x <= 1978)], res[(x >= 1979) & (x <= 1990)]
        r["trend C/decade"] = b * 10
        r["step 1979 (resid, C)"] = (post.mean() - pre.mean()) if len(pre) and len(post) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).round(2)
    print(t.to_string(index=False))
    print("\nReading: a 'step 1979' beyond about +-0.7 C (or a trend far from 0.1-0.4 C/decade) suggests the pre-1979 ERA5 "
          "record is not homogeneous -> use 1979+ as the primary period.")


def main():
    section("L1 - STATE-LEVEL LONG-RECORD TABLE")
    ic = read_icrisat()
    yl = state_yields(ic)
    print(f"state-year yields: {len(yl)} rows | states {sorted(yl['State'].unique())}")

    w = read_weather()
    print(f"weather: {len(w)} daily rows | states {sorted(w['State'].unique())} | harvest years "
          f"{int(w['HYear'].min())}-{int(w['HYear'].max())}")
    homogeneity(w)

    # frozen Tmax climatology (state x day-of-season), 1981-2010
    ref = w[(w["HYear"] >= CLIM_REF[0]) & (w["HYear"] <= CLIM_REF[1])]
    clim = ref.groupby(["State", "dos"])["tmax"].mean()
    clim = clim.groupby(level=0).transform(lambda s: s.rolling(15, center=True, min_periods=1).mean())
    key = pd.MultiIndex.from_frame(w[["State", "dos"]])
    w["dep"] = w["tmax"].to_numpy() - clim.reindex(key).to_numpy()

    section("ANCHORS AND FEATURES")
    P = anchors(sorted(w["State"].unique()))
    rows = []
    for (st, y), g in w.groupby(["State", "HYear"]):
        g = g.sort_values("date").set_index("date")
        r = {"State": st, "District": st, "Year": int(y), "P_doy": int(P[st])}
        for wn, (a_, b_) in a1.W_WINDOWS.items():
            sl = g.loc[a1.date_of(y, P[st] + a_):a1.date_of(y, P[st] + b_)]
            r.update(a1.wx_feats(sl, sl["dep"] if len(sl) else sl.get("dep"), wn))
        rows.append(r)
    feat = pd.DataFrame(rows)
    out = feat.merge(yl, on=["State", "Year"], how="left")
    out["P_assumed"] = out["State"].isin(ASSUMED_P).astype(int) & (~out["State"].isin(
        pd.read_csv(PHENO_REF)["State"].unique() if os.path.exists(PHENO_REF) else []))
    out = out.sort_values(["State", "Year"]).reset_index(drop=True)

    cnt = out.dropna(subset=["Yield"]).groupby("State")["Year"].agg(["min", "max", "count"])
    print("\nrows with yield (State: first year, last year, n):")
    print(cnt.to_string())
    miss = out[[c for c in out.columns if c.startswith("wx_")]].isna().mean()
    print(f"\nfeature columns {int(miss.size)} | worst missing share {miss.max():.3f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH} ({len(out)} rows). Next: L2 tests on the independent years.")


if __name__ == "__main__":
    main()