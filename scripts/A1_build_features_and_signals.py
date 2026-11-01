"""
OPTION A — STEP A1: PHENOLOGY-ANCHORED FEATURES + CANDIDATE LABEL SIGNALS
=========================================================================

Reads the raw GEE series (A0) and builds ONE row per State x District x harvest Year.

Quality filter: fog / cloud-contaminated composites are removed first (see quality_filter()).

Phenology anchor  P  (per district)
-----------------------------------
P = median day-of-year of the smoothed-NDVI peak (Jan 1 - Apr 15) over a FIXED early
reference period (PHENO_REF_YEARS, default 2003-2010). Using early years only means no
validation/test year influences the windows. Everything is expressed in days relative to P,
so Punjab (P ~ early Mar) and MP/Rajasthan (P ~ mid/late Feb) are treated consistently.

Windows (days relative to P)
----------------------------
  RESPONSE variables (NDVI, EVI, NDMI, LST) - only BEFORE the peak, never inside the label window
      v1 = P-75 .. P-45     v2 = P-45 .. P-15     v3 = P-15 .. P-1
  LABEL window (response after the peak)           post = P .. P+POST_LEN (default 40 d)
  EXPOSURE variables (weather) - may extend into/over the label window (different sensor)
      pre  = P-60 .. P-15   anth = P-15 .. P+10   fill = P+10 .. P+POST_LEN

Candidate label signals (higher = healthier; damage = negative z, see proxy_label_lib)
      S_ndvi   mean NDVI in post window
      S_dndvi  S_ndvi - veg_NDVI_v3  (decline relative to the pre-peak state)
      S_evi    mean EVI in post window
      S_gpp    integrated GPP in post window (g C m-2)
A2 compares them; A3 trains on the chosen one.

Heat departure uses a Tmax climatology from the SAME reference years (no later data).

Input : data/raw/gee_ts/ts_{era5,chirps,modis}_YYYY.csv  [+ ts_mask_coverage.csv]
Output: data/processed/A_features_signals.csv
        data/processed/A_phenology_reference.csv
Run   : python scripts/A1_build_features_and_signals.py
"""

import glob
import os
import re

import numpy as np
import pandas as pd

RAW_DIR = "data/raw/gee_ts"
OUT_PATH = "data/processed/A_features_signals.csv"
PHENO_OUT = "data/processed/A_phenology_reference.csv"

PHENO_REF_YEARS = list(range(2003, 2011))
PEAK_SEARCH = (1, 1, 4, 15)               # Jan 1 .. Apr 15
POST_LEN = 40
P_MIN, P_MAX = 40, 90                     # plausible wheat peak: Feb 9 .. Mar 31 (day of year)

# --- quality filters (fog / cloud) ---
LOW_COUNT_FRAC = 0.20      # a veg composite built from < 20% of the district's usual pixels is dropped
FOG_WINDOW_DAYS = 15       # ... and 8-day NDMI / LST composites starting inside that 16-day window too
LST_DAY_MIN, LST_DAY_MAX = -2.0, 60.0   # land daytime LST outside this = cloud-top reading
LST_DAY_NIGHT_TOL = 3.0    # daytime LST more than 3 C BELOW night LST = contaminated
MAX_GAP_DAYS = 40          # gap-fill limit (two missing 16-day composites in a row can still be bridged)
MIN_COVERAGE = 0.5                        # share of valid days required inside a window

V_WINDOWS = {"v1": (-75, -45), "v2": (-45, -15), "v3": (-15, -1)}
POST = (0, POST_LEN)
W_WINDOWS = {"pre": (-60, -15), "anth": (-15, 10), "fill": (10, POST_LEN)}

# (output name, candidate GEE column names, composite-centre shift in days, per-day divisor)
MODIS_VARS = {
    "veg": [("NDVI", ["NDVI_mean"], 8, 1), ("EVI", ["EVI_mean"], 8, 1)],
    "ndmi": [("NDMI", ["NDMI", "mean"], 4, 1)],
    "lst": [("LST_day", ["LST_Day_1km"], 4, 1), ("LST_night", ["LST_Night_1km"], 4, 1)],
    "gpp": [("GPP", ["Gpp", "mean"], 4, 8)],     # g C m-2 per 8 days -> per day
}
RESPONSE_FEATURE_VARS = ["NDVI", "EVI", "NDMI", "LST_day", "LST_night"]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def read_ts(kind):
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW_DIR, f"ts_{kind}_*.csv"))):
        m = re.search(r"_(\d{4})\.csv$", f)
        if not m:
            continue
        d = pd.read_csv(f)
        d["HYear"] = int(m.group(1))
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"No ts_{kind}_YYYY.csv files in {RAW_DIR}")
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    return d.rename(columns={"PROJECT_STATE": "State", "PROJECT_DISTRICT": "District"})


def jan1(y):
    return pd.Timestamp(int(y), 1, 1)


def date_of(y, p):
    return jan1(y) + pd.Timedelta(days=int(p) - 1)


def wmean(s, y, P, a, b):
    w = s.loc[date_of(y, P + a):date_of(y, P + b)]
    if len(w) == 0 or w.notna().mean() < MIN_COVERAGE:
        return np.nan
    return float(w.mean())


def max_run(mask):
    m = np.asarray(mask, dtype=int)
    if m.sum() == 0:
        return 0
    d = np.diff(np.r_[0, m, 0])
    return int((np.where(d == -1)[0] - np.where(d == 1)[0]).max())


def daily_interp(dates, values, shift, y):
    s = pd.Series(values.to_numpy(float), index=pd.to_datetime(dates) + pd.Timedelta(days=shift))
    s = s[~s.index.duplicated()].sort_index().dropna()
    full = pd.date_range(pd.Timestamp(y - 1, 10, 1), pd.Timestamp(y, 5, 16), freq="D")
    if len(s) < 2:
        return pd.Series(np.nan, index=full)
    s = s.reindex(s.index.union(full)).interpolate("time", limit=MAX_GAP_DAYS, limit_area="inside")
    return s.reindex(full)


# ----------------------------------------------------------------------
# quality filter: fog / cloud-contaminated composites
# ----------------------------------------------------------------------
def quality_filter(modis):
    """Remove composites that the pixel counts / physics show to be fog- or cloud-contaminated.

    1. veg  : NDVI/EVI set to NaN where NDVI_count < LOW_COUNT_FRAC x the district's median count
              (the wheat mask is fixed, so a low count means most pixels were screened out by cloud).
    2. ndmi / lst : composites starting inside a flagged veg composite's 16-day window are set to
              NaN as well (same fog episode; these products have no pixel count of their own).
    3. LST  : daytime LST outside [LST_DAY_MIN, LST_DAY_MAX], or colder than night LST by more than
              LST_DAY_NIGHT_TOL, is cloud-top temperature -> NaN.
    Everything removed is later bridged by interpolation (up to MAX_GAP_DAYS) or left NaN.
    """
    m = modis.copy()
    report = {}
    if "NDVI_count" not in m.columns:
        print("[WARNING] NDVI_count column missing - fog filter skipped.")
        return m
    veg = m["source"] == "veg"
    med = m[veg].groupby("District")["NDVI_count"].median()
    thr = m["District"].map(med) * LOW_COUNT_FRAC
    low = veg & (m["NDVI_count"].fillna(0) <= 0) | (veg & (m["NDVI_count"] < thr))
    report["veg composites removed"] = int(low.sum())
    for c in ("NDVI_mean", "EVI_mean"):
        if c in m.columns:
            m.loc[low, c] = np.nan

    fog = m.loc[low, ["District", "HYear", "date"]]
    flagged = pd.Series(False, index=m.index)
    other = m["source"].isin(["ndmi", "lst"])
    for (dist, hy), g in fog.groupby(["District", "HYear"]):
        sel = other & (m["District"] == dist) & (m["HYear"] == hy)
        if not sel.any():
            continue
        dts = m.loc[sel, "date"]
        hit = pd.Series(False, index=dts.index)
        for d0 in g["date"]:
            hit |= (dts >= d0) & (dts <= d0 + pd.Timedelta(days=FOG_WINDOW_DAYS))
        flagged.loc[hit[hit].index] = True
    for c in ("NDMI", "LST_Day_1km", "LST_Night_1km"):
        if c in m.columns:
            m.loc[flagged, c] = np.nan
    report["ndmi/lst composites removed (same fog window)"] = int(flagged.sum())

    if {"LST_Day_1km", "LST_Night_1km"} <= set(m.columns):
        d, n = m["LST_Day_1km"], m["LST_Night_1km"]
        bad = ((d < LST_DAY_MIN) | (d > LST_DAY_MAX) | (d < n - LST_DAY_NIGHT_TOL)) & (m["source"] == "lst")
        report["LST_day values removed (physically implausible)"] = int(bad.sum())
        m.loc[bad, "LST_Day_1km"] = np.nan

    section("QUALITY FILTER (fog / cloud)")
    for k, v in report.items():
        print(f"  {k}: {v}")
    if low.any():
        by = m.loc[low, "date"].dt.strftime("%m-%d").value_counts().head(6).to_dict()
        print(f"  most affected composite dates (month-day): {by}")
    return m


# ----------------------------------------------------------------------
# build series
# ----------------------------------------------------------------------
def build_modis_series(modis):
    series = {}
    for src, specs in MODIS_VARS.items():
        sub = modis[modis["source"] == src]
        for (st, dist, y), g in sub.groupby(["State", "District", "HYear"]):
            for name, cols, shift, div in specs:
                col = pick(g, cols)
                if col is None or g[col].notna().sum() < 2:
                    continue
                s = daily_interp(g["date"], g[col] / div, shift, y)
                series[(dist, y, name)] = s
    return series


def build_weather(era5, chirps):
    p = chirps[["District", "HYear", "date"]].copy()
    p["rain"] = chirps[pick(chirps, ["precipitation", "mean"])]
    w = era5.merge(p, on=["District", "HYear", "date"], how="left")
    w["tmax"] = w["temperature_2m_max"] - 273.15
    w["tmin"] = w["temperature_2m_min"] - 273.15
    w["dew"] = w["dewpoint_temperature_2m"] - 273.15
    w["solar"] = w["surface_solar_radiation_downwards_sum"] / 1e6
    w["sm"] = w["volumetric_soil_water_layer_1"]
    w["dos"] = (w["date"] - w["HYear"].map(lambda y: pd.Timestamp(y - 1, 10, 1))).dt.days
    return w[["State", "District", "HYear", "date", "dos", "tmax", "tmin", "dew", "solar", "sm", "rain"]]


def add_departure(w):
    ref = w[w["HYear"].isin(PHENO_REF_YEARS)]
    clim = ref.groupby(["District", "dos"])["tmax"].mean()
    clim = clim.groupby(level=0).transform(lambda s: s.rolling(15, center=True, min_periods=1).mean())
    key = pd.MultiIndex.from_frame(w[["District", "dos"]])
    w["dep"] = w["tmax"].to_numpy() - clim.reindex(key).to_numpy()
    return w


def phenology(series):
    rows = []
    for (dist, y, name), s in series.items():
        if name != "NDVI" or y not in PHENO_REF_YEARS:
            continue
        ss = s.loc[pd.Timestamp(y, PEAK_SEARCH[0], PEAK_SEARCH[1]):pd.Timestamp(y, PEAK_SEARCH[2], PEAK_SEARCH[3])]
        ss = ss.rolling(15, center=True, min_periods=5).mean()
        if ss.notna().sum() < 30:
            continue
        pk = ss.idxmax()
        edge = pk <= ss.dropna().index[3] or pk >= ss.dropna().index[-4]
        rows.append({"District": dist, "Year": y, "peak_doy": pk.dayofyear, "edge": bool(edge)})
    ph = pd.DataFrame(rows)
    if ph.empty:
        raise ValueError("No NDVI peaks found in the reference years - check data/raw/gee_ts.")
    ref = ph.groupby("District").agg(P_doy=("peak_doy", "median"), n_ref=("Year", "size"),
                                     edge_share=("edge", "mean"), spread=("peak_doy", "std")).reset_index()
    ref["P_doy"] = ref["P_doy"].round().astype(int)
    return ref


def wx_feats(w, dep, name):
    f = {}
    ok = w["tmax"].notna().mean() >= MIN_COVERAGE if len(w) else False
    keys = ["tmax_mean", "tmax_max", "tmin_mean", "hot35_days", "hdd35", "run35", "anomhot_days",
            "vpd_mean", "solar_sum", "rain_sum", "rain_max", "dry_max", "sm_mean"] + (["frost_days"] if name == "pre" else [])
    if not ok:
        return {f"wx_{name}_{k}": np.nan for k in keys}
    t, tn = w["tmax"], w["tmin"]
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    ea = 0.6108 * np.exp(17.27 * w["dew"] / (w["dew"] + 237.3))
    f["tmax_mean"], f["tmax_max"], f["tmin_mean"] = t.mean(), t.max(), tn.mean()
    f["hot35_days"] = int((t >= 35).sum())
    f["hdd35"] = float((t - 35).clip(lower=0).sum())
    f["run35"] = max_run((t >= 35).to_numpy())
    f["anomhot_days"] = int(((dep >= 4) & (t >= 32)).sum())
    f["vpd_mean"] = float((es - ea).clip(lower=0).mean())
    f["solar_sum"] = float(w["solar"].sum())
    f["rain_sum"] = float(w["rain"].sum())
    f["rain_max"] = float(w["rain"].max())
    f["dry_max"] = max_run((w["rain"] < 1).to_numpy())
    f["sm_mean"] = float(w["sm"].mean())
    if name == "pre":
        f["frost_days"] = int((tn <= 2).sum())
    return {f"wx_{name}_{k}": v for k, v in f.items()}


# ----------------------------------------------------------------------
def main():
    section("A1 — READING RAW TIME SERIES")
    era5, chirps, modis = read_ts("era5"), read_ts("chirps"), read_ts("modis")
    print(f"era5 {len(era5)} rows | chirps {len(chirps)} | modis {len(modis)}")
    state_of = era5.drop_duplicates("District").set_index("District")["State"]
    modis = quality_filter(modis)

    section("MODIS COLUMN CHECK")
    expect = {"veg": ["NDVI_mean", "EVI_mean"], "ndmi": ["NDMI", "mean"],
              "lst": ["LST_Day_1km"], "gpp": ["Gpp", "mean"]}
    for src, cols in expect.items():
        sub = modis[modis["source"] == src]
        col = pick(sub, cols)
        valid = float(sub[col].notna().mean()) if (col and len(sub)) else 0.0
        print(f"  {src:5s} rows {len(sub):7d} | column {col} | valid {valid:6.1%}")
        if src == "veg" and (col is None or valid < 0.5):
            raise ValueError("NDVI/EVI columns missing or mostly empty - re-export with A0.")
        if src == "gpp" and len(sub) == 0:
            print("  [INFO] no GPP rows: GEE's MOD17A2H v061 starts in 2021, so S_gpp is not used (expected).")
        if len(sub) and col is None:
            print(f"  [WARNING] '{src}' rows exist but their value column is absent from the CSV header. "
                  "Earth Engine builds the header from the first feature only; re-export the ts_modis_* "
                  "tasks with the fixed A0 (explicit selectors). Features/signals from this source stay empty.")

    series = build_modis_series(modis)
    wx = add_departure(build_weather(era5, chirps))

    section("PHENOLOGY REFERENCE (peak NDVI date per district)")
    ref = phenology(series)
    ref["State"] = ref["District"].map(state_of)
    ref["P_raw"] = ref["P_doy"]
    ok = ref["P_doy"].between(P_MIN, P_MAX)
    allP = int(ref.loc[ok, "P_doy"].median())
    stateP = ref[ok].groupby("State")["P_doy"].median().round().astype(int)
    ref["P_fallback"] = ~ok
    if (~ok).any():
        print(f"\n[WARNING] {int((~ok).sum())} district(s) have an NDVI peak outside day {P_MIN}-{P_MAX} "
              "(not a wheat-like peak: mustard/gram/orchard/forest in the mask, or a missing green-up). "
              "They are flagged P_fallback=1 (anchor = state median) and EXCLUDED from A2/A3 by default "
              "(A3: --keep-fallback to include them):")
        print(ref.loc[~ok, ["State", "District", "P_raw", "n_ref", "edge_share", "spread"]].to_string(index=False))
        ref.loc[~ok, "P_doy"] = ref.loc[~ok, "State"].map(stateP).fillna(allP).astype(int)
    border = ref[ok & (ref["P_raw"] < P_MIN + 6)]
    if len(border):
        print(f"\n[INFO] {len(border)} district(s) pass the guard but peak early (day {P_MIN}-{P_MIN + 5}); "
              "they may be partly mustard/gram-contaminated - check against local crop statistics:")
        print(border[["State", "District", "P_raw", "spread"]].sort_values("P_raw").to_string(index=False))
    print(ref.groupby("State")["P_doy"].agg(["min", "median", "max", "count"]).to_string())
    bad = ref[(ref["edge_share"] >= 0.5) | (ref["n_ref"] < 3)]
    if len(bad):
        print("\n[WARNING] unreliable peak (edge of search window or <3 reference years):")
        print(bad[["State", "District", "P_doy", "n_ref", "edge_share"]].to_string(index=False))
    ref.to_csv(PHENO_OUT, index=False)
    Pmap = ref.set_index("District")["P_doy"].to_dict()
    FBmap = ref.set_index("District")["P_fallback"].astype(int).to_dict()   # 1 = peak not wheat-like

    section("BUILDING DISTRICT-YEAR ROWS")
    wx_groups = {k: g.sort_values("date").set_index("date") for k, g in wx.groupby(["District", "HYear"])}
    rows = []
    for (dist, y), g in wx_groups.items():
        st = state_of.get(dist)
        P = Pmap.get(dist, stateP.get(st, allP))
        r = {"State": st, "District": dist, "Year": int(y), "P_doy": int(P),
             "P_fallback": int(FBmap.get(dist, 1))}   # 1 = anchor guessed -> A2/A3 exclude by default

        # response variables before the peak
        for name in RESPONSE_FEATURE_VARS:
            s = series.get((dist, y, name))
            for wn, (a, b) in V_WINDOWS.items():
                r[f"veg_{name}_{wn}"] = wmean(s, y, P, a, b) if s is not None else np.nan
        nd = series.get((dist, y, "NDVI"))
        if nd is not None:
            seg = nd.loc[date_of(y, P + V_WINDOWS["v1"][0]):date_of(y, P - 1)]
            r["veg_NDVI_premax"] = float(seg.max()) if seg.notna().mean() >= MIN_COVERAGE else np.nan
        r["veg_NDVI_rise"] = r.get("veg_NDVI_v2", np.nan) - r.get("veg_NDVI_v1", np.nan)
        r["veg_NDVI_flat"] = r.get("veg_NDVI_v3", np.nan) - r.get("veg_NDVI_v2", np.nan)
        r["veg_LST_day_gap"] = r.get("veg_LST_day_v3", np.nan) - r.get("veg_LST_day_v2", np.nan)

        # label signals (response AFTER the peak)
        a, b = POST
        s_nd, s_ev, s_gp = (series.get((dist, y, n)) for n in ("NDVI", "EVI", "GPP"))
        r["S_ndvi"] = wmean(s_nd, y, P, a, b) if s_nd is not None else np.nan
        r["S_dndvi"] = r["S_ndvi"] - r.get("veg_NDVI_v3", np.nan)
        r["S_evi"] = wmean(s_ev, y, P, a, b) if s_ev is not None else np.nan
        if s_gp is not None:
            m = wmean(s_gp, y, P, a, b)
            r["S_gpp"] = m * (b - a + 1) if m == m else np.nan
        else:
            r["S_gpp"] = np.nan
        # --- extra candidate signals (season-level vegetation response) ---
        #   S_peak : maximum NDVI around the anchor          (classic peak-greenness yield correlate)
        #   S_int  : mean NDVI P-30..P+40                     (season "integral")
        #   S_late : mean NDVI P+15..P+40                     (grain-fill / senescence only)
        #   S_p20  : mean NDVI P+17..P+23                     (single mid-grain-fill snapshot)
        if s_nd is not None:
            seg = s_nd.loc[date_of(y, P - 20):date_of(y, P + 20)]
            r["S_peak"] = float(seg.max()) if len(seg) and seg.notna().mean() >= MIN_COVERAGE else np.nan
            r["S_int"] = wmean(s_nd, y, P, -30, 40)
            r["S_late"] = wmean(s_nd, y, P, 15, 40)
            r["S_p20"] = wmean(s_nd, y, P, 17, 23)
        else:
            r["S_peak"] = r["S_int"] = r["S_late"] = r["S_p20"] = np.nan
        if s_nd is not None:
            w_ = s_nd.loc[date_of(y, P + a):date_of(y, P + b)]
            r["S_cov"] = float(w_.notna().mean()) if len(w_) else np.nan

        # exposure variables
        for wn, (a_, b_) in W_WINDOWS.items():
            sl = g.loc[date_of(y, P + a_):date_of(y, P + b_)]
            r.update(wx_feats(sl, sl["dep"] if len(sl) else sl.get("dep"), wn))
        rows.append(r)

    out = pd.DataFrame(rows).sort_values(["State", "District", "Year"]).reset_index(drop=True)

    cov_path = os.path.join(RAW_DIR, "ts_mask_coverage.csv")
    if os.path.exists(cov_path):
        cov = pd.read_csv(cov_path).rename(columns={"PROJECT_DISTRICT": "District"})
        out = out.merge(cov[["District", "mask_frac"]].drop_duplicates("District"), on="District", how="left")
        low = out.drop_duplicates("District").query("mask_frac < 0.02")
        if len(low):
            print(f"\n[WARNING] {len(low)} district(s) have <2% of their area in the winter-crop mask: "
                  f"{low['District'].tolist()}")

    empty = [c for c in out.columns if c.startswith(("veg_", "wx_", "S_")) and out[c].isna().all()]
    if empty:
        print(f"\n[WARNING] dropping {len(empty)} completely empty column(s): {empty}")
        out = out.drop(columns=empty)

    for c in [c for c in out.columns if c.startswith("S_") and out[c].notna().mean() < 0.5]:
        print(f"[INFO] dropping {c}: only {out[c].notna().mean():.0%} of district-years have a value "
              "(a signal must cover most years to give a past-only z-score).")
        out = out.drop(columns=c)

    section("QUALITY CONTROL")
    sig = [c for c in ["S_ndvi", "S_dndvi", "S_evi", "S_peak", "S_int", "S_late", "S_p20", "S_gpp"] if c in out.columns]
    print(f"rows {len(out)} | districts {out['District'].nunique()} | years {out['Year'].min()}-{out['Year'].max()}")
    print("\nmissing share of label signals:")
    print(out[sig].isna().mean().round(3).to_string())
    feats = [c for c in out.columns if c.startswith(("veg_", "wx_"))]
    miss = out[feats].isna().mean().sort_values(ascending=False)
    print(f"\n{len(feats)} feature columns; worst missing share:")
    print(miss.head(8).round(3).to_string())
    print("\nP_doy by state (days since Jan 1):")
    print(out.groupby("State")["P_doy"].describe()[["min", "50%", "max"]].to_string())

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}  ({len(out)} rows)  |  {PHENO_OUT}")
    print("Next: A2_label_threshold_analysis.py")


if __name__ == "__main__":
    main()