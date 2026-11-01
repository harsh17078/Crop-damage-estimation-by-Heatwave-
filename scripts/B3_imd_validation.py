"""
B3 - INDEPENDENT VALIDATION OF THE HEAT-EXPOSURE LABEL AND THE SATELLITE DETECTOR AGAINST IMD
==============================================================================================

WHY: every B1 label (HS2/HS3/HS5) comes from ONE source, ERA5-Land. B1 therefore only shows that MODIS LST agrees
     with a reanalysis. IMD gridded daily Tmax is built from station observations (India Meteorological Dept.), so it
     is an independent yardstick.

WHAT IT DOES (nothing is trained or tuned; B1 stays frozen):
  1. Takes your district-year table (reports/exposure/exposure_table.csv) and each district's centroid.
  2. Extracts IMD daily Tmax at the centroid (bilinear, nearest cell as fallback) for the SAME window P..P+40.
  3. Daily check : ERA5-Land minus IMD Tmax (bias, RMSE, correlation), pooled and by state.
  4. Label check : recomputes HS2/HS3/HS5 from IMD and compares with the ERA5 label (accuracy, kappa, prevalence,
                   by year, state-year correlation, IMD-threshold sensitivity).
  5. Satellite   : AUC of raw LST p90 (no fitting) and of the frozen B1 model's saved test scores against the IMD
                   label vs against the ERA5 label, with year-block bootstrap CIs.
  6. Verdict against criteria written below BEFORE running (edit now, then do not change).

CAVEAT (state it in the report): the IMD Tmax archive is a 1 degree grid (~110 km). It cannot resolve districts, so
  district-level disagreement is partly resolution. State-year agreement is the fairer test and is reported too.

NEEDS
  pip install imdlib xarray scipy            (imdlib downloads IMD's binary grids; needs internet)
  data/external/district_centroids.csv       columns: District, lat, lon   (see GEE snippet below)
  reports/exposure/exposure_table.csv        (written by B1)
  data/raw/gee_ts/ts_era5_YYYY.csv           (A0)

GEE snippet to export centroids (names must match your project's District names; unmatched ones are listed):
  var d = ee.FeatureCollection('FAO/GAUL/2015/level2')
            .filter(ee.Filter.eq('ADM0_NAME', 'India'))
            .filter(ee.Filter.inList('ADM1_NAME', ['Punjab','Haryana','Uttar Pradesh']));
  Export.table.toDrive(d.map(function(f){ var c = f.geometry().centroid(1);
    return ee.Feature(null, {District: f.get('ADM2_NAME'), State: f.get('ADM1_NAME'),
      lat: c.coordinates().get(1), lon: c.coordinates().get(0)}); }), 'district_centroids', 'csv');

RUN
  python scripts/B3_imd_validation.py --centroids data/external/district_centroids.csv
  python scripts/B3_imd_validation.py --centroids ... --imd-nc path/to/tmax.nc   (if you already have IMD as NetCDF)
  python scripts/B3_imd_validation.py --selftest                                   (synthetic data, no downloads)
Output: reports/exposure/B3_imd/*.csv and run_B3.log
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, cohen_kappa_score,
                             roc_auc_score)

warnings.filterwarnings("ignore")

# ---------------------------- pre-registered criteria (edit BEFORE running) ----------------------------
KAPPA_MIN = 0.60            # (a) ERA5 HS3 label vs IMD HS3 label: Cohen's kappa at least this
SAT_AUC_MIN = 0.85          # (b) raw LST p90 vs IMD HS3 label: pooled AUC at least this
SAT_AUC_DROP_MAX = 0.05     # (c) AUC against IMD may be at most this much below AUC against ERA5
# -------------------------------------------------------------------------------------------------------
TABLE = "reports/exposure/exposure_table.csv"
ERA5_GLOB = "data/raw/gee_ts/ts_era5_*.csv"
IMD_DIR = "data/external/imd"
CENTROIDS = "data/external/district_centroids.csv"
OUT = "reports/exposure/B3_imd"
WINDOW = (0, 40)
HOT_C = 35.0
MIN_VALID_FRAC = 0.80       # share of the 41 window days that must have valid IMD data
MIN_CLASS = 3
SENS_T = (33, 34, 35, 36, 37)
SEED = 42


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for st in self.s:
            st.write(x)

    def flush(self):
        for st in self.s:
            st.flush()


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def date_of(y, p):
    return pd.Timestamp(int(y), 1, 1) + pd.Timedelta(days=int(p) - 1)


# ------------------------------------------------------------------ inputs
def load_centroids(path, districts):
    c = pd.read_csv(path)
    low = {k.lower(): k for k in c.columns}

    def col(*names):
        for n in names:
            if n in low:
                return low[n]
        return None
    dc, la, lo = col("district", "project_district", "adm2_name", "name"), col("lat", "latitude", "y"), \
        col("lon", "lng", "longitude", "x")
    if None in (dc, la, lo):
        raise SystemExit(f"[ERROR] {path} needs columns District, lat, lon (found {list(c.columns)})")
    c = c.rename(columns={dc: "District", la: "lat", lo: "lon"})[["District", "lat", "lon"]]
    key = lambda s: s.astype(str).str.strip().str.lower()
    c["k"] = key(c["District"])
    t = pd.DataFrame({"District": sorted(districts)})
    t["k"] = key(t["District"])
    m = t.merge(c.drop_duplicates("k")[["k", "lat", "lon"]], on="k", how="left")
    miss = m.loc[m["lat"].isna(), "District"].tolist()
    if miss:
        print(f"[WARNING] {len(miss)} district(s) have no centroid and are skipped (fix the names in the CSV): {miss}")
    return m.dropna(subset=["lat"])[["District", "lat", "lon"]].reset_index(drop=True)


def load_era5_daily(pattern):
    out = {}
    frames = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f, usecols=lambda c: c in ("PROJECT_DISTRICT", "District", "date", "temperature_2m_max"))
        d = d.rename(columns={"PROJECT_DISTRICT": "District"})
        frames.append(d)
    if not frames:
        print(f"[WARNING] no ERA5 daily files match {pattern}: the daily bias check is skipped")
        return out
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["tmax"] = d["temperature_2m_max"] - 273.15
    for dist, g in d.groupby("District"):
        s = g.drop_duplicates("date").set_index("date")["tmax"].sort_index()
        out[dist] = s
    return out


def load_imd(years, imd_dir, nc):
    import xarray as xr
    if nc:
        ds = xr.open_dataset(nc)
        name = "tmax" if "tmax" in ds else list(ds.data_vars)[0]
        da = ds[name]
    else:
        try:
            import imdlib as imd
        except ImportError:
            raise SystemExit("[ERROR] imdlib is not installed. Run: pip install imdlib   "
                             "(or pass an existing NetCDF with --imd-nc)")
        os.makedirs(os.path.join(imd_dir, "tmax"), exist_ok=True)
        parts = []
        for y in years:
            have = glob.glob(os.path.join(imd_dir, "tmax", f"*{y}*"))
            try:
                if not have:
                    imd.get_data("tmax", y, y, fn_format="yearwise", file_dir=imd_dir)
                parts.append(imd.open_data("tmax", y, y, "yearwise", imd_dir).get_xarray()["tmax"])
            except Exception as e:
                print(f"  [WARNING] IMD Tmax {y} unavailable ({type(e).__name__}: {str(e)[:80]}) - year skipped")
        if not parts:
            raise SystemExit("[ERROR] no IMD year could be loaded. Check internet access or use --imd-nc.")
        da = xr.concat(parts, dim="time")
    ren = {k: v for k, v in {"latitude": "lat", "longitude": "lon", "LATITUDE": "lat", "LONGITUDE": "lon"}.items()
           if k in da.coords}
    da = da.rename(ren).sortby("lat").sortby("lon")
    return da.where((da > -50) & (da < 70))      # IMD missing values (-999 / 99.9) -> NaN


def imd_district_series(da, cent):
    import xarray as xr
    lat = xr.DataArray(cent["lat"].to_numpy(), dims="d")
    lon = xr.DataArray(cent["lon"].to_numpy(), dims="d")
    pt = da.interp(lat=lat, lon=lon, method="linear")
    nn = da.sel(lat=lat, lon=lon, method="nearest")
    pt = pt.fillna(nn)
    df = pd.DataFrame(pt.transpose("time", "d").values, index=pd.to_datetime(pt["time"].values).normalize(),
                      columns=cent["District"].to_numpy())
    return df[~df.index.duplicated()].sort_index()


# ------------------------------------------------------------------ computations
def window_metrics(tab, imd_df, era5):
    rows, pairs = [], []
    need = MIN_VALID_FRAC * (WINDOW[1] - WINDOW[0] + 1)
    for r in tab.itertuples(index=False):
        a, b = date_of(r.Year, r.P_doy + WINDOW[0]), date_of(r.Year, r.P_doy + WINDOW[1])
        out = {"District": r.District, "Year": r.Year}
        if r.District in imd_df.columns:
            i = imd_df[r.District].loc[a:b]
            out["imd_valid"] = int(i.notna().sum())
            if out["imd_valid"] >= need:
                out["n35_imd"] = int((i >= HOT_C).sum())
                out["tmax_mean_imd"] = float(i.mean())
                for t in SENS_T:
                    out[f"nge{t}_imd"] = int((i >= t).sum())
                e = era5.get(r.District)
                if e is not None:
                    j = pd.concat([e.loc[a:b].rename("era5"), i.rename("imd")], axis=1, join="inner").dropna()
                    if len(j):
                        out["bias_era5_minus_imd"] = float((j["era5"] - j["imd"]).mean())
                        pairs.append(j.assign(District=r.District, Year=r.Year))
        rows.append(out)
    return pd.DataFrame(rows), (pd.concat(pairs) if pairs else pd.DataFrame())


def safe_auc(y, s):
    y = np.asarray(y).astype(int)
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan


def within_year_auc(d, ycol, scol):
    v = []
    for _, g in d.groupby("Year"):
        n1 = int(g[ycol].sum())
        if n1 >= MIN_CLASS and len(g) - n1 >= MIN_CLASS:
            v.append(safe_auc(g[ycol], g[scol]))
    return (np.mean(v) if v else np.nan), len(v)


def year_bootstrap(d, ycols, scol, n_boot):
    rng = np.random.default_rng(SEED)
    years = sorted(d["Year"].unique())
    by = {y: g for y, g in d.groupby("Year")}
    res = {k: [] for k in ycols}
    res["diff"] = []
    for _ in range(n_boot):
        x = pd.concat([by[y] for y in rng.choice(years, len(years), replace=True)])
        aucs = {k: safe_auc(x[k], x[scol]) for k in ycols}
        if any(np.isnan(v) for v in aucs.values()):
            continue
        for k in ycols:
            res[k].append(aucs[k])
        res["diff"].append(aucs[ycols[1]] - aucs[ycols[0]])
    ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5)) if len(a) else (np.nan, np.nan)
    return {k: ci(v) for k, v in res.items()}


def agreement_table(tab):
    rows = []
    d = tab.dropna(subset=["n35_imd"])
    for K in (2, 3, 5):
        ye, yi = (d["n35"] >= K).astype(int), (d["n35_imd"] >= K).astype(int)
        both = int(((ye == 1) & (yi == 1)).sum())
        rows.append({"label": f"HS{K}", "n": len(d), "prev_era5": ye.mean(), "prev_imd": yi.mean(),
                     "accuracy": (ye == yi).mean(), "kappa": cohen_kappa_score(ye, yi),
                     "era5_only": int(((ye == 1) & (yi == 0)).sum()),
                     "imd_only": int(((ye == 0) & (yi == 1)).sum()), "both": both,
                     "spearman_n35": spearmanr(d["n35"], d["n35_imd"])[0]})
    return pd.DataFrame(rows)


def main_run(a, imd_da=None):
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run_B3.log"), "w")
    old = sys.stdout
    sys.stdout = Tee(old, logf)
    try:
        return _run(a, imd_da)
    finally:
        sys.stdout = old
        logf.close()


def _run(a, imd_da):
    section("B3 - IMD VALIDATION")
    tab = pd.read_csv(a.table)
    need = {"State", "District", "Year", "P_doy", "n35", "sat_lst_day_p90"}
    if need - set(tab.columns):
        raise SystemExit(f"[ERROR] {a.table} lacks {sorted(need - set(tab.columns))}: rerun B1")
    print(f"table: {len(tab)} district-years, {tab['District'].nunique()} districts, years {tab['Year'].min()}-{tab['Year'].max()}")
    cent = load_centroids(a.centroids, tab["District"].unique())
    tab = tab[tab["District"].isin(cent["District"])].reset_index(drop=True)
    print(f"districts with centroids: {len(cent)}")
    era5 = load_era5_daily(a.era5_glob)
    years = sorted(tab["Year"].unique())
    da = imd_da if imd_da is not None else load_imd(years, a.imd_dir, a.imd_nc)
    got = sorted(set(pd.to_datetime(da["time"].values).year))
    print(f"IMD Tmax years loaded: {got[0]}..{got[-1]} ({len(got)} years); grid {da.sizes.get('lat')} x {da.sizes.get('lon')}")
    imd_df = imd_district_series(da, cent)
    wm, pairs = window_metrics(tab, imd_df, era5)
    tab = tab.merge(wm, on=["District", "Year"], how="left")
    ok = tab["n35_imd"].notna()
    print(f"district-years with enough IMD data: {int(ok.sum())} of {len(tab)}")
    if ok.sum() < 100:
        raise SystemExit("[ERROR] too few district-years with IMD data; check years / centroid coordinates.")
    d = tab[ok].copy()
    d.to_csv(os.path.join(a.out, "district_year_imd.csv"), index=False)

    # ---- 1. daily check
    section("1. DAILY Tmax: ERA5-Land minus IMD (days inside the P..P+40 windows)")
    if len(pairs):
        pairs = pairs.merge(tab[["District", "State"]].drop_duplicates(), on="District", how="left")
        diff = pairs["era5"] - pairs["imd"]
        hot = pairs["imd"] >= 33
        print(f"pooled: n={len(pairs)} days | mean bias {diff.mean():+.2f} C | RMSE {np.sqrt((diff ** 2).mean()):.2f} C | "
              f"r={pearsonr(pairs['era5'], pairs['imd'])[0]:.3f}")
        print(f"hot days only (IMD >= 33 C): n={int(hot.sum())} | mean bias {diff[hot].mean():+.2f} C")
        st = pairs.assign(diff=diff).groupby("State").agg(n=("diff", "size"), bias=("diff", "mean"),
                                                          rmse=("diff", lambda x: np.sqrt((x ** 2).mean())))
        print(st.round(2).to_string())
        st.to_csv(os.path.join(a.out, "daily_bias_by_state.csv"))
        print("(negative bias = ERA5-Land cooler than IMD; a 1-degree IMD cell smooths local extremes)")
    else:
        print("skipped (no ERA5 daily series)")

    # ---- 2. label agreement
    section("2. LABEL AGREEMENT: HSk from ERA5-Land vs HSk from IMD")
    ag = agreement_table(d)
    ag.to_csv(os.path.join(a.out, "label_agreement.csv"), index=False)
    print(ag.round(3).to_string(index=False))
    yr = d.groupby("Year").apply(lambda g: pd.Series({
        "n": len(g), "HS3_era5": (g["n35"] >= 3).mean(), "HS3_imd": (g["n35_imd"] >= 3).mean(),
        "agree": ((g["n35"] >= 3) == (g["n35_imd"] >= 3)).mean(),
        "mean_n35_era5": g["n35"].mean(), "mean_n35_imd": g["n35_imd"].mean()}))
    yr.to_csv(os.path.join(a.out, "label_agreement_by_year.csv"))
    print("\nBy year (HS3):")
    print(yr.round(2).to_string())
    sy = d.groupby(["State", "Year"])[["n35", "n35_imd"]].mean()
    rho_sy = spearmanr(sy["n35"], sy["n35_imd"])[0]
    yl = d.groupby("Year")[["n35", "n35_imd"]].mean()
    print(f"\nstate-year Spearman of mean hot-day count (ERA5 vs IMD): {rho_sy:.3f} (n={len(sy)}) | "
          f"year-level: {spearmanr(yl['n35'], yl['n35_imd'])[0]:.3f} (n={len(yl)})")
    print("\nSensitivity (descriptive only): which IMD threshold best reproduces the ERA5 HS3 label?")
    sens = []
    ye = (d["n35"] >= 3).astype(int)
    for t in SENS_T:
        yi = (d[f"nge{t}_imd"] >= 3).astype(int)
        sens.append({"imd_threshold_C": t, "prev_imd": yi.mean(), "kappa_vs_era5_35C": cohen_kappa_score(ye, yi)})
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(a.out, "imd_threshold_sensitivity.csv"), index=False)
    print(sens.round(3).to_string(index=False))

    # ---- 3. satellite vs both labels
    section("3. SATELLITE DETECTION vs ERA5 label and vs IMD label (raw LST p90, nothing fitted)")
    rows = []
    boots = {}
    for K in (2, 3, 5):
        x = d.dropna(subset=["sat_lst_day_p90"]).copy()
        x["y_era5"], x["y_imd"] = (x["n35"] >= K).astype(int), (x["n35_imd"] >= K).astype(int)
        wy_e, nwy = within_year_auc(x, "y_era5", "sat_lst_day_p90")
        wy_i, _ = within_year_auc(x, "y_imd", "sat_lst_day_p90")
        ci = year_bootstrap(x, ["y_era5", "y_imd"], "sat_lst_day_p90", a.n_boot)
        boots[K] = ci
        rows.append({"label": f"HS{K}", "n": len(x), "prev_era5": x["y_era5"].mean(), "prev_imd": x["y_imd"].mean(),
                     "AUC_era5": safe_auc(x["y_era5"], x["sat_lst_day_p90"]),
                     "AUC_imd": safe_auc(x["y_imd"], x["sat_lst_day_p90"]),
                     "AUC_imd_lo95": ci["y_imd"][0], "AUC_imd_hi95": ci["y_imd"][1],
                     "diff_imd_minus_era5": safe_auc(x["y_imd"], x["sat_lst_day_p90"]) - safe_auc(x["y_era5"], x["sat_lst_day_p90"]),
                     "diff_lo95": ci["diff"][0], "diff_hi95": ci["diff"][1],
                     "within_year_AUC_era5": wy_e, "within_year_AUC_imd": wy_i, "n_within_years": nwy,
                     "PR_lift_imd": average_precision_score(x["y_imd"], x["sat_lst_day_p90"]) / x["y_imd"].mean()})
    sat = pd.DataFrame(rows)
    sat.to_csv(os.path.join(a.out, "satellite_vs_imd.csv"), index=False)
    print(sat.round(3).to_string(index=False))

    print("\nFrozen B1 model (saved test-year scores) vs the IMD label:")
    mrows = []
    for L in ("HS3", "HS2", "HS5"):
        p = os.path.join(a.pred_dir, L, "test_predictions.csv")
        if not os.path.exists(p):
            continue
        K = int(L[2:])
        m = pd.read_csv(p).merge(d[["District", "Year", "n35_imd"]], on=["District", "Year"], how="inner").dropna()
        if len(m) < 50:
            continue
        yi = (m["n35_imd"] >= K).astype(int)
        mrows.append({"label": L, "n": len(m), "test_years": f"{m['Year'].min()}-{m['Year'].max()}",
                      "AUC_vs_era5_label": safe_auc(m["y"], m["s"]), "AUC_vs_imd_label": safe_auc(yi, m["s"]),
                      "bal_acc_vs_era5": balanced_accuracy_score(m["y"], m["p"]),
                      "bal_acc_vs_imd": balanced_accuracy_score(yi, m["p"]),
                      "recall_vs_imd": m.loc[yi == 1, "p"].mean() if yi.sum() else np.nan,
                      "FPR_vs_imd": m.loc[yi == 0, "p"].mean() if (yi == 0).sum() else np.nan})
    if mrows:
        mt = pd.DataFrame(mrows)
        mt.to_csv(os.path.join(a.out, "frozen_model_vs_imd.csv"), index=False)
        print(mt.round(3).to_string(index=False))
    else:
        print(f"  (no test_predictions.csv under {a.pred_dir}/<LABEL>/ - run B1 first)")

    # ---- 4. verdict
    section("4. VERDICT against the pre-registered criteria (HS3)")
    kap = float(ag.loc[ag["label"] == "HS3", "kappa"].iloc[0])
    s3 = sat[sat["label"] == "HS3"].iloc[0]
    ca = kap >= KAPPA_MIN
    cb = s3["AUC_imd"] >= SAT_AUC_MIN
    cc = (s3["AUC_era5"] - s3["AUC_imd"]) <= SAT_AUC_DROP_MAX
    print(f"(a) ERA5 vs IMD HS3 label kappa = {kap:.3f} (>= {KAPPA_MIN}) -> {'ok' if ca else 'FAIL'}")
    print(f"(b) LST p90 vs IMD HS3 AUC = {s3['AUC_imd']:.3f} [{s3['AUC_imd_lo95']:.3f}, {s3['AUC_imd_hi95']:.3f}] "
          f"(>= {SAT_AUC_MIN}) -> {'ok' if cb else 'FAIL'}")
    print(f"(c) AUC drop vs ERA5 label = {s3['AUC_era5'] - s3['AUC_imd']:+.3f} (<= {SAT_AUC_DROP_MAX}) -> {'ok' if cc else 'FAIL'}")
    print()
    if cb and cc:
        msg = "MODIS LST detects heat exposure as defined by independent station-based IMD data too: the result is not an ERA5 artefact."
        if not ca:
            msg += " ERA5-Land and IMD disagree on the district label itself (see section 1/2: bias and the 1-degree IMD grid); " \
                   "report that as a label-uncertainty limitation."
    elif ca and not cb:
        msg = "The two labels agree but the satellite detector does not reach the bar on IMD: report B1 as consistent with ERA5-Land only."
    else:
        msg = "Both the labels and the satellite check fail against IMD at district level. Check state-year agreement (section 2) " \
              "before concluding; the 1-degree IMD grid may be too coarse for district labels. Report B1 as validated against ERA5-Land only."
    print(msg)
    print(f"\nAll tables saved in {a.out}")
    return {"kappa": kap, "AUC_imd": s3["AUC_imd"], "AUC_era5": s3["AUC_era5"]}


# ------------------------------------------------------------------ selftest
def selftest(a):
    import xarray as xr
    rng = np.random.default_rng(3)
    base = "/tmp/b3_selftest"
    os.makedirs(base + "/era5", exist_ok=True)
    days_all = {y: pd.date_range(pd.Timestamp(y - 1, 10, 1), pd.Timestamp(y, 5, 16)) for y in range(2003, 2025)}
    hot = {y: rng.normal() for y in days_all}
    lats, lons = np.arange(22.0, 33.0, 1.0), np.arange(73.0, 84.0, 1.0)
    dists = [(f"D{i}", "Punjab" if i < 6 else "Uttar Pradesh", rng.uniform(24, 31), rng.uniform(74, 82)) for i in range(12)]
    tab_rows, grid_t, grid_v = [], [], []
    for y, days in days_all.items():
        doy = np.arange(len(days))
        reg = 17 + 0.15 * np.clip(doy - 90, 0, None) + 4 * hot[y] + rng.normal(0, 1.5, len(days))
        gv = reg[:, None, None] - 0.8 + 0.1 * (lats[None, :, None] - 27) + rng.normal(0, 0.6, (len(days), len(lats), len(lons)))
        grid_t.append(days.values); grid_v.append(gv)
        er = []
        for dname, st, la, lo in dists:
            tm = reg + 0.3 * (la - 27) + rng.normal(0, 0.8, len(days))
            er.append(pd.DataFrame({"PROJECT_STATE": st, "PROJECT_DISTRICT": dname, "date": days,
                                    "temperature_2m_max": tm + 273.15}))
            P = 65
            w = tm[(days >= pd.Timestamp(y, 1, 1) + pd.Timedelta(days=P - 1)) &
                   (days <= pd.Timestamp(y, 1, 1) + pd.Timedelta(days=P + 39))]
            tab_rows.append({"State": st, "District": dname, "Year": y, "P_doy": P, "n35": int((w >= 35).sum()),
                             "sat_lst_day_p90": float(np.quantile(w, 0.9) + 2 + rng.normal(0, 1.2))})
        pd.concat(er).to_csv(f"{base}/era5/ts_era5_{y}.csv", index=False)
    pd.DataFrame(tab_rows).to_csv(base + "/exposure_table.csv", index=False)
    pd.DataFrame([{"District": d[0], "lat": d[2], "lon": d[3]} for d in dists]).to_csv(base + "/centroids.csv", index=False)
    t = np.concatenate(grid_t)
    v = np.concatenate(grid_v)
    da = xr.DataArray(v, coords={"time": t, "lat": lats, "lon": lons}, dims=("time", "lat", "lon"), name="tmax")
    a.table, a.centroids, a.era5_glob = base + "/exposure_table.csv", base + "/centroids.csv", base + "/era5/ts_era5_*.csv"
    a.out, a.pred_dir = base + "/out", base + "/none"
    return da


def main():
    ap = argparse.ArgumentParser(description="B3: validate the ERA5 label and the satellite detector against IMD Tmax")
    ap.add_argument("--centroids", default=CENTROIDS)
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--era5-glob", default=ERA5_GLOB)
    ap.add_argument("--imd-dir", default=IMD_DIR)
    ap.add_argument("--imd-nc", default=None, help="existing IMD Tmax NetCDF instead of downloading with imdlib")
    ap.add_argument("--pred-dir", default="reports/exposure", help="folder holding <LABEL>/test_predictions.csv from B1")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if os.path.isdir(os.path.join(root, "data")) and os.path.abspath(os.getcwd()) != root and not a.selftest:
        os.chdir(root)
    da = selftest(a) if a.selftest else None
    main_run(a, da)


if __name__ == "__main__":
    main()
    