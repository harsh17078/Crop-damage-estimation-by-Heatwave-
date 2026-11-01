"""
A6 - PRE-SPECIFIED STRESS-INDEX TEST (no fitted model, far fewer degrees of freedom)
====================================================================================

Why
---
With ~20 independent years, a model with 16 free coefficients cannot be learned (A5). The sample
supports ONE or TWO hypotheses with the sign fixed in advance. This script tests exactly those:

    HEAT  = mean z of  hot days (>=35C) in the anthesis and grain-fill windows, heat-degree days,
                       maximum Tmax, anomalous hot days            expected effect on yield: NEGATIVE
    WET   = mean z of  rain sum / maximum daily rain around anthesis and grain fill
                                                                   expected effect on yield: NEGATIVE
    STRESS= (HEAT + WET) / 2                                       expected: NEGATIVE
    VIGOR = mean z of  pre-peak NDVI (v2, v3)                      expected: POSITIVE (secondary)
  z-scores are per district over all years (features only, no labels, no fitting of any weights).

Target: detrended log yield. A linear trend leaves a low-frequency "hump" (2016-18 high, 2021-22
low in all three states), so the default is a LOCAL detrend: residual against the mean of the
district's neighbouring years (+-3, excluding the year itself). Options: linear | quad | local.

Tests (one-sided, sign pre-specified): Spearman of the index with the yield anomaly at STATE-YEAR
level (main) and DISTRICT-YEAR level, p-value from a PLACEBO in which the index of year k is assigned
to a random other year (same shuffle for every district, so regional co-movement is kept).

Also printed: per-state results, a year table (index vs anomaly) and a drop-one-year check.
Compare the extreme years with known events (2004 early heat, 2014 and 2015 rain/hail, 2022 heat wave).

Input : data/processed/A_features_signals.csv, data/processed/des_yield_clean.csv
Output: reports/yield_anomaly/index_test_*.csv
Run   : python scripts/A6_index_test.py [--detrend local|quad|linear] [--perm 2000]
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

IN_PATH = "data/processed/A_features_signals.csv"
DES_PATH = "data/processed/des_yield_clean.csv"
OUT_DIR = "reports/yield_anomaly"
SEED = 42

INDICES = {
    "HEAT": (["wx_anth_hot35_days", "wx_fill_hot35_days", "wx_fill_hdd35", "wx_fill_tmax_max",
              "wx_fill_anomhot_days", "wx_anth_tmax_max"], -1),
    "WET": (["wx_fill_rain_sum", "wx_fill_rain_max", "wx_anth_rain_sum", "wx_anth_rain_max"], -1),
    "VIGOR": (["veg_NDVI_v2", "veg_NDVI_v3"], +1),
}


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def detrend(g, mode):
    """Residual of log yield for one district (g sorted by Year)."""
    y = g["y"].to_numpy(float)
    x = g["Year"].to_numpy(float)
    if mode == "linear":
        b, a = np.polyfit(x, y, 1)
        return y - (a + b * x)
    if mode == "quad":
        c = np.polyfit(x - x.mean(), y, 2)
        return y - np.polyval(c, x - x.mean())
    res = np.full(len(y), np.nan)
    for i in range(len(y)):
        m = (np.abs(x - x[i]) <= 3) & (x != x[i])
        if m.sum() >= 3:
            res[i] = y[i] - y[m].mean()
    return res


def stat(frame, col, level):
    d = frame.dropna(subset=[col, "anom"])
    if level == "state":
        d = d.groupby(["State", "Year"])[[col, "anom"]].mean()
    if len(d) < 8 or d[col].nunique() < 3:
        return np.nan
    return spearmanr(d[col], d["anom"])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detrend", choices=["local", "quad", "linear"], default="local")
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--des", default=DES_PATH)
    ap.add_argument("--states", nargs="*", default=[])
    ap.add_argument("--drop-years", nargs="*", type=int, default=[])
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(IN_PATH)
    if "P_fallback" in df.columns:
        bad = sorted(df.loc[df["P_fallback"] == 1, "District"].unique())
        if bad:
            print(f"Excluding {len(bad)} non-wheat-peak district(s): {bad}")
        df = df[df["P_fallback"] == 0]
    des = pd.read_csv(a.des)[["District", "Year", "Yield"]]
    t = df[["State", "District", "Year"]].merge(des, on=["District", "Year"], how="inner")
    t = t[t["Yield"] > 0].assign(y=lambda d: np.log(d["Yield"])).sort_values(["District", "Year"])
    anom = []
    for d, g in t.groupby("District"):
        if len(g) >= 8:
            r = detrend(g, a.detrend)
            s = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
            anom.append(g.assign(anom=r / s if s > 0 else r))
    t = pd.concat(anom)[["State", "District", "Year", "anom"]]

    # ---------------- indices (z per district over all years; no labels used) ----------------
    idx_cols = {}
    for name, (cols, sign) in INDICES.items():
        have = [c for c in cols if c in df.columns]
        if not have:
            print(f"[skip] {name}: none of its features exist")
            continue
        z = df[have].sub(df.groupby("District")[have].transform("mean")).div(
            df.groupby("District")[have].transform("std").replace(0, np.nan))
        df[name] = z.mean(axis=1)
        idx_cols[name] = (have, sign)
    if "HEAT" in idx_cols and "WET" in idx_cols:
        df["STRESS"] = (df["HEAT"] + df["WET"]) / 2
        idx_cols["STRESS"] = (idx_cols["HEAT"][0] + idx_cols["WET"][0], -1)

    frame = t.merge(df[["District", "Year"] + list(idx_cols)], on=["District", "Year"], how="inner")
    if a.states:
        frame = frame[frame["State"].isin(a.states)]
    if a.drop_years:
        frame = frame[~frame["Year"].isin(a.drop_years)]
    years = sorted(frame["Year"].unique())

    section("A6 - STRESS-INDEX TEST")
    print(f"rows {len(frame)} | districts {frame['District'].nunique()} | years {years[0]}-{years[-1]} ({len(years)}) "
          f"| detrend {a.detrend} | states {a.states or 'all'} | dropped {a.drop_years or 'none'}")
    for n, (cols, sign) in idx_cols.items():
        print(f"  {n:7s} expected {'NEGATIVE' if sign < 0 else 'POSITIVE'} | {len(cols)} features: {', '.join(cols)}")

    # ---------------- tests with placebo ----------------
    section(f"RESULTS (one-sided p in the expected direction; {a.perm} year-shuffles)")
    rows = []
    for name, (cols, sign) in idx_cols.items():
        for level in ("state", "district"):
            real = stat(frame, name, level)
            if np.isnan(real):
                continue
            null = []
            for _ in range(a.perm):
                m = dict(zip(years, rng.permutation(years)))
                fp = frame[["District", "Year", name]].copy()
                fp["Year"] = fp["Year"].map(m)
                null.append(stat(frame.drop(columns=[name]).merge(fp, on=["District", "Year"]), name, level))
            null = np.array(null, dtype=float)
            p = (1 + np.sum(null * sign >= real * sign)) / (len(null) + 1)
            rows.append({"index": name, "level": level, "expected": "neg" if sign < 0 else "pos",
                         "spearman": real, "placebo_5%": np.nanpercentile(null, 5),
                         "placebo_95%": np.nanpercentile(null, 95), "p_one_sided": p})
    res = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(os.path.join(OUT_DIR, "index_test_results.csv"), index=False)
    print(res.round(3).to_string(index=False))

    # ---------------- per state (state-year level) ----------------
    if not a.states and "STRESS" in idx_cols:
        section("PER STATE (state-year Spearman of the STRESS / HEAT / WET index with yield anomaly)")
        out = []
        for st, g in frame.groupby("State"):
            r = {"State": st}
            for nm in ("HEAT", "WET", "STRESS"):
                if nm in g:
                    d = g.groupby("Year")[[nm, "anom"]].mean().dropna()
                    r[nm] = spearmanr(d[nm], d["anom"])[0] if len(d) >= 8 else np.nan
            out.append(r)
        print(pd.DataFrame(out).round(3).to_string(index=False))

    # ---------------- year table ----------------
    section("YEAR TABLE (mean over districts): index z vs detrended yield anomaly (MAD units)")
    cols = [c for c in ("HEAT", "WET", "STRESS") if c in frame]
    yt = frame.groupby("Year")[cols + ["anom"]].mean().round(2)
    print(yt.to_string())
    yt.to_csv(os.path.join(OUT_DIR, "index_test_year_table.csv"))
    if "STRESS" in yt:
        worst = yt.sort_values("anom").head(5).index.tolist()
        stress_top = yt.sort_values("STRESS", ascending=False).head(5).index.tolist()
        print(f"\n5 worst yield years      : {worst}")
        print(f"5 highest STRESS-index years: {stress_top}")
        print(f"overlap: {sorted(set(worst) & set(stress_top))}   (5 years drawn at random from {len(yt)} overlap "
              f"on average {25 / len(yt):.1f})")
        if "STRESS" in idx_cols:
            drop = []
            for y in years:
                f2 = frame[frame["Year"] != y]
                drop.append((y, stat(f2, "STRESS", "state")))
            print("STRESS state-year Spearman with one year dropped:", ", ".join(f"{y}:{v:+.2f}" for y, v in drop))
    print("\nReading: p < 0.05 in the EXPECTED direction = the pre-specified hypothesis is supported. "
          "A significant result of the WRONG sign (large p here) means the data contradict the hypothesis.")


if __name__ == "__main__":
    main()