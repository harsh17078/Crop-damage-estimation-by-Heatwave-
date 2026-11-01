"""
A7 - EVENT DOSSIER: WHAT WAS DIFFERENT IN THE WORST (AND BEST) YEARS?
=====================================================================

A5/A6 found no statistical link between the heat / wet indices and yield anomaly, and the two worst
years (2014, 2021) looked weather-NORMAL in the indices. This script looks at those years directly,
descriptively (no model, no p-values):

 1. For every state, the worst and best years of the detrended DES yield anomaly.
 2. For those state-years, the 8 most extreme FEATURES (per-district z-score over all years, averaged
    over the state's districts) - shows which variables, if any, were unusual.
 3. From the RAW daily series (if data/raw/gee_ts exists), event-type statistics for Feb 1 - Apr 15
    that district-mean features can hide:
        rain_1d_max, rain_3d_max, days_ge10mm, days_ge25mm        (CHIRPS)
        tmax_max, days_ge35C, days_ge38C, tmin_min                (ERA5-Land)
    each shown with its PERCENTILE RANK over the years (100 = the most extreme year for that state).
 4. A verdict per event year: did ANY variable reach the 90th percentile?  If not, the loss is not
    explained by the available weather variables (hail, wind, disease or reporting noise are candidates).

Input : data/processed/A_features_signals.csv, data/processed/des_yield_clean.csv,
        data/raw/gee_ts/ts_chirps_YYYY.csv, ts_era5_YYYY.csv (optional)
Output: reports/yield_anomaly/event_dossier_*.csv
Run   : python scripts/A7_event_dossier.py [--worst 3] [--best 2]
"""

import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

IN_PATH = "data/processed/A_features_signals.csv"
DES_PATH = "data/processed/des_yield_clean.csv"
RAW_DIR = "data/raw/gee_ts"
OUT_DIR = "reports/yield_anomaly"


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def local_anomaly(g):
    y = np.log(g["Yield"].to_numpy(float))
    x = g["Year"].to_numpy(float)
    res = np.full(len(y), np.nan)
    for i in range(len(y)):
        m = (np.abs(x - x[i]) <= 3) & (x != x[i])
        if m.sum() >= 3:
            res[i] = y[i] - y[m].mean()
    s = 1.4826 * np.nanmedian(np.abs(res - np.nanmedian(res)))
    return res / s if s > 0 else res


def max_run_sum(r, k):
    return float(pd.Series(r).rolling(k).sum().max())


def raw_stats(districts):
    """District-year event statistics from the raw daily files (Feb 1 - Apr 15)."""
    rows = []
    chirps = sorted(glob.glob(os.path.join(RAW_DIR, "ts_chirps_*.csv")))
    era5 = {int(re.search(r"_(\d{4})\.csv$", f).group(1)): f for f in glob.glob(os.path.join(RAW_DIR, "ts_era5_*.csv"))}
    for f in chirps:
        y = int(re.search(r"_(\d{4})\.csv$", f).group(1))
        c = pd.read_csv(f)
        c["date"] = pd.to_datetime(c["date"])
        c = c[(c["date"] >= pd.Timestamp(y, 2, 1)) & (c["date"] <= pd.Timestamp(y, 4, 15)) & c["PROJECT_DISTRICT"].isin(districts)]
        col = "precipitation" if "precipitation" in c.columns else "mean"
        out = {}
        for (st, d), g in c.sort_values("date").groupby(["PROJECT_STATE", "PROJECT_DISTRICT"]):
            r = g[col].fillna(0).to_numpy()
            out[(st, d)] = {"State": st, "District": d, "Year": y, "rain_1d_max": r.max(), "rain_3d_max": max_run_sum(r, 3),
                            "days_ge10mm": int((r >= 10).sum()), "days_ge25mm": int((r >= 25).sum())}
        if y in era5:
            e = pd.read_csv(era5[y])
            e["date"] = pd.to_datetime(e["date"])
            e = e[(e["date"] >= pd.Timestamp(y, 2, 1)) & (e["date"] <= pd.Timestamp(y, 4, 15)) & e["PROJECT_DISTRICT"].isin(districts)]
            for (st, d), g in e.groupby(["PROJECT_STATE", "PROJECT_DISTRICT"]):
                tx = g["temperature_2m_max"].to_numpy() - 273.15
                tn = g["temperature_2m_min"].to_numpy() - 273.15
                out.setdefault((st, d), {"State": st, "District": d, "Year": y}).update(
                    {"tmax_max": tx.max(), "days_ge35C": int((tx >= 35).sum()), "days_ge38C": int((tx >= 38).sum()),
                     "tmin_min": tn.min()})
        rows.extend(out.values())
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worst", type=int, default=3)
    ap.add_argument("--best", type=int, default=2)
    a = ap.parse_args()

    df = pd.read_csv(IN_PATH)
    if "P_fallback" in df.columns:
        df = df[df["P_fallback"] == 0]
    des = pd.read_csv(DES_PATH)[["District", "Year", "Yield"]]
    t = df[["State", "District", "Year"]].merge(des, on=["District", "Year"], how="inner")
    t = t[t["Yield"] > 0].sort_values(["District", "Year"])
    t = pd.concat([g.assign(anom=local_anomaly(g)) for _, g in t.groupby("District") if len(g) >= 8])
    sy = t.groupby(["State", "Year"])["anom"].mean().reset_index().dropna()
    os.makedirs(OUT_DIR, exist_ok=True)

    feats = [c for c in df.columns if c.startswith(("veg_", "wx_"))]
    z = df[feats].sub(df.groupby("District")[feats].transform("mean")).div(
        df.groupby("District")[feats].transform("std").replace(0, np.nan))
    zf = pd.concat([df[["State", "Year"]], z], axis=1).groupby(["State", "Year"])[feats].mean()

    have_raw = bool(glob.glob(os.path.join(RAW_DIR, "ts_chirps_*.csv")))
    raw = None
    if have_raw:
        raw = raw_stats(set(df["District"]))
        sr = raw.groupby(["State", "Year"]).mean(numeric_only=True)
        metrics = [c for c in ("rain_1d_max", "rain_3d_max", "days_ge10mm", "days_ge25mm",
                               "tmax_max", "days_ge35C", "days_ge38C", "tmin_min") if c in sr.columns]
        pct = sr[metrics].groupby(level=0).rank(pct=True) * 100
        if "tmin_min" in pct:
            pct["tmin_min"] = 100 - pct["tmin_min"]          # colder = more extreme
        sr.to_csv(os.path.join(OUT_DIR, "event_dossier_raw_stats.csv"))
    else:
        print(f"[note] {RAW_DIR} not found - raw event statistics skipped (feature z-scores only).")

    verdicts = []
    for st, g in sy.groupby("State"):
        g = g.sort_values("anom")
        events = pd.concat([g.head(a.worst).assign(kind="WORST"), g.tail(a.best).assign(kind="BEST")])
        section(f"{st.upper()}  -  yield anomaly (MAD units) of {len(g)} years; worst {a.worst} and best {a.best} analysed")
        for _, r in events.iterrows():
            yr = int(r["Year"])
            print(f"\n[{r['kind']}] {yr}   state-mean yield anomaly {r['anom']:+.2f}")
            zr = zf.loc[(st, yr)].dropna()
            top = zr.reindex(zr.abs().sort_values(ascending=False).index).head(8)
            print("  most extreme features (z over years, state mean): " +
                  ", ".join(f"{k} {v:+.1f}" for k, v in top.items()))
            if raw is not None and (st, yr) in pct.index:
                p = pct.loc[(st, yr)]
                vals = sr.loc[(st, yr)]
                print("  raw event stats [value (percentile rank over years)]: " +
                      ", ".join(f"{m} {vals[m]:.0f} ({p[m]:.0f})" for m in metrics))
                top_pct = p.max()
                verdicts.append({"State": st, "Year": yr, "kind": r["kind"], "anom": r["anom"],
                                 "max_percentile": top_pct, "most_extreme": p.idxmax(),
                                 "max_abs_feature_z": float(top.abs().max())})
                if r["kind"] == "WORST":
                    print("  -> " + (f"weather WAS extreme: {p.idxmax()} at the {top_pct:.0f}th percentile"
                                    if top_pct >= 90 else "NO raw weather statistic reached the 90th percentile: "
                                    "not explained by these variables (consider hail / wind / disease / reporting)"))
    if verdicts:
        v = pd.DataFrame(verdicts)
        v.to_csv(os.path.join(OUT_DIR, "event_dossier_verdicts.csv"), index=False)
        section("SUMMARY: was the weather extreme in the worst years?")
        w = v[v["kind"] == "WORST"]
        print(w.round(2).to_string(index=False))
        print(f"\nworst state-years with a weather statistic >= 90th percentile: "
              f"{int((w['max_percentile'] >= 90).sum())} of {len(w)}")
        b = v[v["kind"] == "BEST"]
        print(f"best state-years with a weather statistic >= 90th percentile (extreme weather WITHOUT loss): "
              f"{int((b['max_percentile'] >= 90).sum())} of {len(b)}")
    print("\nReading: with 8 statistics per year, a random year reaches the 90th percentile in at least one of "
          "them quite often (up to 57% if they were independent), so only the 'best years' row gives the baseline: "
          "compare how many WORST and BEST state-years show a >= 90th-percentile weather extreme.")


if __name__ == "__main__":
    main()