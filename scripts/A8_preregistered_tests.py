"""
A8 - FROZEN MECHANISM HYPOTHESES: EXPLORATORY NOW, CONFIRMATORY ON NEW DATA
===========================================================================

STATUS OF THE HYPOTHESES (be honest about it in the report)
    They were formulated AFTER looking at 2003-2022 in Punjab/Haryana/UP (A5 coefficients, A7 event
    dossier). On those data they are EXPLORATORY: any p-value there is optimistic, because many
    specifications were already examined. The only valid test is on data that played no role in
    forming them:   (a) seasons after REF_END (2023, 2024 ...)   (b) states other than Punjab/
    Haryana/UP (MP, Rajasthan, ...).   Phase 2 does exactly that and nothing else.

THE FOUR HYPOTHESES (index = mean of sign-weighted z-scores; z uses a FROZEN per-district reference
(mean/std over years <= REF_END). Expected sign refers to the effect on yield.)
    H1  grain-fill rain harms         wx_fill_rain_sum, wx_fill_rain_max, wx_fill_sm_mean      (-)
    H2  anthesis moisture helps       wx_anth_rain_sum, -wx_anth_dry_max, wx_anth_sm_mean      (+)
    H3  anthesis heat harms           wx_anth_hot35_days, wx_anth_hdd35, wx_anth_anomhot_days,
                                      wx_anth_tmax_max                                          (-)
    H4  grain-fill heat harms         wx_fill_hot35_days, wx_fill_hdd35, wx_fill_anomhot_days,
                                      wx_fill_tmax_max                                          (-)
Multiplicity: Bonferroni over the 4 hypotheses (alpha = 0.0125).

PHASE 1 (exploratory; Punjab/Haryana/UP, years <= REF_END)
    target = log-yield anomaly against a LOCAL baseline (neighbouring years +-3, no data after REF_END)
    statistic = state-year Spearman, one-sided year-shuffle placebo p (same shuffle for all districts).
PHASE 2 (confirmatory; years > REF_END and/or states outside the exploratory set)
    target = log-yield anomaly against a PAST-ONLY baseline (linear trend of the district's previous
             10 years, >= 6 required), scaled by the pooled MAD of the reference residuals
    statistics = state-year and district-year Spearman, sign-concordance count with an exact one-sided
                 binomial p, and a year-shuffle placebo only if there are >= 8 distinct years.
    If no confirmatory yield data exist the script says what to add. Nothing is tuned in Phase 2.

Input : data/processed/A_features_signals.csv (A1; must contain the new years / states),
        data/processed/des_yield_clean.csv    (District, Year, Yield - append the new years / states)
Output: reports/yield_anomaly/prereg_*.csv
Run   : python scripts/A8_preregistered_tests.py
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr

warnings.filterwarnings("ignore")

IN_PATH = "data/processed/A_features_signals.csv"
DES_PATH = "data/processed/des_yield_clean.csv"
OUT_DIR = "reports/yield_anomaly"
REF_END = 2022
EXPLORATORY_STATES = ["Punjab", "Haryana", "Uttar Pradesh"]
ALPHA = 0.05
SEED = 42

HYP = {
    "H1_fill_rain_harms": (["wx_fill_rain_sum", "wx_fill_rain_max", "wx_fill_sm_mean"], [1, 1, 1], -1),
    "H2_anthesis_moisture_helps": (["wx_anth_rain_sum", "wx_anth_dry_max", "wx_anth_sm_mean"], [1, -1, 1], +1),
    "H3_anthesis_heat_harms": (["wx_anth_hot35_days", "wx_anth_hdd35", "wx_anth_anomhot_days", "wx_anth_tmax_max"], [1, 1, 1, 1], -1),
    "H4_fill_heat_harms": (["wx_fill_hot35_days", "wx_fill_hdd35", "wx_fill_anomhot_days", "wx_fill_tmax_max"], [1, 1, 1, 1], -1),
}


def section(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def local_resid(g):
    y, x = g["y"].to_numpy(float), g["Year"].to_numpy(float)
    r = np.full(len(y), np.nan)
    for i in range(len(y)):
        m = (np.abs(x - x[i]) <= 3) & (x != x[i])
        if m.sum() >= 3:
            r[i] = y[i] - y[m].mean()
    return r


def trend10_resid(g):
    y, x = g["y"].to_numpy(float), g["Year"].to_numpy(float)
    r = np.full(len(y), np.nan)
    for i in range(len(y)):
        m = (x < x[i]) & (x >= x[i] - 10)
        if m.sum() >= 6:
            b, a = np.polyfit(x[m], y[m], 1)
            r[i] = y[i] - (a + b * x[i])
    return r


def mad_scale(r):
    r = r[~np.isnan(r)]
    return 1.4826 * np.median(np.abs(r - np.median(r))) if len(r) else np.nan


def spear(frame, col, level):
    d = frame.dropna(subset=[col, "anom"])
    if level == "state":
        d = d.groupby(["State", "Year"])[[col, "anom"]].mean()
    if len(d) < 6 or d[col].nunique() < 3:
        return np.nan
    return spearmanr(d[col], d["anom"])[0]


def placebo_p(frame, col, sign, level, B, rng):
    years = sorted(frame["Year"].unique())
    real = spear(frame, col, level)
    base = frame.drop(columns=[col])
    null = []
    for _ in range(B):
        m = dict(zip(years, rng.permutation(years)))
        fp = frame[["District", "Year", col]].assign(Year=lambda d: d["Year"].map(m))
        null.append(spear(base.merge(fp, on=["District", "Year"]), col, level))
    null = np.array(null, dtype=float)
    return real, (1 + np.sum(null * sign >= real * sign)) / (B + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--ref-end", type=int, default=REF_END)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    ref_end = a.ref_end

    df = pd.read_csv(IN_PATH)
    if "P_fallback" in df.columns:
        df = df[df["P_fallback"] == 0]
    des = pd.read_csv(DES_PATH)[["District", "Year", "Yield"]]
    need = sorted({f for fs, _, _ in HYP.values() for f in fs})
    miss = [f for f in need if f not in df.columns]
    if miss:
        raise ValueError(f"features missing from the table: {miss}")

    # ---------- frozen indices ----------
    ref = df[df["Year"] <= ref_end]
    mu = ref.groupby("District")[need].mean()
    sd = ref.groupby("District")[need].std().replace(0, np.nan)
    gsd = ref[need].std().replace(0, np.nan)
    Z = (df[need] - mu.reindex(df["District"].values).to_numpy()) / sd.reindex(df["District"].values).fillna(gsd).to_numpy()
    Z = pd.DataFrame(Z, index=df.index, columns=need)
    for name, (fs, sg, _) in HYP.items():
        df[name] = np.nanmean(np.column_stack([Z[f] * s for f, s in zip(fs, sg)]), axis=1)

    t = df[["State", "District", "Year"] + list(HYP)].merge(des, on=["District", "Year"], how="inner")
    t = t[t["Yield"] > 0].assign(y=lambda d: np.log(d["Yield"])).sort_values(["District", "Year"])

    section("A8 - FROZEN MECHANISM HYPOTHESES")
    for n, (fs, sg, ex) in HYP.items():
        print(f"  {n:28s} expected effect on yield {'NEGATIVE' if ex < 0 else 'POSITIVE'} | " +
              ", ".join(("-" if s < 0 else "") + f for f, s in zip(fs, sg)))
    print(f"reference period for z-scores: years <= {ref_end}; Bonferroni alpha = {ALPHA / len(HYP):.4f}")

    # =============== PHASE 1 ===============
    section("PHASE 1 - EXPLORATORY (Punjab/Haryana/UP, years <= REF_END)   p-values are OPTIMISTIC")
    p1 = t[t["State"].isin(EXPLORATORY_STATES) & (t["Year"] <= ref_end)]
    parts = []
    for _, g in p1.groupby("District"):
        if len(g) >= 8:
            parts.append(g.assign(res=local_resid(g)))
    p1 = pd.concat(parts)
    s1 = mad_scale(p1["res"].to_numpy())
    p1["anom"] = p1["res"] / s1
    p1 = p1.dropna(subset=["anom"])
    print(f"rows {len(p1)} | districts {p1['District'].nunique()} | years {p1['Year'].min()}-{p1['Year'].max()}")
    rows = []
    for name, (_, _, ex) in HYP.items():
        for level in ("state", "district"):
            real, p = placebo_p(p1, name, ex, level, a.perm, rng)
            rows.append({"hypothesis": name, "level": level, "expected": "neg" if ex < 0 else "pos", "spearman": real,
                         "p_one_sided": p, "p_bonferroni": min(1.0, p * len(HYP))})
    r1 = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    r1.to_csv(os.path.join(OUT_DIR, "prereg_phase1.csv"), index=False)
    print(r1.round(3).to_string(index=False))

    # =============== PHASE 2 ===============
    section("PHASE 2 - CONFIRMATORY (years > REF_END and/or states outside the exploratory set)")
    parts = []
    for _, g in t.groupby("District"):
        parts.append(g.assign(res=trend10_resid(g)))
    allr = pd.concat(parts)
    refres = allr[(allr["Year"] <= ref_end) & allr["State"].isin(EXPLORATORY_STATES)]["res"].to_numpy()
    s2 = mad_scale(refres)
    allr["anom"] = allr["res"] / s2
    conf = allr[((allr["Year"] > ref_end) | (~allr["State"].isin(EXPLORATORY_STATES)))].dropna(subset=["anom"])
    if conf.empty:
        print("No confirmatory data yet. To run Phase 2, add to des_yield_clean.csv (and make sure A1 produced features for):")
        print(f"  * yields for seasons after {ref_end} (2023, 2024 ...) in Punjab/Haryana/UP, and/or")
        print("  * district yields 2003-2024 for other states (MP, Rajasthan, ...).")
        print("The hypotheses above are frozen; do not modify them after seeing those results.")
        return
    print(f"confirmatory rows {len(conf)} | states {sorted(conf['State'].unique())} | years {int(conf['Year'].min())}-{int(conf['Year'].max())}")
    rows = []
    for name, (_, _, ex) in HYP.items():
        sy = conf.groupby(["State", "Year"])[[name, "anom"]].mean().dropna()
        concord = int(((-ex * sy[name]) * sy["anom"] < 0).sum())       # stress index and anomaly point opposite ways
        n = int((sy[name] != 0).sum())
        pb = float(binom.sf(concord - 1, n, 0.5)) if n else np.nan
        r = {"hypothesis": name, "n_state_years": len(sy), "spearman_stateyear": spear(conf, name, "state"),
             "spearman_district": spear(conf, name, "district"), "sign_concordant": f"{concord}/{n}", "p_sign_one_sided": pb}
        if conf["Year"].nunique() >= 8:
            _, pp = placebo_p(conf, name, ex, "state", a.perm, rng)
            r["p_placebo"] = pp
        rows.append(r)
    r2 = pd.DataFrame(rows)
    r2["p_bonferroni_sign"] = (r2["p_sign_one_sided"] * len(HYP)).clip(upper=1)
    if "p_placebo" in r2:
        r2["p_bonferroni_placebo"] = (r2["p_placebo"] * len(HYP)).clip(upper=1)
    exp_sign = {n: ex for n, (_, _, ex) in HYP.items()}
    best_p = r2["p_bonferroni_placebo"] if "p_bonferroni_placebo" in r2 else r2["p_bonferroni_sign"]
    right_dir = [(r["spearman_stateyear"] * exp_sign[r["hypothesis"]]) > 0 for _, r in r2.iterrows()]
    r2["verdict"] = ["SUPPORTED" if (d and p < ALPHA) else "NOT_SUPPORTED" for d, p in zip(right_dir, best_p)]
    r2.to_csv(os.path.join(OUT_DIR, "prereg_phase2.csv"), index=False)
    print(r2.round(3).to_string(index=False))
    print("\nverdict = SUPPORTED only if the state-year Spearman has the EXPECTED sign AND the Bonferroni-adjusted p "
          "(placebo if >= 8 years, otherwise the sign test) is < 0.05. With only a few new state-years the sign test "
          "has little power (6 of 6 concordant gives p = 0.016): add MP/Rajasthan.")


if __name__ == "__main__":
    main()