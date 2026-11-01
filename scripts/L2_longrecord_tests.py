"""
L2 - FROZEN HYPOTHESES ON INDEPENDENT YEARS (state-level long record)
=====================================================================

The four hypotheses of A8 (H1 grain-fill rain harms, H2 anthesis moisture helps, H3 anthesis heat harms,
H4 grain-fill heat harms) were formed from DISTRICT data of Punjab/Haryana/UP in 2003-2022. This script
tests them, with the SAME frozen definitions (imported from A8), on data that played no role in forming them:

  PRIMARY    CONFIRMATORY   1979-2002   Punjab, Haryana, Uttar Pradesh    (real NDVI-derived anchors)
  SECONDARY  CONFIRMATORY   1979-2002   all states in the table          (MP/Rajasthan/Chhattisgarh anchors
                                                                          are ASSUMED -> weaker evidence)
  SENSITIVITY               1966-2002   Punjab, Haryana, UP              (ERA5 before 1979 is not homogeneous
                                                                          with later years - see L1 check)
  REPLICATION (exploratory) 2003-2017   Punjab, Haryana, UP              (state-level version of the data
                                                                          that generated the hypotheses)

Frozen choices (fixed before looking at any long-record result)
  * z-scores use a per-state climate normal, 1981-2010 (weather only, no yield information);
  * target = log-yield anomaly against a PAST-ONLY baseline (linear trend of the previous 10 years, >= 6 needed),
    scaled by the state's MAD of those residuals over 1972-2017 (affects pooling across states only);
  * statistic = pooled state-year Spearman, ONE-SIDED year-shuffle placebo (the same shuffle for every state,
    so cross-state co-movement is kept), Bonferroni over the 4 hypotheses (alpha = 0.0125);
  * "SUPPORTED" requires the expected sign AND Bonferroni p < 0.05 in a CONFIRMATORY set.

Also printed per set: years, effective number of independent state-years (from the between-state correlation of the
yield anomalies) and the smallest correlation detectable at 5% (one-sided) and with 80% power.

Input : data/processed/L_state_features.csv (L1)
Output: reports/yield_anomaly/longrecord_tests.csv
Run   : python scripts/L2_longrecord_tests.py [--perm 2000]
"""

import argparse
import os
import sys
import warnings
from math import sqrt, tanh

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import A8_preregistered_tests as a8                 # frozen hypothesis definitions and test machinery

warnings.filterwarnings("ignore")

IN_PATH = "data/processed/L_state_features.csv"
OUT_DIR = "reports/yield_anomaly"
CLIM = (1981, 2010)
CORE3 = ["Punjab", "Haryana", "Uttar Pradesh"]
SEED = 42
ALPHA = 0.05


def section(t):
    print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76)


def detectability(frame):
    w = frame.groupby(["Year", "State"])["anom"].mean().unstack()
    k = w.shape[1]
    if k < 2:
        return k, len(w), np.nan, np.nan, np.nan
    c = w.corr(method="spearman").values[np.triu_indices(k, 1)]
    rbar = float(np.nanmean(c))
    n_eff = max(len(w) * k / (1 + (k - 1) * max(rbar, 0)), 5)
    r5 = tanh(1.645 / sqrt(max(n_eff - 3, 1)))
    r80 = tanh(2.487 / sqrt(max(n_eff - 3, 1)))
    return rbar, n_eff, r5, r80, len(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(IN_PATH)
    need = sorted({f for fs, _, _ in a8.HYP.values() for f in fs})
    miss = [f for f in need if f not in df.columns]
    if miss:
        raise ValueError(f"features missing: {miss}")
    states_all = sorted(df["State"].unique())

    # ---------- frozen z-scores (per-state 1981-2010 normal) and hypothesis indices ----------
    ref = df[(df["Year"] >= CLIM[0]) & (df["Year"] <= CLIM[1])]
    mu = ref.groupby("State")[need].mean()
    sd = ref.groupby("State")[need].std().replace(0, np.nan)
    Z = (df[need].to_numpy(float) - mu.reindex(df["State"].values).to_numpy()) / sd.reindex(df["State"].values).to_numpy()
    Z = pd.DataFrame(Z, index=df.index, columns=need)
    for name, (fs, sg, _) in a8.HYP.items():
        df[name] = np.nanmean(np.column_stack([Z[f] * s for f, s in zip(fs, sg)]), axis=1)

    # ---------- anomalies ----------
    t = df.dropna(subset=["Yield"]).copy()
    t = t[t["Yield"] > 0].assign(y=lambda d: np.log(d["Yield"])).sort_values(["State", "Year"])
    parts = []
    for st, g in t.groupby("State"):
        g = g.assign(res=a8.trend10_resid(g))
        s = a8.mad_scale(g.loc[(g["Year"] >= 1972) & (g["Year"] <= 2017), "res"].to_numpy())
        parts.append(g.assign(anom=g["res"] / s))
    t = pd.concat(parts).dropna(subset=["anom"])

    sets = [
        ("REPLICATION (exploratory) 2003-2017, Punjab/Haryana/UP", CORE3, 2003, 2017, False),
        ("PRIMARY CONFIRMATORY 1979-2002, Punjab/Haryana/UP", CORE3, 1979, 2002, True),
        ("SECONDARY CONFIRMATORY 1979-2002, all states (some anchors ASSUMED)", states_all, 1979, 2002, True),
        ("SENSITIVITY 1966-2002, Punjab/Haryana/UP (pre-1979 ERA5 inhomogeneous)", CORE3, 1966, 2002, False),
    ]

    section("L2 - FROZEN HYPOTHESES ON INDEPENDENT YEARS")
    for n, (fs, sg, ex) in a8.HYP.items():
        print(f"  {n:28s} expected {'NEGATIVE' if ex < 0 else 'POSITIVE'}")
    print(f"states in table: {states_all}")

    rows, per_state = [], []
    for label, sts, lo, hi, confirmatory in sets:
        fr = t[t["State"].isin(sts) & (t["Year"] >= lo) & (t["Year"] <= hi)].copy()
        fr["District"] = fr["State"]
        if fr["Year"].nunique() < 8:
            print(f"\n[skip] {label}: fewer than 8 years")
            continue
        rbar, n_eff, r5, r80, ny = detectability(fr)
        section(label)
        print(f"state-years {len(fr)} | years {ny} | states {fr['State'].nunique()} | mean between-state anomaly correlation "
              f"{rbar:+.2f} | effective N = {n_eff:.0f} | detectable rho: 5% one-sided {r5:.2f}, 80% power {r80:.2f}")
        res = []
        for name, (_, _, ex) in a8.HYP.items():
            real, p = a8.placebo_p(fr, name, ex, "state", a.perm, rng)
            res.append({"set": label.split(" ")[0], "hypothesis": name, "expected": "neg" if ex < 0 else "pos",
                        "spearman": real, "p_one_sided": p, "p_bonferroni": min(1.0, p * len(a8.HYP)),
                        "n_years": ny, "N_eff": n_eff, "min_detectable_80": r80})
            for st, g in fr.groupby("State"):
                if g["Year"].nunique() >= 8:
                    per_state.append({"set": label.split(" ")[0], "hypothesis": name, "State": st,
                                      "spearman": spearmanr(g[name], g["anom"], nan_policy="omit")[0]})
        r = pd.DataFrame(res)
        sign = r["expected"].map({"neg": -1, "pos": 1})
        r["verdict"] = np.where(confirmatory, np.where((r["spearman"] * sign > 0) & (r["p_bonferroni"] < ALPHA),
                                                       "SUPPORTED", "NOT_SUPPORTED"), "exploratory_only")
        rows.append(r)
        print(r[["hypothesis", "expected", "spearman", "p_one_sided", "p_bonferroni", "verdict"]].round(3).to_string(index=False))

    out = pd.concat(rows, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, "longrecord_tests.csv"), index=False)

    section("DIRECTION CONSISTENCY: Spearman by set (expected sign: H1 -, H2 +, H3 -, H4 -)")
    piv = out.pivot(index="hypothesis", columns="set", values="spearman").round(2)
    print(piv.to_string())
    ps = pd.DataFrame(per_state)
    if len(ps):
        section("PER STATE (primary and secondary sets): Spearman")
        sub = ps[ps["set"].isin(["PRIMARY", "SECONDARY"])]
        print(sub.pivot_table(index=["set", "hypothesis"], columns="State", values="spearman").round(2).to_string())
    print("\nReading: a hypothesis counts as confirmed only if it is SUPPORTED in the PRIMARY set. SECONDARY adds climate "
          "diversity but rests on assumed windows for some states. Compare 'min_detectable_80' with the observed effect: "
          "an effect of about 0.2 needs far more independent state-years than 1979-2002 offers, so a NOT_SUPPORTED verdict "
          "means 'no effect large enough to detect', not 'no effect'.")


if __name__ == "__main__":
    main()