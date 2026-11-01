"""
AUDIT_feasibility - CAN THIS DATA SUPPORT A WEATHER -> YIELD-ANOMALY MODEL AT ALL?
==================================================================================

Before any more modelling, measure what is theoretically attainable. Eight checks:

 1. DATA INTEGRITY   rows, duplicates, years per district, missing, yield outliers, drifting features,
                     and the DES year-alignment test (NDVI signal vs anomaly at shift -1/0/+1).
 2. TARGET VARIANCE  how much of the detrended yield anomaly is SHARED by districts in the same
                     state-year (intraclass correlation, ICC). Weather is regional, so ICC is the CEILING
                     for any weather model: the district-specific part (management, reporting noise) cannot
                     be explained by weather.
 3. ORACLE CEILING   leave-one-district-out R2 / Spearman of "anomaly = mean anomaly of the other districts
                     of the same state-year": what a PERFECT regional-weather model could reach.
 4. TEMPORAL         lag-1 autocorrelation of state-year anomalies and correlation between states.
 5. EFFECTIVE N      independent state-year observations and the smallest correlation detectable at
                     5% (one-sided) and with 80% power.
 6. FEATURE STRUCTURE  regional share of feature variance, drifting features, effective dimensionality.
 7. GLOBAL NULL TEST every feature vs the state-year anomaly with a year-shuffle placebo: how many features
                     are significant versus the number expected by chance (BH-FDR, p-value uniformity).
 8. TARGET STABILITY do linear / quadratic / local detrending agree on which years were bad?

A verdict is printed at the end. It never promises accuracy: it states what the data can support.

Input : data/processed/A_features_signals.csv (A1), data/processed/des_yield_clean.csv
Output: reports/audit/*.csv
Run   : python scripts/AUDIT_feasibility.py [--perm 2000]
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kstest, rankdata, spearmanr

warnings.filterwarnings("ignore")

IN_PATH = "data/processed/A_features_signals.csv"
DES_PATH = "data/processed/des_yield_clean.csv"
OUT_DIR = "reports/audit"
SEED = 42


def section(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def detrend(g, mode):
    y = np.log(g["Yield"].to_numpy(float))
    x = g["Year"].to_numpy(float)
    if mode == "linear":
        b, a = np.polyfit(x, y, 1)
        r = y - (a + b * x)
    elif mode == "quad":
        c = np.polyfit(x - x.mean(), y, 2)
        r = y - np.polyval(c, x - x.mean())
    else:
        r = np.full(len(y), np.nan)
        for i in range(len(y)):
            m = (np.abs(x - x[i]) <= 3) & (x != x[i])
            if m.sum() >= 3:
                r[i] = y[i] - y[m].mean()
    s = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
    return r / s if s > 0 else r


def anomalies(t, mode):
    out = []
    for _, g in t.groupby("District"):
        if len(g) >= 8:
            out.append(g.assign(anom=detrend(g.sort_values("Year"), mode)))
    return pd.concat(out).dropna(subset=["anom"])


def icc(d, col, grp):
    d = d[[col] + grp].dropna()
    g = d.groupby(grp)[col]
    n = g.size().to_numpy(float)
    m = g.mean().to_numpy()
    N, G = n.sum(), len(n)
    if G < 2 or N <= G:
        return np.nan
    grand = d[col].mean()
    msb = (n * (m - grand) ** 2).sum() / (G - 1)
    msw = ((d[col] - g.transform("mean")) ** 2).sum() / (N - G)
    n0 = (N - (n ** 2).sum() / N) / (G - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = np.empty_like(p, dtype=float)
    m = len(p)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        k = m - rank
        prev = min(prev, p[i] * m / k)
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--des", default=DES_PATH)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(IN_PATH)
    if "P_fallback" in df.columns:
        df = df[df["P_fallback"] == 0]
    des = pd.read_csv(a.des)[["District", "Year", "Yield"]]
    feats = [c for c in df.columns if c.startswith(("veg_", "wx_"))]
    t = df[["State", "District", "Year"]].merge(des, on=["District", "Year"], how="inner")
    t = t[t["Yield"] > 0].sort_values(["District", "Year"])

    # =============== 1. DATA INTEGRITY ===============
    section("1. DATA INTEGRITY")
    cnt = t.groupby("District")["Year"].nunique()
    print(f"labelled rows {len(t)} | districts {t['District'].nunique()} | years {t['Year'].min()}-{t['Year'].max()} "
          f"| years/district: min {cnt.min()}, median {int(cnt.median())}")
    print(f"duplicate District-Year rows: {int(t.duplicated(['District', 'Year']).sum())}")
    A = anomalies(t, "local")
    out = A[np.abs(A["anom"]) > 5]
    print(f"yield outliers (|local anomaly| > 5 MAD): {len(out)} rows"
          + (f" e.g. {out[['District', 'Year']].head(5).values.tolist()}" if len(out) else ""))
    miss = df[feats].isna().mean()
    print(f"features {len(feats)} | with >5% missing: {int((miss > 0.05).sum())} | constant: "
          f"{int((df[feats].std() < 1e-12).sum())}")
    drift = {f: spearmanr(df["Year"], df[f], nan_policy="omit")[0] for f in feats}
    drifting = sorted([f for f, r in drift.items() if abs(r) > 0.4], key=lambda f: -abs(drift[f]))
    print(f"features drifting with Year (|rho| > 0.4): {len(drifting)}"
          + (": " + ", ".join(f"{f} {drift[f]:+.2f}" for f in drifting[:6]) if drifting else ""))
    if "S_ndvi" in df.columns:
        d = df[["District", "Year", "S_ndvi"]].copy()
        d["s"] = d["S_ndvi"] - d.groupby("District")["S_ndvi"].transform("mean")
        res = []
        for sh in (-1, 0, 1):
            y2 = A[["District", "Year", "anom"]].assign(Year=lambda x: x["Year"] - sh)
            m = d.merge(y2, on=["District", "Year"]).dropna()
            res.append(f"shift {sh:+d}: {spearmanr(m['s'], m['anom'])[0]:+.3f}")
        print("year alignment (NDVI signal vs anomaly): " + " | ".join(res) + "   (compare |rho|: shift +0 should be the largest)")

    # =============== 2. TARGET VARIANCE ===============
    section("2. HOW MUCH OF THE YIELD ANOMALY IS REGIONAL? (the ceiling for any weather model)")
    icc_sy = icc(A, "anom", ["State", "Year"])
    icc_y = icc(A, "anom", ["Year"])
    print(f"ICC state-year : {icc_sy:.3f}   (share of anomaly variance common to districts of one state-year)")
    print(f"ICC year only  : {icc_y:.3f}")
    A["_n"] = A.groupby(["State", "Year"])["anom"].transform("size")
    ok = A["_n"] > 1
    pred = (A.groupby(["State", "Year"])["anom"].transform("sum") - A["anom"]) / (A["_n"] - 1)
    r2 = 1 - ((A["anom"] - pred)[ok] ** 2).sum() / ((A["anom"][ok] - A["anom"][ok].mean()) ** 2).sum()
    rho = spearmanr(pred[ok], A["anom"][ok])[0]
    print(f"ORACLE regional predictor (mean of the other districts, same state-year): R2 = {r2:+.3f}, Spearman = {rho:+.3f}")
    print("  -> a weather model can at best approach this; the rest is district-specific noise or management.")

    # =============== 4. TEMPORAL ===============
    section("4. TEMPORAL STRUCTURE")
    sy = A.groupby(["State", "Year"])["anom"].mean().unstack(0)
    ac = {s: sy[s].autocorr(1) for s in sy.columns}
    print("lag-1 autocorrelation of state anomalies: " + ", ".join(f"{s} {v:+.2f}" for s, v in ac.items()))
    cm = sy.corr(method="spearman")
    print("correlation between states' anomaly series:")
    print(cm.round(2).to_string())

    # =============== 5. EFFECTIVE N / POWER ===============
    section("5. EFFECTIVE SAMPLE SIZE AND DETECTABLE EFFECT")
    S, Y = sy.shape[1], sy.shape[0]
    pairs = cm.values[np.triu_indices(S, 1)]
    rbar = float(np.nanmean(pairs)) if len(pairs) else 0.0
    neff_states = S / (1 + (S - 1) * max(rbar, 0))
    neff = max(Y * neff_states, 5)
    from math import sqrt, tanh
    r_a = tanh(1.645 / sqrt(max(neff - 3, 1)))
    r_80 = tanh((1.645 + 0.842) / sqrt(max(neff - 3, 1)))
    print(f"years {Y} | states {S} | mean between-state correlation {rbar:+.2f} -> effective independent states "
          f"{neff_states:.2f} -> effective N = {neff:.0f}")
    print(f"smallest correlation detectable at 5% (one-sided): {r_a:.2f}   with 80% power: {r_80:.2f}")
    print("  (analytic estimate - optimistic for clustered weather; the year-shuffle thresholds in A5/A6, about 0.35 "
          "on your data, are the more realistic figure)")

    # =============== 6. FEATURE STRUCTURE ===============
    section("6. FEATURE STRUCTURE")
    z = df[feats].sub(df.groupby("District")[feats].transform("mean")).div(
        df.groupby("District")[feats].transform("std").replace(0, np.nan))
    zz = pd.concat([df[["State", "Year"]], z], axis=1)
    f_icc = pd.Series({f: icc(zz, f, ["State", "Year"]) for f in feats})
    print(f"regional share of feature variance (ICC state-year): median {f_icc.median():.2f} "
          f"(weather is regional; target ICC is {icc_sy:.2f})")
    Fsy = zz.groupby(["State", "Year"])[feats].mean().dropna(axis=1)
    ev = np.linalg.svd((Fsy - Fsy.mean()) / Fsy.std().replace(0, 1), compute_uv=False) ** 2
    k90 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.90) + 1)
    print(f"effective dimensionality: {k90} principal components explain 90% of the state-year feature variance "
          f"(vs {len(feats)} columns and effective N = {neff:.0f})")

    # =============== 7. GLOBAL NULL TEST ===============
    section(f"7. GLOBAL NULL TEST - every feature vs state-year anomaly ({a.perm} year-shuffles)")
    Aw = A.groupby(["State", "Year"])["anom"].mean().unstack(1)
    Fw = {f: zz.groupby(["State", "Year"])[f].mean().unstack(1) for f in Fsy.columns}
    years = [y for y in Aw.columns if all(y in Fw[f].columns for f in list(Fw)[:1])]
    states = list(Aw.index)
    cols = [y for y in years if not Aw[y].isna().any()]
    Aarr = Aw.loc[states, cols].to_numpy()
    names, R = [], []
    for f in Fw:
        M = Fw[f].reindex(index=states, columns=cols).to_numpy()
        if np.isnan(M).any() or np.nanstd(M) == 0:
            continue
        names.append(f)
        R.append(rankdata(M.ravel()).reshape(M.shape))
    R = np.array(R)                                   # K x S x Y
    rA = rankdata(Aarr.ravel())
    rAc = rA - rA.mean()
    nA = np.sqrt((rAc ** 2).sum())

    def corrs(Rp):
        flat = Rp.reshape(Rp.shape[0], -1)
        fc = flat - flat.mean(1, keepdims=True)
        return (fc @ rAc) / (np.sqrt((fc ** 2).sum(1)) * nA)

    obs = corrs(R)
    null = np.empty((a.perm, len(names)))
    for b in range(a.perm):
        null[b] = corrs(R[:, :, rng.permutation(R.shape[2])])
    p = (1 + (np.abs(null) >= np.abs(obs)[None, :]).sum(0)) / (a.perm + 1)
    q = bh(p)
    tab = pd.DataFrame({"feature": names, "spearman": obs, "p_two_sided": p, "q_BH": q}).sort_values("p_two_sided")
    tab.to_csv(os.path.join(OUT_DIR, "global_null_features.csv"), index=False)
    K = len(names)
    print(f"{K} features tested at state-year level ({len(cols)} years x {len(states)} states)")
    print(f"p < 0.05: {int((p < 0.05).sum())} (expected by chance {0.05 * K:.1f}) | p < 0.01: {int((p < 0.01).sum())} "
          f"(expected {0.01 * K:.1f}) | BH q < 0.10: {int((q < 0.10).sum())}")
    ks = kstest(p, "uniform").pvalue
    print(f"uniformity of the p-values (KS test p = {ks:.2f}): a uniform distribution means 'no signal in any feature'")
    print(tab.head(6).round(3).to_string(index=False))

    # =============== 8. TARGET STABILITY ===============
    section("8. IS THE TARGET WELL DEFINED? (agreement between detrending choices)")
    var = {m: anomalies(t, m).groupby(["State", "Year"])["anom"].mean() for m in ("linear", "quad", "local")}
    for m1, m2 in (("linear", "quad"), ("linear", "local"), ("quad", "local")):
        j = pd.concat([var[m1], var[m2]], axis=1, keys=["a", "b"]).dropna()
        print(f"  state-year Spearman {m1} vs {m2}: {spearmanr(j['a'], j['b'])[0]:+.2f}")

    # =============== VERDICT ===============
    section("VERDICT (what the data can support)")
    n_sig = int((q < 0.10).sum())
    lines = []
    lines.append(f"* Regional (shareable) part of the yield anomaly: ICC = {icc_sy:.2f}; oracle regional R2 = {r2:+.2f}. "
                 + ("Most of the variance is district-specific, so even a perfect regional weather model could explain little."
                    if icc_sy < 0.25 else "A substantial regional component exists, so a weather effect is possible in principle."))
    lines.append(f"* Effective sample: {neff:.0f} independent state-years; a correlation must exceed {r_a:.2f} to be detected at 5% "
                 f"and {r_80:.2f} for 80% power. " + ("Smaller true effects cannot be found with this data."
                                                       if r_a > 0.25 else ""))
    lines.append(f"* Features vs target: {int((p < 0.05).sum())} of {K} features at p<0.05 (chance: {0.05 * K:.1f}); "
                 f"{n_sig} survive BH-FDR. " + ("No evidence of any feature-yield relationship." if n_sig == 0 and (p < 0.05).sum() <= 0.05 * K + 3
                                                else "Some features deserve a pre-specified follow-up test."))
    lines.append("* Recommendation: " + (
        "do NOT train a predictive classifier/regressor on this panel. Report the audit as a rigorous negative/low-power result, "
        "add independent events (more seasons, MP/Rajasthan, better-quality yield such as crop-cutting or insurance data) and use "
        "event case studies (A7) for the narrative."
        if (n_sig == 0 and icc_sy < 0.35) else
        "a modest regional model is defensible: use few pre-specified indices with year-grouped validation, and report the oracle "
        "ceiling next to the achieved skill."))
    for l in lines:
        print(l)
    pd.Series({"icc_state_year": icc_sy, "oracle_r2": r2, "oracle_spearman": rho, "n_eff": neff,
               "min_detectable_5pct": r_a, "min_detectable_80pct": r_80, "n_features_q10": n_sig}).to_csv(
        os.path.join(OUT_DIR, "audit_summary.csv"))


if __name__ == "__main__":
    main()