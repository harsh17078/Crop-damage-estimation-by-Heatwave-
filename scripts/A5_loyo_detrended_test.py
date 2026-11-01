"""
A5 - IS THERE ANY WEATHER -> YIELD SIGNAL?  (leave-one-year-out, detrended, with a placebo test)
================================================================================================

Why a different design from A3/A4
---------------------------------
A3/A4 used forward-chaining with only 4 test years (2019-2022), a target that drifts with the yield
trend (yield z against a rolling baseline) and raw features that also drift (warming, NDVI): shared
time trends can create correlations that reverse later (A4 found hot grain-fill "raising" yield in
training years, lowering it in test years). This script removes those problems:

  * LEAVE-ONE-YEAR-OUT over all labelled years (15 test years instead of 4). For damage assessment
    after the season this is legitimate: the question is "given this year's weather, how far was the
    yield from normal?", not forecasting.
  * Everything is fitted inside each fold on the OTHER years only:
        target  = log yield minus the district's linear trend (fold-fitted), in units of its std
        features= per-district z-scores (district mean/std from the training years)
        model   = ridge (penalty chosen on the training rows)
  * A small physically motivated feature set (heat, rain, moisture, pre-peak greenness), not all 59.
  * PLACEBO TEST: the weather of year k is re-assigned to a random other year (same shuffle for all
    districts, so regional co-movement is kept) and the whole LOYO is repeated N times. The p-value is
    the share of shuffles that do as well as the real pairing.
  * BOOTSTRAP over years gives a confidence interval for the skill.

Reading
-------
  p < 0.05 and CI above 0  -> the features carry real (if modest) yield information.
  p large / CI contains 0  -> no detectable signal in this panel; more states / events or better
                              features are needed, no model change will fix it.

Input : data/processed/A_features_signals.csv (A1), data/processed/des_yield_clean.csv
Output: reports/yield_anomaly/loyo_*.csv
Run   : python scripts/A5_loyo_detrended_test.py [--perm 200] [--feats core|all]
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

IN_PATH = "data/processed/A_features_signals.csv"
DES_PATH = "data/processed/des_yield_clean.csv"
OUT_DIR = "reports/yield_anomaly"
MIN_PTS = 8                 # years needed to fit a district trend
ALPHAS = np.logspace(1, 5, 14)
ALPHA_MODE = "year"         # "year": penalty chosen with year-grouped folds (correct); "row": legacy leave-one-row-out
SEED = 42
CLIP = 3.0                  # winsorise the TRAINING target (std units); set by --clip
SD_MODE = "std"             # "std" or "mad" (robust scale; set by --sd)

CORE = ["wx_anth_tmax_max", "wx_fill_tmax_max", "wx_anth_hot35_days", "wx_fill_hot35_days",
        "wx_fill_hdd35", "wx_fill_anomhot_days", "wx_anth_tmin_mean", "wx_fill_tmin_mean",
        "wx_anth_rain_sum", "wx_fill_rain_sum", "wx_fill_rain_max", "wx_fill_vpd_mean",
        "wx_pre_frost_days", "wx_anth_sm_mean", "veg_NDVI_v3", "veg_NDMI_v3"]


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def fit_ridge(X, y, years):
    """Ridge with the penalty chosen by YEAR-GROUPED folds: districts of one year are near-duplicates, so a
    row-wise leave-one-out (legacy) lets siblings of the held-out row into training and under-regularises."""
    if ALPHA_MODE == "row" or len(np.unique(years)) < 4:
        return RidgeCV(alphas=ALPHAS).fit(X, y)
    k = min(5, len(np.unique(years)))
    splits = list(GroupKFold(k).split(X, y, years))
    return RidgeCV(alphas=ALPHAS, cv=splits).fit(X, y)


def loyo(frame, feats):
    """Leave-one-year-out predictions. frame: State, District, Year, y (=log yield), feats."""
    out = []
    for v in sorted(frame["Year"].unique()):
        tr, te = frame[frame["Year"] != v], frame[frame["Year"] == v]
        fit = {}
        for d, g in tr.groupby("District"):
            if len(g) >= MIN_PTS:
                b, a = np.polyfit(g["Year"].to_numpy(float), g["y"].to_numpy(float), 1)
                fit[d] = (a, b)
        if not fit:
            continue
        A = {d: ab[0] for d, ab in fit.items()}
        B = {d: ab[1] for d, ab in fit.items()}

        def target(df):
            a, b = df["District"].map(A), df["District"].map(B)
            return df["y"] - (a + b * df["Year"])

        r_tr, r_te = target(tr), target(te)
        r0 = r_tr.dropna()
        sd = r0.std() if SD_MODE == "std" else 1.4826 * np.median(np.abs(r0 - np.median(r0)))
        t_tr, t_te = (r_tr / sd).to_numpy(), (r_te / sd).to_numpy()

        mu = tr.groupby("District")[feats].mean()
        sdv = tr.groupby("District")[feats].std()
        g_sd = tr[feats].std().replace(0, np.nan).fillna(1.0).to_numpy()
        g_mu = tr[feats].mean().to_numpy()

        def z(df):
            m = mu.reindex(df["District"].values).to_numpy()
            s = sdv.reindex(df["District"].values).to_numpy()
            m = np.where(np.isnan(m), g_mu, m)
            s = np.maximum(np.where(np.isnan(s), g_sd, s), 0.25 * g_sd)
            return np.nan_to_num((df[feats].to_numpy(dtype=float) - m) / s)

        ok_tr = ~np.isnan(t_tr)
        ok_te = ~np.isnan(t_te)
        if ok_tr.sum() < 100 or ok_te.sum() == 0:
            continue
        model = fit_ridge(z(tr)[ok_tr], np.clip(t_tr[ok_tr], -CLIP, CLIP), tr["Year"].to_numpy()[ok_tr])
        p = model.predict(z(te)[ok_te])
        out.append(pd.DataFrame({"State": te["State"].values[ok_te], "District": te["District"].values[ok_te],
                                 "Year": v, "y": t_te[ok_te], "p": p}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def stats(r):
    if r.empty or r["p"].nunique() < 2:
        return {"spearman": np.nan, "spearman_stateyear": np.nan, "skill": np.nan, "AUC_loss": np.nan}
    sy = r.groupby(["State", "Year"])[["p", "y"]].mean()
    loss = (r["y"] <= -1.0).astype(int)
    return {"spearman": spearmanr(r["p"], r["y"])[0],
            "spearman_stateyear": spearmanr(sy["p"], sy["y"])[0] if len(sy) >= 8 else np.nan,
            "skill": 1 - np.mean((r["y"] - r["p"]) ** 2) / np.mean(r["y"] ** 2),
            "AUC_loss": roc_auc_score(loss, -r["p"]) if loss.nunique() == 2 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--feats", choices=["core", "all"], default="core")
    ap.add_argument("--des", default=DES_PATH)
    ap.add_argument("--drop-years", nargs="*", type=int, default=[], help="leave these years out entirely, e.g. 2014")
    ap.add_argument("--states", nargs="*", default=[], help="analyse only these states, e.g. Punjab or 'Uttar Pradesh'")
    ap.add_argument("--sd", choices=["std", "mad"], default="std", help="scale of the target; mad is robust to outlier years")
    ap.add_argument("--clip", type=float, default=3.0, help="winsorise the training target at +-clip std (99 = off)")
    ap.add_argument("--alpha", choices=["year", "row"], default="year", help="penalty selection: year-grouped (default) or legacy row-wise")
    a = ap.parse_args()
    global CLIP, SD_MODE, ALPHA_MODE
    CLIP, SD_MODE, ALPHA_MODE = a.clip, a.sd, a.alpha
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(IN_PATH)
    if "P_fallback" in df.columns:
        bad = sorted(df.loc[df["P_fallback"] == 1, "District"].unique())
        if bad:
            print(f"Excluding {len(bad)} non-wheat-peak district(s): {bad}")
        df = df[df["P_fallback"] == 0]
    des = pd.read_csv(a.des)[["District", "Year", "Yield"]]
    all_feats = [c for c in df.columns if c.startswith(("veg_", "wx_"))]
    feats = [c for c in CORE if c in df.columns] if a.feats == "core" else all_feats
    if not feats:
        raise ValueError("No features found.")

    tgt = df[["State", "District", "Year"]].merge(des, on=["District", "Year"], how="inner")
    tgt = tgt[tgt["Yield"] > 0].assign(y=lambda d: np.log(d["Yield"]))[["State", "District", "Year", "y"]]
    frame = tgt.merge(df[["District", "Year"] + feats], on=["District", "Year"], how="inner")
    if a.states:
        frame, tgt = frame[frame["State"].isin(a.states)], tgt[tgt["State"].isin(a.states)]
    if a.drop_years:
        frame, tgt = frame[~frame["Year"].isin(a.drop_years)], tgt[~tgt["Year"].isin(a.drop_years)]
    years = sorted(frame["Year"].unique())

    section("A5 - LEAVE-ONE-YEAR-OUT, DETRENDED")
    print(f"options: states {a.states or 'all'} | dropped years {a.drop_years or 'none'} | target scale {a.sd} | clip {a.clip}")
    print(f"rows {len(frame)} | districts {frame['District'].nunique()} | years {years[0]}-{years[-1]} "
          f"({len(years)}) | features {len(feats)} ({a.feats})")

    # ---------------- real pairing ----------------
    pred = loyo(frame, feats)
    real = stats(pred)
    os.makedirs(OUT_DIR, exist_ok=True)
    pred.to_csv(os.path.join(OUT_DIR, "loyo_predictions.csv"), index=False)
    print("\nREAL PAIRING:", {k: round(v, 3) for k, v in real.items()})

    # ---------------- bootstrap over years ----------------
    section("BOOTSTRAP OVER YEARS (95% interval)")
    by = {y: g for y, g in pred.groupby("Year")}
    bs = []
    for _ in range(a.boot):
        ys = rng.choice(years, size=len(years), replace=True)
        r = pd.concat([by[y] for y in ys if y in by], ignore_index=True)
        s = stats(r)
        bs.append([s["spearman"], s["spearman_stateyear"], s["skill"]])
    bs = np.array(bs, dtype=float)
    for i, nm in enumerate(["spearman (district-year)", "spearman (state-year)", "skill vs zero"]):
        lo, hi = np.nanpercentile(bs[:, i], [2.5, 97.5])
        print(f"  {nm:26s} {np.nanmean(bs[:, i]):+.3f}   95% interval [{lo:+.3f}, {hi:+.3f}]")

    # ---------------- placebo ----------------
    section(f"PLACEBO: weather re-assigned to a random year ({a.perm} shuffles)")
    feat_tbl = frame[["District", "Year"] + feats]
    null = []
    for _ in range(a.perm):
        perm = rng.permutation(years)
        m = dict(zip(years, perm))
        fp = feat_tbl.assign(Year=feat_tbl["Year"].map(m))
        fr = tgt.merge(fp, on=["District", "Year"], how="inner")
        null.append(stats(loyo(fr, feats)))
    null = pd.DataFrame(null)
    if "spearman_stateyear" in null:
        print(f"  -> with {len(years)} years, a state-year Spearman below ~{null['spearman_stateyear'].quantile(0.95):+.2f} "
              "cannot be told apart from chance (minimum detectable effect).")
    for k in ("spearman", "spearman_stateyear", "skill", "AUC_loss"):
        if k in null and not np.isnan(real[k]):
            pv = (1 + (null[k] >= real[k]).sum()) / (len(null) + 1)
            print(f"  {k:20s} real {real[k]:+.3f} | placebo mean {null[k].mean():+.3f}, 95th pct "
                  f"{null[k].quantile(0.95):+.3f} | p = {pv:.3f}")

    # ---------------- by year ----------------
    section("BY YEAR (mean over districts): predicted vs actual detrended yield anomaly (std units)")
    yt = pred.groupby("Year").agg(n=("y", "size"), mean_pred=("p", "mean"), mean_actual=("y", "mean")).round(3)
    print(yt.to_string())
    print(f"\ncorrelation across years of the yearly means: {spearmanr(yt['mean_pred'], yt['mean_actual'])[0]:+.3f}")
    drop = []
    for y in years:
        r = pred[pred["Year"] != y]
        drop.append((y, stats(r)["spearman_stateyear"]))
    print("state-year Spearman when one year is dropped:",
          ", ".join(f"{y}:{v:+.2f}" for y, v in drop))
    print("\nReading: p < 0.05 and an interval above 0 = real signal. If the result collapses when one year is "
          "dropped, a single event is driving it.")

    # ---------------- coefficients of the final model on all years ----------------
    sc = []
    mu = frame.groupby("District")[feats].transform("mean")
    sd = frame.groupby("District")[feats].transform("std").replace(0, np.nan)
    Z = np.nan_to_num(((frame[feats] - mu) / sd).to_numpy(dtype=float))
    fit = {d: np.polyfit(g["Year"].to_numpy(float), g["y"].to_numpy(float), 1)
           for d, g in frame.groupby("District") if len(g) >= MIN_PTS}
    res = frame.apply(lambda r: r["y"] - np.polyval(fit[r["District"]], r["Year"]) if r["District"] in fit else np.nan, axis=1)
    ok = res.notna().to_numpy()
    mdl = fit_ridge(Z[ok], np.clip((res[ok] / res[ok].std()).to_numpy(), -CLIP, CLIP), frame["Year"].to_numpy()[ok])
    coef = pd.Series(mdl.coef_, index=feats).sort_values()
    coef.to_csv(os.path.join(OUT_DIR, "loyo_coefficients.csv"))
    section("RIDGE COEFFICIENTS (all years; standardised) - heat should be NEGATIVE")
    print(coef.round(3).to_string())


if __name__ == "__main__":
    main()