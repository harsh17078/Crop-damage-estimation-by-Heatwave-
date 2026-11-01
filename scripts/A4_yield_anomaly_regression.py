"""
A4 - DO THE FEATURES CARRY ANY INFORMATION ABOUT YIELD ANOMALY?  (continuous version of the test)
================================================================================================

Why
---
Binary yield-loss classification on this panel has ~45 positive events (z <= -1), 14 of them in
one year (2021): far too few to learn from, and the CV scores it produced were selection noise.
This script uses EVERY labelled row (continuous target) and asks two simpler questions.

PART 1  Univariate scan
    For each feature: within-district Spearman correlation with the yield anomaly, on training
    years (< SPLIT_YEAR) and on test years. A feature is only credible if it has the same sign and a
    non-trivial size in BOTH periods.

PART 2  Rolling-origin regression (one year ahead, everything refit on earlier years only)
    target  = yield z  (past-only z of log yield, see proxy_label_lib), winsorised at +-3 for training
    models  = zero baseline | ridge | rf | hgb        feature sets = veg | wx | all
    metrics = RMSE skill vs zero baseline, Spearman(pred, y), state-year Spearman,
              AUC of -pred for 'yield loss' (y <= -1)   <- prevalence-independent damage detection
    A derived classifier ("damage if predicted z <= thr") needs no extra training: if AUC > ~0.65 on
    the TEST years the features can detect yield loss; if ~0.5 they cannot, whatever the model.

Input : data/processed/A_features_signals.csv (A1), data/processed/des_yield_clean.csv
Output: reports/yield_anomaly/*.csv
Run   : python scripts/A4_yield_anomaly_regression.py [--baseline rolling] [--fast]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_label_lib import past_z

warnings.filterwarnings("ignore")
try:
    from project_config import SPLIT_YEAR
except ModuleNotFoundError:
    SPLIT_YEAR = 2019

IN_PATH = "data/processed/A_features_signals.csv"
DES_PATH = "data/processed/des_yield_clean.csv"
OUT_DIR = "reports/yield_anomaly"
MIN_HIST, MIN_TRAIN_YEARS, SEED, Z_CLIP = 5, 4, 42, 3.0


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def make_model(name, fast):
    n = 120 if fast else 300
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(0, 4, 12))
    if name == "rf":
        return RandomForestRegressor(n_estimators=n, min_samples_leaf=8, max_features=0.4, n_jobs=-1, random_state=SEED)
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=150 if fast else 250,
                                         min_samples_leaf=25, l2_regularization=2.0, random_state=SEED)


def prep(tr, va, cols):
    Xtr, Xva = tr[cols].copy(), va[cols].copy()
    med = Xtr.median().fillna(0.0)
    Xtr, Xva = Xtr.fillna(med), Xva.fillna(med)
    keep = [c for c in Xtr.columns if Xtr[c].std() > 1e-12]
    sc = StandardScaler().fit(Xtr[keep])
    return sc.transform(Xtr[keep]), sc.transform(Xva[keep])


def rolling(df, years, model, cols, fast):
    parts = []
    for v in years:
        tr = df[(df["Year"] < v) & df["yz"].notna()]
        va = df[(df["Year"] == v) & df["yz"].notna()]
        if len(tr) < 80 or len(va) == 0:
            continue
        if model == "zero":
            p = np.zeros(len(va))
        else:
            Xtr, Xva = prep(tr, va, cols)
            p = make_model(model, fast).fit(Xtr, np.clip(tr["yz"].to_numpy(), -Z_CLIP, Z_CLIP)).predict(Xva)
        parts.append(pd.DataFrame({"State": va["State"].values, "Year": v, "y": va["yz"].values, "p": p}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def metrics(r, zero_mse):
    mse = float(np.mean((r["y"] - r["p"]) ** 2))
    out = {"n": len(r), "RMSE": np.sqrt(mse), "skill_vs_zero": 1 - mse / zero_mse}
    out["spearman"] = spearmanr(r["p"], r["y"])[0] if r["p"].nunique() > 1 else np.nan
    sy = r.groupby(["State", "Year"])[["p", "y"]].mean()
    out["spearman_stateyear"] = spearmanr(sy["p"], sy["y"])[0] if len(sy) >= 10 and sy["p"].nunique() > 1 else np.nan
    loss = (r["y"] <= -1.0).astype(int)
    out["loss_rate"] = float(loss.mean())
    out["AUC_loss"] = roc_auc_score(loss, -r["p"]) if loss.nunique() == 2 and r["p"].nunique() > 1 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="rolling", choices=["rolling", "expanding", "trend"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--des", default=DES_PATH)
    a = ap.parse_args()

    df = pd.read_csv(IN_PATH)
    if "P_fallback" in df.columns:
        bad = sorted(df.loc[df["P_fallback"] == 1, "District"].unique())
        if bad:
            print(f"Excluding {len(bad)} non-wheat-peak district(s): {bad}")
        df = df[df["P_fallback"] == 0]
    df = df.reset_index(drop=True)
    des = pd.read_csv(a.des)[["District", "Year", "Yield"]]
    tmp = df[["State", "District", "Year"]].merge(des, on=["District", "Year"], how="left")
    tmp["logy"] = np.log(tmp["Yield"].where(tmp["Yield"] > 0))
    df["yz"] = past_z(tmp, "logy", a.baseline, MIN_HIST).values

    veg = [c for c in df.columns if c.startswith("veg_")]
    wx = [c for c in df.columns if c.startswith("wx_")]
    feats = veg + wx
    os.makedirs(OUT_DIR, exist_ok=True)
    lab = df[df["yz"].notna()]
    section("A4 - CONTINUOUS YIELD-ANOMALY TEST")
    print(f"labelled rows {len(lab)} | years {int(lab['Year'].min())}-{int(lab['Year'].max())} | "
          f"baseline {a.baseline} | split {SPLIT_YEAR} | features: veg {len(veg)}, wx {len(wx)}")

    # ---------------- PART 1: univariate scan ----------------
    section("PART 1 - UNIVARIATE SCAN (within-district Spearman with yield z)")
    tr, te = lab[lab["Year"] < SPLIT_YEAR], lab[lab["Year"] >= SPLIT_YEAR]
    mu = tr.groupby("District")[feats].mean()

    def centred(d):
        return d[feats] - mu.reindex(d["District"].values).to_numpy()

    ctr, cte = centred(tr), centred(te)
    rows = []
    for f in feats:
        a1, a2 = ctr[f].notna(), cte[f].notna()
        r1 = spearmanr(ctr.loc[a1, f], tr.loc[a1, "yz"])[0] if a1.sum() > 30 else np.nan
        r2 = spearmanr(cte.loc[a2, f], te.loc[a2, "yz"])[0] if a2.sum() > 30 else np.nan
        rows.append({"feature": f, "rho_train": r1, "rho_test": r2})
    sc = pd.DataFrame(rows)
    sc["same_sign"] = np.sign(sc["rho_train"]) == np.sign(sc["rho_test"])
    sc["consistent"] = sc["same_sign"] & (sc["rho_train"].abs() >= 0.10) & (sc["rho_test"].abs() >= 0.10)
    sc = sc.reindex(sc["rho_train"].abs().sort_values(ascending=False).index)
    sc.to_csv(os.path.join(OUT_DIR, "feature_scan.csv"), index=False)
    print(sc.head(15).round(3).to_string(index=False))
    print(f"\nfeatures with the same sign in train and test and |rho| >= 0.10 in both: "
          f"{int(sc['consistent'].sum())} of {len(sc)}")
    print("Reading: heat features should be NEGATIVE, rain / soil-moisture mixed, vegetation POSITIVE. "
          "If almost nothing is consistent, the features carry no stable yield information here.")

    # ---------------- PART 2: rolling regression ----------------
    section("PART 2 - ROLLING-ORIGIN REGRESSION (one year ahead)")
    years = sorted(lab["Year"].unique())
    first_val = years[MIN_TRAIN_YEARS]
    cv_years = [int(v) for v in years if first_val <= v < SPLIT_YEAR]
    test_years = [int(v) for v in years if v >= SPLIT_YEAR]
    print(f"CV years {cv_years[0]}..{cv_years[-1]} | test years {test_years}")

    fsets = {"veg": veg, "wx": wx, "all": feats}
    res = []
    for period, yrs in (("CV", cv_years), ("TEST", test_years)):
        zero = rolling(lab, yrs, "zero", feats, a.fast)
        if zero.empty:
            continue
        zmse = float(np.mean(zero["y"] ** 2))
        res.append({"period": period, "fset": "-", "model": "zero", **metrics(zero, zmse)})
        for fs, cols in fsets.items():
            for mn in ("ridge", "rf", "hgb"):
                r = rolling(lab, yrs, mn, cols, a.fast)
                if not r.empty:
                    res.append({"period": period, "fset": fs, "model": mn, **metrics(r, zmse)})
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(OUT_DIR, "regression_results.csv"), index=False)
    show = ["fset", "model", "n", "RMSE", "skill_vs_zero", "spearman", "spearman_stateyear", "loss_rate", "AUC_loss"]
    for period in ("CV", "TEST"):
        sub = res[res["period"] == period]
        if len(sub):
            print(f"\n{period}:")
            print(sub.sort_values("skill_vs_zero", ascending=False)[show].round(3).to_string(index=False))

    # ---------------- by-year view of the best CV config on TEST ----------------
    cvr = res[(res["period"] == "CV") & (res["model"] != "zero")]
    if len(cvr) and test_years:
        b = cvr.sort_values("skill_vs_zero", ascending=False).iloc[0]
        r = rolling(lab, test_years, b["model"], fsets[b["fset"]], a.fast)
        print(f"\nTEST by year for the best CV config ({b['model']} / {b['fset']}): mean predicted z vs mean actual yield z")
        by = r.groupby("Year").agg(n=("y", "size"), mean_pred=("p", "mean"), mean_actual=("y", "mean"))
        print(by.round(3).to_string())

    print("\nHow to read: skill_vs_zero > 0 means better than 'no information'. AUC_loss ~0.5 = cannot detect yield "
          "loss; >= 0.65 on TEST = usable damage detector. Compare CV and TEST: a gap means selection noise.")


if __name__ == "__main__":
    main()