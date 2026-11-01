"""
B2 - VALIDATION OF THE HEAT-EXPOSURE CLASSIFIER (reads the table that B1 already writes)
========================================================================================
Input : reports/exposure/exposure_table.csv   (written by B1_heat_exposure_classifier.py)
Output: reports/exposure/B2/<LABEL>/*.csv  + a printed PASS/FAIL verdict

What it does (all rolling-origin or leave-one-year-out, nothing fitted on the evaluation year):
  1. Proper baselines: raw LST p90 threshold, LR on p90 alone, LR on p90 + district climatology,
     LST-mean threshold (the old B1 reference), climatology, and the production model (LR on 8 LST features).
  2. Skill decomposition: pooled AUC vs MEAN WITHIN-YEAR AUC vs year-level correlation vs within-district rho.
  3. Year-block bootstrap (resamples YEARS, not rows) for AUC / balanced accuracy / MCC and for the
     production-minus-baseline differences.
  4. Leave-one-year-out over all years (detection check) next to the conservative rolling origin.
  5. Continuous target (hot-day count) so the 35 C / 3-day threshold is not part of the test.
  6. False-alarm audit (are false positives near-misses with 1-2 hot days?) and a NaN audit per year
     (is the 2024 failure of all_sat a fog/missing-data problem?).

Pre-registered success criterion (edit BEFORE running, then do not change):
  PASS  if  (a) the year-bootstrap 95% CI of [AUC(production) - AUC(raw p90 threshold)] excludes zero (lower > 0)
        and (b) mean within-year AUC of the production model > WITHIN_YEAR_AUC_MIN.
  FAIL  otherwise -> report B1 honestly as "a calibrated LST p90 threshold" and freeze it.

Run   : python scripts/B2_validation.py --label HS3
        python scripts/B2_validation.py --selftest          (synthetic data, checks the script works)
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------- pre-registered settings -----------------------------
TABLE = "reports/exposure/exposure_table.csv"
OUT_BASE = "reports/exposure/B2"
SPLIT_YEAR = 2019            # must equal a3.SPLIT_YEAR used in B1
MIN_TRAIN_YEARS = 5          # same as B1
MIN_TRAIN_ROWS = 80
MIN_CLASS_PER_YEAR = 3       # a year needs >=3 positives and >=3 negatives to get a within-year AUC
WITHIN_YEAR_AUC_MIN = 0.65
N_BOOT = 1000
SEED = 42
PRODUCTION = "lr_lst"
BASELINE = "thr_p90"
# -------------------------------------------------------------------------------------


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------- model pieces ----------------------------------
def best_thr(y, s):
    """threshold maximising balanced accuracy on the training data (same idea as B1's TUNE='bal')."""
    y = np.asarray(y).astype(int)
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return np.median(s)
    best, bt = -1.0, np.median(s)
    for t in np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97))):
        pred = s >= t
        ba = (((pred) & (y == 1)).sum() / P + ((~pred) & (y == 0)).sum() / N) / 2
        if ba > best:
            best, bt = ba, t
    return bt


def clim_feature(tr, ytr, va):
    """district training frequency; for training rows it excludes the row itself (no label leakage)."""
    d = tr["District"].to_numpy()
    agg = pd.Series(ytr).groupby(d).agg(["sum", "count"])
    sum_d = pd.Series(d).map(agg["sum"]).to_numpy()
    n_d = pd.Series(d).map(agg["count"]).to_numpy()
    tr_c = np.where(n_d > 1, (sum_d - ytr) / np.maximum(n_d - 1, 1), ytr.mean())
    freq = agg["sum"] / agg["count"]
    va_c = va["District"].map(freq).fillna(ytr.mean()).to_numpy()
    return tr_c, va_c


def fit_score(kind, cols, tr, va, ytr):
    """returns (score on va, threshold chosen on training data only)."""
    if kind == "clim":
        s_tr, s_va = clim_feature(tr, ytr, va)
        return s_va, best_thr(ytr, s_tr)
    if kind == "thr":
        med = tr[cols[0]].median()
        s_tr, s_va = tr[cols[0]].fillna(med).to_numpy(), va[cols[0]].fillna(med).to_numpy()
        return s_va, best_thr(ytr, s_tr)
    Xtr, Xva = tr[cols].to_numpy(float), va[cols].to_numpy(float)
    if kind == "lr_clim":
        c_tr, c_va = clim_feature(tr, ytr, va)
        Xtr, Xva = np.column_stack([Xtr, c_tr]), np.column_stack([Xva, c_va])
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], best_thr(ytr, m.predict_proba(Xtr)[:, 1])


def make_specs(df):
    sat = [c for c in df.columns if c.startswith("sat_")]
    lst = [c for c in sat if "lst" in c]
    return {
        "clim": ("clim", []),
        "thr_mean": ("thr", ["sat_lst_day_mean"]),   # the old B1 'single LST' reference
        "thr_p90": ("thr", ["sat_lst_day_p90"]),     # the fair baseline
        "lr_p90": ("lr", ["sat_lst_day_p90"]),
        "lr_p90_clim": ("lr_clim", ["sat_lst_day_p90"]),
        "lr_lst": ("lr", lst),                       # production candidate
        "lr_all_sat": ("lr", sat),                   # ablation
    }


def run(df, y, eval_years, scheme, specs):
    out = []
    for v in eval_years:
        tr = df[(df["Year"] < v) if scheme == "rolling" else (df["Year"] != v)]
        tr = tr[y.loc[tr.index].notna()]
        va = df[df["Year"] == v]
        if len(tr) < MIN_TRAIN_ROWS or len(va) == 0:
            continue
        ytr = y.loc[tr.index].to_numpy().astype(int)
        if ytr.min() == ytr.max():
            continue
        for name, (kind, cols) in specs.items():
            s, thr = fit_score(kind, cols, tr, va, ytr)
            out.append(pd.DataFrame({
                "model": name, "Year": int(v), "District": va["District"].to_numpy(),
                "State": va["State"].to_numpy(), "y": y.loc[va.index].to_numpy().astype(int),
                "s": s, "p": (s >= thr).astype(int), "n35": va["n35"].to_numpy(),
                "hdd35": va["hdd35"].to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------ metrics ------------------------------------
def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan


def safe_rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return spearmanr(a, b)[0]


def summarize(d):
    y, s, p = d["y"].to_numpy(), d["s"].to_numpy(), d["p"].to_numpy()
    prev = y.mean()
    o = {"n": len(d), "prevalence": prev, "AUC": auc(y, s),
         "PR_lift": average_precision_score(y, s) / prev if 0 < prev < 1 else np.nan,
         "F1": f1_score(y, p, zero_division=0), "bal_acc": balanced_accuracy_score(y, p),
         "MCC": matthews_corrcoef(y, p), "recall": p[y == 1].mean() if y.sum() else np.nan,
         "FPR": p[y == 0].mean() if (y == 0).sum() else np.nan}
    wy = []
    for _, g in d.groupby("Year"):
        npos = g["y"].sum()
        if npos >= MIN_CLASS_PER_YEAR and len(g) - npos >= MIN_CLASS_PER_YEAR:
            wy.append(auc(g["y"], g["s"]))
    o["within_year_AUC"] = np.mean(wy) if wy else np.nan
    o["n_within_years"] = len(wy)
    yr = d.groupby("Year").agg(ms=("s", "mean"), pv=("y", "mean"))
    o["year_level_r"] = pearsonr(yr["ms"], yr["pv"])[0] if len(yr) >= 4 and yr["ms"].std() > 0 and yr["pv"].std() > 0 else np.nan
    s_a = d["s"] - d.groupby("District")["s"].transform("mean")
    y_a = d["y"] - d.groupby("District")["y"].transform("mean")
    o["within_district_rho"] = safe_rho(s_a, y_a)
    return o


def summary_table(pred):
    return pd.DataFrame([{"model": m, **summarize(g)} for m, g in pred.groupby("model", sort=False)])


# ------------------------------------ bootstrap ------------------------------------
def year_bootstrap(pred, models, n_boot, seed):
    rng = np.random.default_rng(seed)
    years = sorted(pred["Year"].unique())
    pm = {m: {yr: g for yr, g in pred[pred["model"] == m].groupby("Year")} for m in models}
    stats = {m: {"AUC": [], "bal_acc": [], "MCC": []} for m in models}
    for _ in range(n_boot):
        ys = rng.choice(years, len(years), replace=True)
        ok = True
        cur = {}
        for m in models:
            d = pd.concat([pm[m][yr] for yr in ys])
            if d["y"].nunique() < 2:
                ok = False
                break
            cur[m] = (roc_auc_score(d["y"], d["s"]), balanced_accuracy_score(d["y"], d["p"]),
                      matthews_corrcoef(d["y"], d["p"]))
        if ok:
            for m in models:
                for k, v in zip(("AUC", "bal_acc", "MCC"), cur[m]):
                    stats[m][k].append(v)
    rows = []
    cols = ["model", "metric", "mean", "lo95", "hi95", "n_boot_ok"]
    if len(stats[models[0]]["AUC"]) == 0:
        print("  (bootstrap skipped: resampled years never contained both classes - too few positives/negatives)")
        return pd.DataFrame(columns=cols)
    for m in models:
        for k in ("AUC", "bal_acc", "MCC"):
            a = np.array(stats[m][k])
            rows.append({"model": m, "metric": k, "mean": a.mean(), "lo95": np.percentile(a, 2.5),
                         "hi95": np.percentile(a, 97.5), "n_boot_ok": len(a)})
        if m != BASELINE:
            for k in ("AUC", "bal_acc", "MCC"):
                diff = np.array(stats[m][k]) - np.array(stats[BASELINE][k])
                rows.append({"model": f"{m} minus {BASELINE}", "metric": k, "mean": diff.mean(),
                             "lo95": np.percentile(diff, 2.5), "hi95": np.percentile(diff, 97.5),
                             "n_boot_ok": len(diff)})
    return pd.DataFrame(rows, columns=cols)


# ------------------------------- continuous target -------------------------------
def continuous_test(df, eval_years, feature_sets, target="n35"):
    rows = []
    for fname, cols in feature_sets.items():
        recs = []
        for v in eval_years:
            tr, va = df[df["Year"] < v], df[df["Year"] == v]
            if len(tr) < MIN_TRAIN_ROWS or len(va) == 0:
                continue
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
            m.fit(tr[cols].to_numpy(float), np.log1p(tr[target]))
            recs.append(pd.DataFrame({"Year": v, "obs": va[target].to_numpy(),
                                      "pred": m.predict(va[cols].to_numpy(float))}))
        d = pd.concat(recs)
        wy = [safe_rho(g["pred"], g["obs"]) for _, g in d.groupby("Year") if g["obs"].nunique() > 1]
        yr = d.groupby("Year").mean()
        rows.append({"target": target, "features": fname, "pooled_rho": safe_rho(d["pred"], d["obs"]),
                     "mean_within_year_rho": np.nanmean(wy) if wy else np.nan, "n_within_years": len(wy),
                     "year_level_rho": safe_rho(yr["pred"], yr["obs"])})
    return pd.DataFrame(rows)


# ------------------------------------ audits ------------------------------------
def false_alarm_audit(pred, model, K):
    d = pred[pred["model"] == model]
    fp = d[(d["y"] == 0) & (d["p"] == 1)]
    tn = d[(d["y"] == 0) & (d["p"] == 0)]
    print(f"false alarms: {len(fp)} of {int((d['y'] == 0).sum())} negatives ({len(fp) / max(1, (d['y'] == 0).sum()):.1%})")
    if len(fp):
        near = ((fp["n35"] >= max(1, K - 2)) & (fp["n35"] < K)).mean()
        print(f"  share of false alarms that were NEAR-MISSES (n35 between {max(1, K - 2)} and {K - 1}): {near:.1%}")
        print(f"  mean hot days: false alarms {fp['n35'].mean():.2f} vs true negatives {tn['n35'].mean():.2f}")
        print(f"  false alarms with n35 == 0: {(fp['n35'] == 0).mean():.1%}")
    by = d.groupby("Year").apply(lambda g: pd.Series({
        "n": len(g), "prevalence": g["y"].mean(), "mean_score": g["s"].mean(), "pred_pos": g["p"].mean(),
        "recall": g.loc[g["y"] == 1, "p"].mean() if g["y"].sum() else np.nan,
        "FPR": g.loc[g["y"] == 0, "p"].mean() if (g["y"] == 0).sum() else np.nan,
        "AUC_within": auc(g["y"], g["s"]), "mean_n35": g["n35"].mean()}))
    return by


def nan_audit(df):
    sat = [c for c in df.columns if c.startswith("sat_")]
    t = df.groupby("Year")[sat].apply(lambda g: g.isna().mean()).round(2)
    t["any_nan_rows"] = df.groupby("Year").apply(lambda g: g[sat].isna().any(axis=1).mean()).round(2)
    if "diag_lst_n_obs" in df.columns:   # real (non-interpolated) LST composites inside the window
        t["lst_real_composites_median"] = df.groupby("Year")["diag_lst_n_obs"].median()
    return t


# ------------------------------------- verdict -------------------------------------
def verdict(tag, summ, boot):
    s = summ.set_index("model")
    row = boot[(boot["model"] == f"{PRODUCTION} minus {BASELINE}") & (boot["metric"] == "AUC")]
    lo = row["lo95"].iloc[0] if len(row) else np.nan
    if lo != lo:
        print(f"[{tag}] bootstrap unavailable -> criterion (a) cannot be evaluated (counts as FAIL)")
    wy = s.loc[PRODUCTION, "within_year_AUC"]
    a = bool(lo > 0)
    b = bool(wy > WITHIN_YEAR_AUC_MIN)
    print(f"[{tag}] (a) AUC(prod) - AUC({BASELINE}) bootstrap lower bound = {lo:+.3f} -> {'ok' if a else 'FAIL'};  "
          f"(b) within-year AUC = {wy:.3f} (> {WITHIN_YEAR_AUC_MIN}) -> {'ok' if b else 'FAIL'}")
    return a, b


# ------------------------------------ selftest ------------------------------------
def make_synthetic(path, seed=0):
    rng = np.random.default_rng(seed)
    years, rows = range(2003, 2025), []
    hot = rng.normal(0, 1, len(years))
    for st, nd in (("Haryana", 12), ("Punjab", 12), ("Uttar Pradesh", 13)):
        for i in range(nd):
            deff = rng.normal(0, 0.6)
            for j, yr in enumerate(years):
                T = 33.0 + 2.0 * hot[j] + deff + rng.normal(0, 0.8)
                n35 = rng.poisson(np.exp(T - 33.5))
                lstp = 36 + 2.5 * (T - 33) + rng.normal(0, 1.5)
                r = {"State": st, "District": f"{st[:2]}_{i}", "Year": yr, "P_doy": 60, "n35": n35,
                     "hdd35": float(max(0, T - 35) * 3), "tmax_mean_W": T}
                for k in (1, 2, 3, 5):
                    r[f"HS{k}"] = int(n35 >= k)
                r.update({"sat_lst_day_mean": lstp - 3 + rng.normal(0, .5), "sat_lst_day_max": lstp + 1 + rng.normal(0, .5),
                          "sat_lst_day_p90": lstp, "sat_lst_day_std": abs(rng.normal(2, .5)),
                          "sat_lst_night_mean": 12 + 0.3 * (T - 33) + rng.normal(0, 1),
                          "sat_lst_dtr": 20 + rng.normal(0, 2), "sat_lst_day_pre": 30 + rng.normal(0, 2),
                          "sat_lst_rise": 5 + rng.normal(0, 2), "sat_ndvi_W": 0.5 + rng.normal(0, .05),
                          "sat_ndvi_change": rng.normal(0, .05), "sat_ndmi_W": rng.normal(0.2, .05),
                          "sat_evi_W": 0.3 + rng.normal(0, .03)})
                rows.append(r)
    df = pd.DataFrame(rows)
    df.loc[(df["Year"] == 2024) & (rng.random(len(df)) < 0.5), ["sat_ndmi_W", "sat_ndvi_W"]] = np.nan
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# -------------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="HS3", choices=["HS1", "HS2", "HS3", "HS5"])
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--split-year", type=int, default=None, help="default: read from exposure_table_meta.json (written by B1)")
    ap.add_argument("--exclude", nargs="*", default=[], help="districts to leave out, e.g. --exclude Jhajjar")
    a = ap.parse_args()
    global SPLIT_YEAR
    if a.selftest:
        a.table = "/tmp/b2_selftest/exposure_table.csv"
        make_synthetic(a.table)
        print("[selftest] synthetic table written to", a.table)
    out_dir = os.path.join("/tmp/b2_selftest/out" if a.selftest else OUT_BASE, a.label)
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(os.path.dirname(a.table), "exposure_table_meta.json")
    if a.split_year is not None:
        SPLIT_YEAR = a.split_year
    elif os.path.exists(meta_path):
        SPLIT_YEAR = int(json.load(open(meta_path))["split_year"])
    df = pd.read_csv(a.table)
    if a.exclude:
        print(f"Excluding districts: {a.exclude}")
        df = df[~df["District"].isin(a.exclude)]
    df = df.sort_values(["State", "District", "Year"]).reset_index(drop=True)
    need = {"District", "State", "Year", "n35", "hdd35", a.label, "sat_lst_day_p90", "sat_lst_day_mean"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"table is missing columns: {sorted(miss)} - rerun B1 first")
    y = df[a.label].astype(float)
    K = int(a.label[2:])
    years = sorted(df["Year"].unique())
    first_val = years[MIN_TRAIN_YEARS]
    specs = make_specs(df)
    print(f"LABEL = {a.label} (>= {K} days with Tmax >= 35 C) | rows {len(df)} | prevalence {y.mean():.3f} | "
          f"rolling years {first_val}..{years[-1]} | test years >= {SPLIT_YEAR}")

    section("0. NaN AUDIT - share of missing satellite values per year (look at 2024)")
    na = nan_audit(df)
    na.to_csv(os.path.join(out_dir, "nan_audit.csv"))
    print(na.loc[na["any_nan_rows"] > 0.05].to_string() if (na["any_nan_rows"] > 0.05).any() else "no year has >5% rows with missing features")

    roll = run(df, y, [v for v in years if v >= first_val], "rolling", specs)
    loyo = run(df, y, years, "loyo", specs)
    sets = {"rolling_test": roll[roll["Year"] >= SPLIT_YEAR], "rolling_all": roll, "loyo_all": loyo}

    show = ["model", "n", "prevalence", "AUC", "within_year_AUC", "n_within_years", "year_level_r",
            "within_district_rho", "PR_lift", "F1", "bal_acc", "MCC", "recall", "FPR"]
    section("1+2. BASELINES AND SKILL DECOMPOSITION")
    print("AUC = pooled; within_year_AUC = mean AUC over years with >=3 positives and >=3 negatives;\n"
          "year_level_r = corr(mean score, prevalence) across years; within_district_rho = Spearman after removing district means.")
    summaries = {}
    for tag, pr in sets.items():
        summ = summary_table(pr)
        summaries[tag] = summ
        summ.to_csv(os.path.join(out_dir, f"summary_{tag}.csv"), index=False)
        print(f"\n--- {tag} (years {sorted(pr['Year'].unique())[0]}..{sorted(pr['Year'].unique())[-1]}, "
              f"{pr['Year'].nunique()} years) ---")
        print(summ[show].round(3).to_string(index=False))

    section("3. YEAR-BLOCK BOOTSTRAP (resampling years) for the key models")
    keep = [BASELINE, "lr_p90", "lr_p90_clim", PRODUCTION, "lr_all_sat"]
    boots = {}
    for tag in ("rolling_all", "loyo_all", "rolling_test"):
        pr = sets[tag][sets[tag]["model"].isin(keep)]
        boots[tag] = year_bootstrap(pr, keep, a.n_boot, SEED)
        boots[tag].to_csv(os.path.join(out_dir, f"bootstrap_{tag}.csv"), index=False)
        print(f"\n--- {tag} ({pr['Year'].nunique()} years; with few years the CI is wide - that is the point) ---")
        print(boots[tag].round(3).to_string(index=False))

    section("4. CONTINUOUS TARGET (hot-day count, log1p) - no threshold involved")
    sat = [c for c in df.columns if c.startswith("sat_")]
    lst = [c for c in sat if "lst" in c]
    cont = pd.concat([continuous_test(df, [v for v in years if v >= first_val],
                                      {"p90_only": ["sat_lst_day_p90"], "lst8": lst, "all_sat": sat}, t)
                      for t in ("n35", "hdd35")])
    cont.to_csv(os.path.join(out_dir, "continuous_target.csv"), index=False)
    print(cont.round(3).to_string(index=False))

    section(f"5. FALSE-ALARM AND PER-YEAR AUDIT for {PRODUCTION} (rolling origin, test years)")
    by = false_alarm_audit(sets["rolling_test"], PRODUCTION, K)
    by.to_csv(os.path.join(out_dir, "per_year_test.csv"))
    print(by.round(2).to_string())
    print(f"\nsame audit, all rolling years:")
    by_all = false_alarm_audit(sets["rolling_all"], PRODUCTION, K)
    by_all.to_csv(os.path.join(out_dir, "per_year_rolling_all.csv"))
    print("\nper-year, all_sat ablation (test years):")
    print(false_alarm_audit(sets["rolling_test"], "lr_all_sat", K)[["n", "prevalence", "pred_pos", "recall", "FPR"]].round(2).to_string())

    section("6. VERDICT against the pre-registered criterion")
    a_roll, b_roll = verdict("rolling_all", summaries["rolling_all"], boots["rolling_all"])
    a_loyo, b_loyo = verdict("loyo_all", summaries["loyo_all"], boots["loyo_all"])
    print()
    if a_roll and b_roll:
        print("PASS (conservative rolling origin): the full LST model adds skill beyond the p90 threshold, "
              "and it ranks districts within a year.")
    else:
        if not a_roll:
            print("(a) FAILED: the 8-feature model does not beat a raw LST p90 threshold. Report B1 as a calibrated "
                  "LST p90 threshold for exposure detection (use lr_p90 as the headline).")
        if not b_roll:
            print("(b) FAILED: skill within a year is weak, so most of the pooled AUC is hot-year vs cool-year "
                  "detection. Say this explicitly in the report; do not describe it as district-level mapping skill.")
        print("Freeze B1 now and stop adding variants (forking paths).")
    if (a_loyo and b_loyo) and not (a_roll and b_roll):
        print("(LOYO passes but rolling origin does not: the gain depends on future years being in training - "
              "do not headline it.)")
    print(f"\nAll tables saved in {out_dir}")


if __name__ == "__main__":
    main()