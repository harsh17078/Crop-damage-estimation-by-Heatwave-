"""
B1 - SATELLITE-ONLY DETECTION OF HEAT-STRESS EXPOSURE   (v2: one command runs everything)
=========================================================================================

Question  : can MODIS observations alone (land surface temperature, vegetation, moisture) tell whether the
            wheat grain-filling period of a district-year was heat-stressed?
Label     : from ERA5-Land daily Tmax.  HSk = 1 if the district had >= k days with Tmax >= 35 C in the window
            P .. P+40  (P = the district's NDVI-peak anchor from A1).  HS3 is the headline label;
            HS2 / HS5 (and HS1) are sensitivity labels.
Features  : MODIS only (sat_*).  No ERA5 variable is a feature.
Models    : LR, SVM, RF, XGBoost via A3 (in-fold threshold on balanced accuracy, no-skill flag).
FROZEN production model (pre-specified, not chosen from the results):  LR on the 8 LST features, raw values.
            The CV grid is still run, but only as an ablation / model-selection-instability check.

HONEST SCOPE: this detects heat EXPOSURE from space. It does not measure or predict crop damage or yield loss
(L2 / A6 / A8 found no detectable link between such exposure indices and yield anomalies).

----------------------------------------------------------------------------------------------------
HOW TO RUN  (from the project root, or from anywhere - the script changes to the project root itself)
----------------------------------------------------------------------------------------------------
  python scripts/B1_heat_exposure_classifier.py --check          # 5-second pre-flight: are all inputs there?
  python scripts/B1_heat_exposure_classifier.py                  # HS3: table + CV + test + LOSO + B2 validation
  python scripts/B1_heat_exposure_classifier.py --all            # HS3 + HS2 + HS5, each with B2 validation
  python scripts/B1_heat_exposure_classifier.py --all --fast     # quicker (fewer trees) for a dry run
  python scripts/B1_heat_exposure_classifier.py --reuse-table    # skip the (slow) table build, reuse the CSV
  python scripts/B1_heat_exposure_classifier.py --exclude Jhajjar   # sensitivity: drop the watch-list district
  python scripts/B1_heat_exposure_classifier.py --no-validate    # B1 only (no B2)
Other options: --label HS2 | --select auc|bal_acc|mcc | --n-boot 2000 | --no-loso

Needs in scripts/: A1_build_features_and_signals.py, A3_train_classifiers.py, proxy_label_lib.py, B2_validation.py
Input : data/raw/gee_ts/ts_era5_YYYY.csv, ts_modis_YYYY.csv (A0), data/processed/A_phenology_reference.csv (A1)
Output: reports/exposure/exposure_table.csv, exposure_table_meta.json, summary_all_labels.csv
        reports/exposure/<LABEL>/  cv_results.csv, test_results.csv, test_predictions.csv, test_by_year.csv,
                                   loso_results.csv, feature_importance.csv/.png, run_B1.log, B2 tables
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "data")) and os.path.abspath(os.getcwd()) != ROOT:
    os.chdir(ROOT)                      # A1 / A3 use paths relative to the project root
sys.path.insert(0, HERE)

try:
    import A1_build_features_and_signals as a1
    import A3_train_classifiers as a3
except ImportError as e:
    sys.exit(f"[ERROR] cannot import A1/A3 ({e}).\nPut this file in the same scripts/ folder as "
             "A1_build_features_and_signals.py, A3_train_classifiers.py and proxy_label_lib.py.")

warnings.filterwarnings("ignore")

PHENO_REF = "data/processed/A_phenology_reference.csv"
OUT_DIR = "reports/exposure"
TABLE = os.path.join(OUT_DIR, "exposure_table.csv")
META = os.path.join(OUT_DIR, "exposure_table_meta.json")
B2_PATH = os.path.join(HERE, "B2_validation.py")
WINDOW = (0, 40)               # days relative to the anchor P (grain-filling period)
HOT_C = 35.0
MIN_TRAIN_YEARS = 5
LST_SHIFT = 4                  # composite-centre shift of the 8-day LST product (same as A1 MODIS_VARS)
FROZEN = ("lr", "lst", False)  # (model, feature set, district_z): the pre-specified production model
ALL_LABELS = ["HS3", "HS2", "HS5"]


# ------------------------------------------------------------------ utilities
class Tee:
    """write to the console and to a log file at the same time (no need for `| tee`)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def preflight(a):
    section("PRE-FLIGHT")
    raw = getattr(a1, "RAW_DIR", "data/raw/gee_ts")
    problems, warns = [], []
    for kind in ("era5", "modis"):
        n = len(glob.glob(os.path.join(raw, f"ts_{kind}_*.csv")))
        print(f"  ts_{kind}_*.csv files in {raw}: {n}")
        if n == 0:
            problems.append(f"no ts_{kind}_YYYY.csv in {raw} -> run the A0 Earth Engine export first")
    if not os.path.exists(PHENO_REF):
        problems.append(f"{PHENO_REF} missing -> run: python scripts/A1_build_features_and_signals.py")
    else:
        print(f"  phenology reference: {PHENO_REF}")
    if not a.no_validate and not os.path.exists(B2_PATH):
        warns.append(f"{B2_PATH} not found -> B2 validation will be skipped (copy B2_validation.py into scripts/)")
    if a.reuse_table and not os.path.exists(TABLE):
        problems.append(f"--reuse-table given but {TABLE} does not exist -> run once without it")
    print(f"  working directory: {os.getcwd()}")
    print(f"  split year (a3.SPLIT_YEAR): {a3.SPLIT_YEAR}  | xgboost available: {getattr(a3, 'HAVE_XGB', '?')}")
    for w in warns:
        print("  [WARNING]", w)
    if problems:
        for p in problems:
            print("  [ERROR]", p)
        sys.exit(1)
    print("  all inputs found.")


# ------------------------------------------------------------------ table
def build_table():
    era5 = a1.read_ts("era5")
    modis = a1.quality_filter(a1.read_ts("modis"))
    series = a1.build_modis_series(modis)
    ph = pd.read_csv(PHENO_REF)
    if "P_fallback" in ph.columns:
        fb = ph["P_fallback"].astype(str).str.lower().isin(["true", "1", "1.0"])
        dropped = sorted(ph.loc[fb, "District"])
        if dropped:
            print(f"districts excluded because the NDVI peak is not wheat-like (P_fallback): {dropped}")
        ph = ph[~fb]
    Pmap = ph.set_index("District")["P_doy"].to_dict()
    era5["tmax"] = era5["temperature_2m_max"] - 273.15

    # real (non-interpolated) LST composites per district-year: A1 interpolates every series to daily values,
    # so 'p90 over 41 days' is really built from only ~5 composites. Keep the count as a diagnostic.
    lst_obs = {}
    lraw = modis[modis["source"] == "lst"]
    col = a1.pick(lraw, ["LST_Day_1km"])
    if col is not None:
        lraw = lraw[lraw[col].notna()]
        for (d, y), g in lraw.groupby(["District", "HYear"]):
            lst_obs[(d, y)] = pd.DatetimeIndex(pd.to_datetime(g["date"]) + pd.Timedelta(days=LST_SHIFT))

    def win(name, d, y, P, lo, hi):
        s = series.get((d, y, name))
        if s is None:
            return None
        w = s.loc[a1.date_of(y, P + lo):a1.date_of(y, P + hi)]
        return w if len(w) and w.notna().mean() >= a1.MIN_COVERAGE else None

    rows, skip_P, skip_wx = [], 0, 0
    for (d, y), g in era5.groupby(["District", "HYear"]):
        if d not in Pmap:
            skip_P += 1
            continue
        P = int(Pmap[d])
        w = g.sort_values("date").set_index("date")["tmax"].loc[a1.date_of(y, P + WINDOW[0]):a1.date_of(y, P + WINDOW[1])]
        if len(w) < 30 or w.isna().mean() > 0.2:
            skip_wx += 1
            continue
        n35 = int((w >= HOT_C).sum())
        r = {"State": g["State"].iloc[0], "District": d, "Year": int(y), "P_doy": P, "n35": n35,
             "hdd35": float((w - HOT_C).clip(lower=0).sum()), "tmax_mean_W": float(w.mean())}
        for k in (1, 2, 3, 5):
            r[f"HS{k}"] = int(n35 >= k)
        ld, ln = win("LST_day", d, y, P, *WINDOW), win("LST_night", d, y, P, *WINDOW)
        lp = win("LST_day", d, y, P, -45, -15)
        nd, npre = win("NDVI", d, y, P, *WINDOW), win("NDVI", d, y, P, -15, -1)
        nm, ev = win("NDMI", d, y, P, *WINDOW), win("EVI", d, y, P, *WINDOW)
        nan = np.nan
        r["sat_lst_day_mean"] = ld.mean() if ld is not None else nan
        r["sat_lst_day_max"] = ld.max() if ld is not None else nan
        r["sat_lst_day_p90"] = ld.quantile(0.9) if ld is not None else nan
        r["sat_lst_day_std"] = ld.std() if ld is not None else nan
        r["sat_lst_night_mean"] = ln.mean() if ln is not None else nan
        r["sat_lst_dtr"] = (ld - ln).mean() if (ld is not None and ln is not None) else nan
        r["sat_lst_day_pre"] = lp.mean() if lp is not None else nan
        r["sat_lst_rise"] = r["sat_lst_day_mean"] - r["sat_lst_day_pre"]
        r["sat_ndvi_W"] = nd.mean() if nd is not None else nan
        r["sat_ndvi_change"] = (nd.mean() - npre.mean()) if (nd is not None and npre is not None) else nan
        r["sat_ndmi_W"] = nm.mean() if nm is not None else nan
        r["sat_evi_W"] = ev.mean() if ev is not None else nan
        idx = lst_obs.get((d, y))
        r["diag_lst_n_obs"] = int(((idx >= a1.date_of(y, P + WINDOW[0])) & (idx <= a1.date_of(y, P + WINDOW[1]))).sum()) \
            if idx is not None else 0
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["State", "District", "Year"]).reset_index(drop=True)
    print(f"district-years skipped: no anchor {skip_P}, ERA5 window incomplete {skip_wx}")
    low = (df["diag_lst_n_obs"] < 4).mean()
    print(f"real LST composites inside the 41-day window: median {df['diag_lst_n_obs'].median():.0f}, "
          f"share of rows with < 4: {low:.1%}  (features are built from interpolated daily series)")
    return df


def reference_scores(df, y, years):
    """no-fit reference points: district climatology, LST-mean and LST-p90 used directly as scores."""
    parts = []
    for v in years:
        tr = df[(df["Year"] < v) & y.notna()]
        va = df[(df["Year"] == v) & y.notna()]
        if len(tr) < 80 or len(va) == 0:
            continue
        ytr = y.loc[tr.index]
        freq = ytr.groupby(tr["District"]).mean()
        parts.append(pd.DataFrame({
            "y": y.loc[va.index].to_numpy().astype(int),
            "clim": va["District"].map(freq).fillna(ytr.mean()).to_numpy(),
            "lst_mean": va["sat_lst_day_mean"].fillna(tr["sat_lst_day_mean"].median()).to_numpy(),
            "lst_p90": va["sat_lst_day_p90"].fillna(tr["sat_lst_day_p90"].median()).to_numpy()}))
    if not parts:
        return {}
    r = pd.concat(parts)
    out = {}
    if r["y"].nunique() > 1:
        prev = r["y"].mean()
        for k in ("clim", "lst_mean", "lst_p90"):
            out[k] = {"ROC_AUC": round(roc_auc_score(r["y"], r[k]), 3),
                      "PR_lift": round(average_precision_score(r["y"], r[k]) / prev, 3)}
    return out


def by_year(pr):
    t = pr.groupby("Year").agg(n=("y", "size"), prevalence=("y", "mean"), pred_pos=("p", "mean"))
    t["recall"] = pr[pr["y"] == 1].groupby("Year")["p"].mean()
    t["FPR"] = pr[pr["y"] == 0].groupby("Year")["p"].mean()
    return t


# ------------------------------------------------------------------ one label
def run_label(df, label, a):
    out_dir = os.path.join(OUT_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    a3.OUT_DIR = out_dir
    a3.TUNE = "bal"
    logf = open(os.path.join(out_dir, "run_B1.log"), "w")
    old = sys.stdout
    sys.stdout = Tee(old, logf)
    try:
        return _run_label(df, label, a, out_dir)
    finally:
        sys.stdout = old
        logf.close()


def _run_label(df, label, a, out_dir):
    section(f"B1  |  ACTIVE LABEL = {label}  (>= {label[2:]} days with Tmax >= {HOT_C:.0f} C in P..P+40)")
    sat = [c for c in df.columns if c.startswith("sat_")]
    lst_cols = [c for c in sat if "lst" in c]
    fsets = {"lst": lst_cols, "all_sat": sat}
    print(f"rows {len(df)} | districts {df['District'].nunique()} | years {df['Year'].min()}-{df['Year'].max()} | "
          f"satellite features {len(sat)} (lst: {len(lst_cols)})")
    print("prevalence of all labels:", {k: round(df[k].mean(), 3) for k in ("HS1", "HS2", "HS3", "HS5")})
    print(f"{label} prevalence by state:", df.groupby("State")[label].mean().round(3).to_dict())
    print(f"{label} prevalence by year :", df.groupby("Year")[label].mean().round(2).to_dict())
    y = df[label].astype(float)
    if not 0.08 <= y.mean() <= 0.7:
        print(f"[WARNING] prevalence of {label} is {y.mean():.2f}; check window / threshold before trusting any metric.")

    years = sorted(df["Year"].unique())
    cv_years = [int(v) for v in years if years[MIN_TRAIN_YEARS] <= v < a3.SPLIT_YEAR]
    test_years = [int(v) for v in years if v >= a3.SPLIT_YEAR]
    print(f"CV years {cv_years[0]}..{cv_years[-1]} | test years {test_years}")

    # ---- CV grid (ablation + selection-instability check) ----
    section("CROSS-VALIDATION (rolling origin) - ablation grid; the production model is FROZEN, not chosen from this")
    ref = reference_scores(df, y, cv_years)
    print("reference points (no model fitting):", ref)
    rows = []
    for fs, cols in fsets.items():
        for dz in (False, True):
            for mn in ("lr", "svm", "rf", "xgb"):
                pr = a3.rolling(df, y, cv_years, mn, cols, dz, a.fast)
                if not pr.empty:
                    rows.append({"fset": fs, "district_z": dz, "model": mn, **a3.metrics(pr["y"], pr["s"], pr["p"])})
    cv = pd.DataFrame(rows)
    if cv.empty:
        print(f"[ERROR] no model could be fitted for {label}: too few positives/negatives in the training years "
              f"(prevalence {y.mean():.3f}). Check the label window, HOT_C and the ERA5 export; skipping this label.")
        return {"label": label, "note": "no model fitted"}
    cv.to_csv(os.path.join(out_dir, "cv_results.csv"), index=False)
    show = ["fset", "district_z", "model", "prevalence", "always_pos_F1", "F1", "precision", "recall", "bal_acc",
            "MCC", "ROC_AUC", "PR_lift", "skill"]
    sel = {"auc": "ROC_AUC", "bal_acc": "bal_acc", "mcc": "MCC"}[a.select]
    print(cv.sort_values(sel, ascending=False)[show].round(3).to_string(index=False))
    best = cv.sort_values(sel, ascending=False).iloc[0]
    best_cfg = (best["model"], best["fset"], bool(best["district_z"]))

    # ---- test ----
    section("TEST (one-year-ahead, refit each year): FROZEN production model vs CV-best (by " + a.select + ")")
    configs = {"FROZEN (headline)": FROZEN}
    if best_cfg != FROZEN:
        configs[f"CV-best ({a.select})"] = best_cfg
    else:
        print("(CV-best is identical to the frozen model)")
    out, preds = [], {}
    for name, (mn, fs, dz) in configs.items():
        pr = a3.rolling(df, y, test_years, mn, fsets[fs], dz, a.fast)
        if pr.empty:
            continue
        preds[name] = pr
        out.append({"config": f"{name}: {mn}/{fs}/dz={dz}", **a3.metrics(pr["y"], pr["s"], pr["p"])})
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(out_dir, "test_results.csv"), index=False)
    print(t[["config", "prevalence", "always_pos_F1", "F1", "precision", "recall", "bal_acc", "MCC", "ROC_AUC",
             "PR_lift", "skill"]].round(3).to_string(index=False))
    ref_t = reference_scores(df, y, test_years)
    print("reference points on the test years:", ref_t)
    summary = {"label": label, "prevalence_test": np.nan}
    if "FROZEN (headline)" in preds:
        pr_f = preds["FROZEN (headline)"]
        pr_f.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
        by = by_year(pr_f)
        by.to_csv(os.path.join(out_dir, "test_by_year.csv"))
        print("\nBy year (FROZEN model):")
        print(by.round(2).to_string())
        m = t.iloc[0]
        summary.update({"prevalence_test": m["prevalence"], "AUC": m["ROC_AUC"], "F1": m["F1"],
                        "always_pos_F1": m["always_pos_F1"], "bal_acc": m["bal_acc"], "MCC": m["MCC"],
                        "PR_lift": m["PR_lift"], "p90_ref_AUC": ref_t.get("lst_p90", {}).get("ROC_AUC", np.nan),
                        "CV_best": f"{best_cfg[0]}/{best_cfg[1]}/dz={best_cfg[2]}"})
        print("\nFEATURE IMPORTANCE (frozen model; permutation importance is split between correlated LST features)")
        try:
            a3.explain(df, y, fsets[FROZEN[1]], FROZEN[0], FROZEN[2], a.fast, f"{FROZEN[0]} {FROZEN[1]} {label}")
        except Exception as e:
            print(f"  (importance skipped: {type(e).__name__}: {e})")

    # ---- leave-one-state-out ----
    if not a.no_loso:
        section("LEAVE-ONE-STATE-OUT (train other states < SPLIT_YEAR, test held-out state >= SPLIT_YEAR)")
        rows = []
        for st in sorted(df["State"].unique()):
            tr = df[(df["State"] != st) & (df["Year"] < a3.SPLIT_YEAR) & y.notna()]
            va = df[(df["State"] == st) & (df["Year"] >= a3.SPLIT_YEAR) & y.notna()]
            if len(va) < 20 or y.loc[tr.index].sum() < 10:
                continue
            ytr = y.loc[tr.index].to_numpy().astype(int)
            s, p = a3.fit_predict(FROZEN[0], tr, va, ytr, fsets[FROZEN[1]], FROZEN[2], a.fast)
            rows.append({"held_out": st, **a3.metrics(y.loc[va.index].to_numpy().astype(int), s, p)})
        if rows:
            lo = pd.DataFrame(rows)
            lo.to_csv(os.path.join(out_dir, "loso_results.csv"), index=False)
            print(lo[["held_out", "n", "prevalence", "F1", "recall", "ROC_AUC", "PR_lift"]].round(3).to_string(index=False))
            print("(states share the same years and weather systems: this checks state-identity leakage, "
                  "not year-level generalisation)")
    return summary


def run_b2(label, a):
    if not os.path.exists(B2_PATH):
        print(f"[skip] B2 validation for {label}: {B2_PATH} not found")
        return
    section(f"B2 VALIDATION for {label}  (baselines, within-year skill, year-block bootstrap, verdict)")
    cmd = [sys.executable, B2_PATH, "--label", label, "--table", TABLE, "--n-boot", str(a.n_boot)]
    if a.exclude:
        cmd += ["--exclude"] + a.exclude
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    logf = open(os.path.join(OUT_DIR, label, "run_B2.log"), "w")
    old = sys.stdout
    sys.stdout = Tee(old, logf)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        if proc.returncode != 0:
            print(f"[ERROR] B2 exited with code {proc.returncode} for {label}")
    finally:
        sys.stdout = old
        logf.close()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="B1 heat-exposure classifier (run with no arguments for the headline HS3 run)")
    ap.add_argument("--label", default="HS3", choices=["HS1", "HS2", "HS3", "HS5"])
    ap.add_argument("--all", action="store_true", help="run HS3 + HS2 + HS5")
    ap.add_argument("--fast", action="store_true", help="fewer trees (dry run)")
    ap.add_argument("--reuse-table", action="store_true", help="skip the table build, reuse reports/exposure/exposure_table.csv")
    ap.add_argument("--no-loso", action="store_true")
    ap.add_argument("--no-validate", action="store_true", help="do not run B2_validation.py afterwards")
    ap.add_argument("--select", default="auc", choices=["auc", "bal_acc", "mcc"],
                    help="criterion for the CV-best ablation model (the headline model is frozen anyway)")
    ap.add_argument("--n-boot", type=int, default=1000, help="year-block bootstrap replicates in B2")
    ap.add_argument("--exclude", nargs="*", default=[], help="districts to leave out, e.g. --exclude Jhajjar")
    ap.add_argument("--check", action="store_true", help="only check that all inputs exist, then exit")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    preflight(a)
    if a.check:
        return
    t0 = time.time()

    if a.reuse_table:
        section("B1 - REUSING TABLE " + TABLE)
        df = pd.read_csv(TABLE)
    else:
        section("B1 - BUILDING THE TABLE (label from ERA5, features from MODIS only)")
        df = build_table()
        df.to_csv(TABLE, index=False)
        with open(META, "w") as f:
            json.dump({"split_year": int(a3.SPLIT_YEAR), "window": list(WINDOW), "hot_c": HOT_C,
                       "min_train_years": MIN_TRAIN_YEARS, "rows": len(df),
                       "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        print(f"saved {TABLE} and {META}")
    if a.exclude:
        print(f"Excluding districts from modelling: {a.exclude}")
        df = df[~df["District"].isin(a.exclude)]
    df = df.sort_values(["State", "District", "Year"]).reset_index(drop=True)

    labels = ALL_LABELS if a.all else [a.label]
    summaries = []
    for lab in labels:
        res = run_label(df, lab, a)
        summaries.append(res)
        if res.get("note"):
            print(f"[skip] B2 validation for {lab}: B1 produced no model")
        elif not a.no_validate:
            run_b2(lab, a)

    section("SUMMARY (frozen production model on the test years)")
    sm = pd.DataFrame(summaries)
    sm.to_csv(os.path.join(OUT_DIR, "summary_all_labels.csv"), index=False)
    print(sm.round(3).to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Logs and tables: {OUT_DIR}/<LABEL>/")
    print("Read next: the B2 verdict (section 6 of run_B2.log). If it says FAIL, report B1 as a calibrated LST p90 "
          "threshold for exposure detection, freeze it, and stop adding variants.")


if __name__ == "__main__":
    main()