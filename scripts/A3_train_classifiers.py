"""
OPTION A — STEP A3: CLASSIFIERS ON THE PROXY LABEL (LR / SVM / RF / XGBoost)
============================================================================

Task  : predict the damage-like decline label  y = 1  if  z(post-peak signal) <= threshold,
        from weather EXPOSURE + pre-peak vegetation STATE (see A1 for the leakage rule).
Labels: past-only z-scores (proxy_label_lib) -> labels of earlier years never depend on later
        years, so plain rolling-origin validation is leakage-free.

Validation (expanding window, one year ahead)
---------------------------------------------
    CV     : validation years  first_val .. SPLIT_YEAR-1   (model / feature-set choice)
    TEST   : years >= SPLIT_YEAR, refit every year on all earlier years (untouched by choices)
In every fold: imputation, scaling, (optional) district normalisation, the decision threshold
(maximising F1 on out-of-fold training scores, grouped by year) and the model are fitted on
training years only.

Reported (pooled over folds): F1, precision, recall, balanced accuracy, MCC, ROC-AUC, PR-AUC,
and the reference values  prevalence, always-positive F1, PR-AUC lift.

!! F1 / recall are NOT comparable across thresholds (prevalence changes). Use ROC-AUC, MCC and
   PR-AUC lift to compare thresholds; use F1 / recall to compare models WITHIN a threshold.

Feature sets : veg (pre-peak vegetation state) | wx (weather exposure) | all
Models       : lr | svm | rf | xgb (falls back to sklearn HistGradientBoosting if xgboost is absent)

Examples
--------
    python A3_train_classifiers.py                       # label from recommended_label.json
    python A3_train_classifiers.py --sweep               # compare all 4 thresholds
    python A3_train_classifiers.py --label des --baseline rolling --sweep    # DIRECT test: DES yield-loss label
    python A3_train_classifiers.py --signal S_ndvi --baseline expanding --threshold -1.0 --loso
Outputs in reports/proxy_labels/: classifier_cv_results.csv, classifier_test_results.csv,
test_by_year.csv, loso_results.csv, feature_importance.csv/.png
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_label_lib import label_from_z, past_z

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

warnings.filterwarnings("ignore")

try:
    from project_config import SPLIT_YEAR
except ModuleNotFoundError:
    SPLIT_YEAR = 2019

IN_PATH = "data/processed/A_features_signals.csv"
OUT_DIR = "reports/proxy_labels"
REC_PATH = os.path.join(OUT_DIR, "recommended_label.json")
SWEEP = [-0.5, -1.0, -1.5, -2.0]
MIN_HIST = 5               # years of history before a z-score exists
MIN_TRAIN_YEARS = 4        # labelled years needed before the first validation year
SEED = 42
TUNE = "bal"               # decision-threshold criterion: bal (balanced accuracy) | mcc | f1


# ======================================================================
# models
# ======================================================================
def make_model(name, pos_weight, fast):
    n = 120 if fast else 300
    if name == "lr":
        return LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000)
    if name == "svm":
        return SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(n_estimators=n, min_samples_leaf=5, max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    if name == "xgb":
        if HAVE_XGB:
            return XGBClassifier(n_estimators=n, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.7, min_child_weight=3, reg_lambda=2.0,
                                 scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=-1,
                                 random_state=SEED, verbosity=0)
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=n,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              class_weight="balanced", random_state=SEED)
    raise ValueError(name)


def score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def default_thr(name):
    return 0.0 if name == "svm" else 0.5


# ======================================================================
# fold preparation
# ======================================================================
def district_z(Xtr, Xva, d_tr, d_va):
    mu, sd = Xtr.groupby(d_tr.values).mean(), Xtr.groupby(d_tr.values).std()
    g_mu, g_sd = Xtr.mean(), Xtr.std().replace(0, np.nan).fillna(1.0)

    def ap(X, d):
        m = np.where(np.isnan(mu.reindex(d.values).to_numpy()), g_mu.to_numpy(), mu.reindex(d.values).to_numpy())
        s = np.where(np.isnan(sd.reindex(d.values).to_numpy()), g_sd.to_numpy(), sd.reindex(d.values).to_numpy())
        s = np.maximum(s, 0.25 * g_sd.to_numpy())
        return pd.DataFrame((X.to_numpy() - m) / s, index=X.index, columns=X.columns)

    return ap(Xtr, d_tr), ap(Xva, d_va)


def prepare(train, val, cols, use_dz):
    Xtr, Xva = train[cols].copy(), val[cols].copy()
    if use_dz:
        Xtr, Xva = district_z(Xtr, Xva, train["District"], val["District"])
    med = Xtr.median().fillna(0.0)
    Xtr, Xva = Xtr.fillna(med), Xva.fillna(med)
    keep = [c for c in Xtr.columns if Xtr[c].std() > 1e-12]
    sc = StandardScaler().fit(Xtr[keep])
    return sc.transform(Xtr[keep]), sc.transform(Xva[keep]), keep


def tune_threshold(name, Xtr, ytr, gtr, pos_weight, fast):
    """Decision threshold maximising TUNE (default balanced accuracy) on out-of-fold TRAINING scores."""
    dthr = default_thr(name)
    if ytr.sum() < 8 or (len(ytr) - ytr.sum()) < 8 or len(np.unique(gtr)) < 3:
        return dthr
    k = min(5, len(np.unique(gtr)))
    oof = np.full(len(ytr), np.nan)
    for a, b in GroupKFold(k).split(Xtr, ytr, gtr):
        if ytr[a].sum() == 0 or ytr[a].sum() == len(a):
            continue
        m = make_model(name, pos_weight, fast).fit(Xtr[a], ytr[a])
        oof[b] = score(m, Xtr[b])
    ok = ~np.isnan(oof)
    if ok.sum() < 30:
        return dthr
    cands = np.unique(np.quantile(oof[ok], np.linspace(0.2, 0.97, 45)))

    def crit(c):
        pr = oof[ok] >= c
        if TUNE == "f1":      # WARNING: with a weak signal the F1-optimal rule is "predict everything positive"
            return f1_score(ytr[ok], pr, zero_division=0)
        if TUNE == "mcc":
            return matthews_corrcoef(ytr[ok], pr) if len(np.unique(pr)) > 1 else -1.0
        return balanced_accuracy_score(ytr[ok], pr)

    return float(cands[int(np.argmax([crit(c) for c in cands]))])


def fit_predict(name, train, val, ytr, cols, use_dz, fast):
    Xtr, Xva, keep = prepare(train, val, cols, use_dz)
    pw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    thr = tune_threshold(name, Xtr, ytr, train["Year"].to_numpy(), pw, fast)
    m = make_model(name, pw, fast).fit(Xtr, ytr)
    s = score(m, Xva)
    return s, (s >= thr).astype(int)


# ======================================================================
# metrics
# ======================================================================
def metrics(y, s, p):
    y = np.asarray(y).astype(int)
    prev = float(y.mean())
    out = {"n": len(y), "prevalence": prev, "always_pos_F1": 2 * prev / (1 + prev) if prev > 0 else np.nan,
           "F1": f1_score(y, p, zero_division=0), "precision": precision_score(y, p, zero_division=0),
           "recall": recall_score(y, p, zero_division=0)}
    out["bal_acc"] = balanced_accuracy_score(y, p) if len(np.unique(y)) > 1 else np.nan
    out["MCC"] = matthews_corrcoef(y, p) if len(np.unique(p)) > 1 and len(np.unique(y)) > 1 else 0.0
    if len(np.unique(y)) > 1:
        out["ROC_AUC"] = roc_auc_score(y, s)
        out["PR_AUC"] = average_precision_score(y, s)
        out["PR_lift"] = out["PR_AUC"] / prev
    else:
        out["ROC_AUC"] = out["PR_AUC"] = out["PR_lift"] = np.nan
    out["skill"] = ("NO SKILL" if (out["ROC_AUC"] == out["ROC_AUC"] and (out["ROC_AUC"] < 0.6 or out["PR_lift"] < 1.15))
                    else "ok")
    return out


# ======================================================================
# rolling origin
# ======================================================================
def rolling(df, y, years, name, cols, use_dz, fast):
    parts = []
    for v in years:
        tr = df[(df["Year"] < v) & y.notna()]
        va = df[(df["Year"] == v) & y.notna()]
        ytr = y.loc[tr.index].to_numpy().astype(int)
        if len(va) == 0 or len(tr) < 80 or ytr.sum() < 5 or (len(ytr) - ytr.sum()) < 5:
            continue
        s, p = fit_predict(name, tr, va, ytr, cols, use_dz, fast)
        parts.append(pd.DataFrame({"State": va["State"].values, "District": va["District"].values,
                                   "Year": v, "y": y.loc[va.index].to_numpy().astype(int), "s": s, "p": p}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def explain(df, y, cols, name, use_dz, fast, label):
    tr = df[(df["Year"] < SPLIT_YEAR) & y.notna()]
    te = df[(df["Year"] >= SPLIT_YEAR) & y.notna()]
    if len(te) < 20:
        te = tr.sample(min(300, len(tr)), random_state=SEED)
    Xtr, Xte, keep = prepare(tr, te, cols, use_dz)
    ytr = y.loc[tr.index].to_numpy().astype(int)
    pw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    m = make_model(name, pw, fast).fit(Xtr, ytr)
    imp = None
    try:
        if name in ("rf", "xgb") and (name == "rf" or HAVE_XGB):
            import shap
            sv = shap.TreeExplainer(m).shap_values(Xte)
            sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
            imp = pd.DataFrame({"feature": keep, "importance": np.abs(sv).mean(0), "method": "mean|SHAP|"})
    except Exception as e:
        print(f"  (SHAP unavailable: {type(e).__name__}) -> permutation importance")
    if imp is None:
        yte = y.loc[te.index].to_numpy().astype(int)
        pi = permutation_importance(m, Xte, yte, n_repeats=8, random_state=SEED,
                                    scoring="average_precision" if len(np.unique(yte)) > 1 else None)
        imp = pd.DataFrame({"feature": keep, "importance": pi.importances_mean, "method": "permutation(AP)"})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(f"Top features - {label}\n({imp['method'].iloc[0]})")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=130); plt.close(fig)
    print(imp.head(12).round(4).to_string(index=False))


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal"); ap.add_argument("--baseline"); ap.add_argument("--threshold", type=float)
    ap.add_argument("--label", choices=["proxy", "des"], default="proxy",
                    help="proxy = satellite decline label (default); des = DES yield-loss label (past-only z of log yield)")
    ap.add_argument("--des", default="data/processed/des_yield_clean.csv", help="DES file for --label des")
    ap.add_argument("--sweep", action="store_true", help="compare all four thresholds")
    ap.add_argument("--fast", action="store_true"); ap.add_argument("--loso", action="store_true")
    ap.add_argument("--tune", choices=["bal", "mcc", "f1"], default="bal", help="threshold-tuning criterion")
    ap.add_argument("--district-z", action="store_true", help="district-normalise features (train stats)")
    ap.add_argument("--exclude", nargs="*", default=[], help="extra districts to leave out, e.g. --exclude Jhajjar")
    ap.add_argument("--keep-fallback", action="store_true", help="keep districts flagged P_fallback=1 (not wheat-like peak)")
    a = ap.parse_args()

    global TUNE
    TUNE = a.tune
    rec = {}
    if os.path.exists(REC_PATH):
        rec = json.load(open(REC_PATH))
    signal = a.signal or rec.get("signal", "S_ndvi")
    baseline = a.baseline or rec.get("baseline", "expanding")
    primary = a.threshold if a.threshold is not None else rec.get("threshold", -1.0)
    thresholds = SWEEP if a.sweep else [primary]
    if primary not in thresholds:
        thresholds = [primary] + thresholds

    df = pd.read_csv(IN_PATH)
    if not a.keep_fallback and "P_fallback" in df.columns and df["P_fallback"].sum() > 0:
        bad = sorted(df.loc[df["P_fallback"] == 1, "District"].unique())
        print(f"Excluding {len(bad)} district(s) whose NDVI peak is not wheat-like (mustard/gram/other): {bad}")
        df = df[df["P_fallback"] == 0]
    if a.exclude:
        print(f"Excluding by request: {a.exclude}")
        df = df[~df["District"].isin(a.exclude)]
    df = df.reset_index(drop=True)
    if a.label == "des":
        des = pd.read_csv(a.des)[["District", "Year", "Yield"]]
        tmp = df[["State", "District", "Year"]].merge(des, on=["District", "Year"], how="left")
        tmp["logy"] = np.log(tmp["Yield"].where(tmp["Yield"] > 0))
        z = past_z(tmp, "logy", baseline, MIN_HIST)
        signal = "DES_yield"
        print(f"LABEL = DES yield loss (past-only z of log yield, harvest-year key); matched rows: {int(tmp['Yield'].notna().sum())}")
    else:
        z = past_z(df, signal, baseline, MIN_HIST)
    veg = [c for c in df.columns if c.startswith("veg_")]
    wx = [c for c in df.columns if c.startswith("wx_")]
    fsets = {"veg": veg, "wx": wx, "all": veg + wx}
    # Signals whose definition overlaps the pre-peak vegetation features (S_dndvi contains veg_NDVI_v3;
    # S_peak / S_int cover P-20/P-30 onwards) would be partly "predicted" by arithmetic -> weather only.
    if signal in ("S_dndvi", "S_peak", "S_int"):
        print(f"NOTE: label {signal} overlaps the pre-peak vegetation window -> only the weather feature set (wx) is used.")
        fsets = {"wx": wx}
    models = ["lr", "svm", "rf", "xgb"]
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70 + f"\nA3 — CLASSIFIERS | label = {signal}/{baseline} | thresholds {thresholds}\n" + "=" * 70)
    print(f"rows {len(df)} | veg feats {len(veg)} | wx feats {len(wx)} | xgboost {HAVE_XGB} | split {SPLIT_YEAR}")

    cv_rows, best_cfg = [], {}
    for thr in thresholds:
        y = label_from_z(z, thr)
        lab_years = sorted(df.loc[y.notna(), "Year"].unique())
        if len(lab_years) <= MIN_TRAIN_YEARS:
            print(f"thr {thr}: too few labelled years"); continue
        first_val = lab_years[MIN_TRAIN_YEARS]
        cv_years = [v for v in lab_years if first_val <= v < SPLIT_YEAR]
        print(f"\n--- threshold z <= {thr}: labelled {int(y.notna().sum())} rows, positives {int(y.sum())} "
              f"({y.mean():.1%}) | CV years {cv_years[0] if cv_years else '-'}..{cv_years[-1] if cv_years else '-'}")
        for fs, cols in fsets.items():
            for mn in models:
                pr = rolling(df, y, cv_years, mn, cols, a.district_z, a.fast)
                if pr.empty:
                    continue
                m = metrics(pr["y"], pr["s"], pr["p"])
                cv_rows.append({"thr": thr, "fset": fs, "model": mn, **m})
        sub = pd.DataFrame([r for r in cv_rows if r["thr"] == thr])
        if len(sub):
            best_cfg[thr] = sub.sort_values(["F1", "recall"], ascending=False).iloc[0][["fset", "model"]].to_dict()

    cv = pd.DataFrame(cv_rows)
    cv.to_csv(os.path.join(OUT_DIR, "classifier_cv_results.csv"), index=False)
    show = ["thr", "fset", "model", "prevalence", "always_pos_F1", "F1", "precision", "recall",
            "bal_acc", "MCC", "ROC_AUC", "PR_lift", "skill"]
    print("\n" + "=" * 70 + "\nCV RESULTS (rolling origin, years < SPLIT_YEAR)\n" + "=" * 70)
    print(cv.sort_values(["thr", "F1"], ascending=[False, False])[show].round(3).to_string(index=False))

    # ---------------- test ----------------
    print("\n" + "=" * 70 + "\nHOLDOUT TEST (one-year-ahead, refit each year) - best CV config per threshold\n" + "=" * 70)
    test_rows, by_year = [], []
    for thr, cfg in best_cfg.items():
        y = label_from_z(z, thr)
        years = sorted(v for v in df.loc[y.notna(), "Year"].unique() if v >= SPLIT_YEAR)
        pr = rolling(df, y, years, cfg["model"], fsets[cfg["fset"]], a.district_z, a.fast)
        if pr.empty:
            continue
        test_rows.append({"thr": thr, **cfg, **metrics(pr["y"], pr["s"], pr["p"])})
        if thr == primary:
            for yr, g in pr.groupby("Year"):
                by_year.append({"Year": yr, "n": len(g), "prevalence": g["y"].mean(), "recall":
                                recall_score(g["y"], g["p"], zero_division=0), "precision":
                                precision_score(g["y"], g["p"], zero_division=0),
                                "pred_pos_rate": g["p"].mean()})
    te = pd.DataFrame(test_rows)
    te.to_csv(os.path.join(OUT_DIR, "classifier_test_results.csv"), index=False)
    if len(te):
        print(te[["thr", "fset", "model", "prevalence", "always_pos_F1", "F1", "precision", "recall",
                  "bal_acc", "MCC", "ROC_AUC", "PR_lift", "skill"]].round(3).to_string(index=False))
        if (te["skill"] == "NO SKILL").all():
            print("\n[NO SKILL] ROC-AUC < 0.6 or PR lift < 1.15 on the test years: F1/recall values above are not "
                  "meaningful (a model that predicts everything positive also gets a high recall).")
    if by_year:
        by = pd.DataFrame(by_year); by.to_csv(os.path.join(OUT_DIR, "test_by_year.csv"), index=False)
        print(f"\nTest by year at z <= {primary}:")
        print(by.round(3).to_string(index=False))

    # ---------------- explainability ----------------
    if primary in best_cfg:
        cfg = best_cfg[primary]
        print(f"\nFEATURE IMPORTANCE for {cfg['model']} / {cfg['fset']} at z <= {primary}")
        explain(df, label_from_z(z, primary), fsets[cfg["fset"]], cfg["model"], a.district_z, a.fast,
                f"{cfg['model']} {cfg['fset']} z<={primary}")

    # ---------------- leave-one-state-out (spatial transfer) ----------------
    if a.loso and primary in best_cfg:
        cfg = best_cfg[primary]
        y = label_from_z(z, primary)
        rows = []
        for st in sorted(df["State"].dropna().unique()):
            tr = df[(df["State"] != st) & (df["Year"] < SPLIT_YEAR) & y.notna()]
            va = df[(df["State"] == st) & (df["Year"] >= SPLIT_YEAR) & y.notna()]
            if len(va) < 20 or y.loc[tr.index].sum() < 10:
                continue
            ytr = y.loc[tr.index].to_numpy().astype(int)
            s, p = fit_predict(cfg["model"], tr, va, ytr, fsets[cfg["fset"]], a.district_z, a.fast)
            rows.append({"held_out_state": st, **metrics(y.loc[va.index].to_numpy().astype(int), s, p)})
        if not rows:
            print("\nLOSO skipped: every held-out state has < 20 labelled test rows or too few training positives "
                  "(lower SPLIT_YEAR or add districts).")
        if rows:
            lo = pd.DataFrame(rows); lo.to_csv(os.path.join(OUT_DIR, "loso_results.csv"), index=False)
            print("\n" + "=" * 70 + "\nLEAVE-ONE-STATE-OUT (train other states < SPLIT_YEAR, test held-out state >= SPLIT_YEAR)\n" + "=" * 70)
            print(lo[["held_out_state", "n", "prevalence", "F1", "recall", "ROC_AUC", "PR_lift"]].round(3).to_string(index=False))

    print("\nInterpretation: a model is useful only if PR_lift >> 1 and F1 > always_pos_F1 on the TEST table; "
          "check the by-year table to see whether it detects events in years it never saw (e.g. 2022).")


if __name__ == "__main__":
    main()