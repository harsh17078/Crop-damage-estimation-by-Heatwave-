"""
B4 - RMSE / R2 BASELINE FOR CONTINUOUS HEAT-EXPOSURE TARGETS
============================================================

The B1 classifier is judged by AUC / MCC / balanced accuracy.  RMSE and R2 only make sense for a CONTINUOUS target,
so this script predicts, from MODIS-only features, quantities you already have from ERA5-Land:

    n35          number of days with Tmax >= 35 C in the window P..P+40          (count)
    hdd35        degree-days above 35 C in the window                             (non-negative continuous)
    tmax_mean_W  mean Tmax in the window                                          (continuous, symmetric)

It is NOT a yield / damage model (the yield link was tested and is null).

DESIGN (same rules as B1-B3)
  * year-blocked validation only: rolling origin (train on earlier years) and leave-one-year-out. No random splits.
  * baselines first: training mean, district climatology, linear model on LST p90 alone.
  * candidates: Ridge (penalty chosen with YEAR-grouped CV), Poisson regression, log-target Ridge with smearing,
    optional gradient boosting (--with-gbm).  Count predictions are clipped at 0.
  * a candidate counts as an IMPROVEMENT only if its year-bootstrap 95% interval for
        RMSE(ols_p90) - RMSE(candidate)
    lies entirely above 0 on BOTH rolling_all and loyo_all.  Otherwise the simplest model (ols_p90) is recommended.

METRICS
  RMSE, MAE, bias, pooled R2 (includes hot-year vs cool-year differences - flattering), R2_within_year (year means
  removed from both observed and predicted: the honest district-ranking number), Spearman, skill_vs_clim
  (1 - MSE/MSE_climatology), year-level correlation, and a calibration table (predicted bins vs observed mean).

Input : reports/exposure/exposure_table.csv  (B1)   [+ exposure_table_meta.json for the split year]
Output: reports/exposure/B4/<target>/*.csv  and  reports/exposure/B4/run_B4.log
Run   : python scripts/B4_regression_baseline.py                    (all three targets)
        python scripts/B4_regression_baseline.py --target n35 --with-gbm --n-boot 2000
        python scripts/B4_regression_baseline.py --selftest           (synthetic data)
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, PoissonRegressor, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------- settings (edit BEFORE running) ----------------------------------
TABLE = "reports/exposure/exposure_table.csv"
OUT_BASE = "reports/exposure/B4"
SPLIT_YEAR = 2019
MIN_TRAIN_YEARS = 5
MIN_TRAIN_ROWS = 80
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
POISSON_ALPHA = 0.1
N_BOOT = 1000
SEED = 42
BASELINE = "ols_p90"
TARGETS = {"n35": "count", "hdd35": "count", "tmax_mean_W": "cont"}
# ----------------------------------------------------------------------------------------------------


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for st in self.s:
            st.write(x)

    def flush(self):
        for st in self.s:
            st.flush()


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ------------------------------------------------------------------ models
def make_specs(df, ttype, with_gbm):
    sat = [c for c in df.columns if c.startswith("sat_")]
    lst = [c for c in sat if "lst" in c]
    p90 = ["sat_lst_day_p90"]
    s = {"mean": ("mean", []), "clim": ("clim", []),
         "ols_p90": ("ols", p90), "ridge_lst8": ("ridge", lst), "ridge_all_sat": ("ridge", sat)}
    if ttype == "count":
        s.update({"poisson_p90": ("poisson", p90), "poisson_lst8": ("poisson", lst),
                  "ridge_log_lst8": ("ridge_log", lst)})
    if with_gbm:
        s["gbm_lst8"] = ("gbm", lst)
    return s


def pre():
    return [SimpleImputer(strategy="median"), StandardScaler()]


def ridge_pipe(Xtr, ytr, years):
    k = min(5, len(np.unique(years)))
    if k >= 3:
        cv = list(GroupKFold(k).split(Xtr, ytr, years))          # year-grouped: avoids the old leave-one-row-out defect
        m = make_pipeline(*pre(), RidgeCV(alphas=RIDGE_ALPHAS, cv=cv))
    else:
        m = make_pipeline(*pre(), RidgeCV(alphas=(10.0,)))
    return m.fit(Xtr, ytr)


def fit_predict(kind, cols, tr, va, ytr, ttype):
    n = len(va)
    if kind == "mean":
        return np.full(n, ytr.mean())
    if kind == "clim":
        m = pd.Series(ytr).groupby(tr["District"].to_numpy()).mean()
        return va["District"].map(m).fillna(ytr.mean()).to_numpy()
    Xtr, Xva = tr[cols].to_numpy(float), va[cols].to_numpy(float)
    if kind == "ols":
        p = make_pipeline(*pre(), LinearRegression()).fit(Xtr, ytr).predict(Xva)
    elif kind == "ridge":
        p = ridge_pipe(Xtr, ytr, tr["Year"].to_numpy()).predict(Xva)
    elif kind == "poisson":
        p = make_pipeline(*pre(), PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000)).fit(Xtr, ytr).predict(Xva)
    elif kind == "ridge_log":
        m = ridge_pipe(Xtr, np.log1p(ytr), tr["Year"].to_numpy())
        smear = np.mean(np.exp(np.log1p(ytr) - m.predict(Xtr)))   # Duan smearing for the back-transform
        p = np.exp(m.predict(Xva)) * smear - 1.0
    elif kind == "gbm":
        g = HistGradientBoostingRegressor(loss="poisson" if ttype == "count" else "squared_error", max_depth=3,
                                          learning_rate=0.05, max_iter=200, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=SEED)
        p = g.fit(Xtr, ytr).predict(Xva)
    else:
        raise ValueError(kind)
    return np.clip(p, 0, None) if ttype == "count" else p


def run(df, target, ttype, eval_years, scheme, specs):
    out = []
    for v in eval_years:
        tr = df[(df["Year"] < v) if scheme == "rolling" else (df["Year"] != v)]
        tr = tr[tr[target].notna()]
        va = df[(df["Year"] == v) & df[target].notna()]
        if len(tr) < MIN_TRAIN_ROWS or len(va) == 0:
            continue
        ytr = tr[target].to_numpy(float)
        for name, (kind, cols) in specs.items():
            p = fit_predict(kind, cols, tr, va, ytr, ttype)
            out.append(pd.DataFrame({"model": name, "Year": int(v), "District": va["District"].to_numpy(),
                                     "State": va["State"].to_numpy(), "obs": va[target].to_numpy(float), "pred": p}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------------------ metrics
def within_year_r2(d):
    oa = d["obs"] - d.groupby("Year")["obs"].transform("mean")
    pa = d["pred"] - d.groupby("Year")["pred"].transform("mean")
    den = (oa ** 2).sum()
    return 1 - ((oa - pa) ** 2).sum() / den if den > 0 else np.nan


def summarize(d):
    o, p = d["obs"].to_numpy(float), d["pred"].to_numpy(float)
    e = o - p
    sse, sst = (e ** 2).sum(), ((o - o.mean()) ** 2).sum()
    yl = d.groupby("Year")[["obs", "pred"]].mean()
    ok = len(yl) >= 4 and yl["pred"].std() > 0
    return {"n": len(d), "obs_mean": o.mean(), "RMSE": np.sqrt(sse / len(d)), "MAE": np.abs(e).mean(),
            "bias": (p - o).mean(), "R2": 1 - sse / sst if sst > 0 else np.nan, "R2_within_year": within_year_r2(d),
            "spearman": spearmanr(p, o)[0] if np.std(p) > 0 else np.nan,
            "year_level_r": pearsonr(yl["pred"], yl["obs"])[0] if ok else np.nan}


def summary_table(pred):
    t = pd.DataFrame([{"model": m, **summarize(g)} for m, g in pred.groupby("model", sort=False)])
    clim = t.loc[t["model"] == "clim", "RMSE"]
    if len(clim):
        t["skill_vs_clim"] = 1 - (t["RMSE"] / clim.iloc[0]) ** 2
    return t


def year_bootstrap(pred, models, base, n_boot):
    years = sorted(pred["Year"].unique())
    g0 = pred[pred["model"] == base].groupby("Year")
    n = g0.size().reindex(years).to_numpy(float)
    s1 = g0["obs"].sum().reindex(years).to_numpy(float)
    s2 = pred[pred["model"] == base].assign(o2=lambda x: x["obs"] ** 2).groupby("Year")["o2"].sum().reindex(years).to_numpy()
    sse = {m: pred[pred["model"] == m].assign(se=lambda x: (x["obs"] - x["pred"]) ** 2)
           .groupby("Year")["se"].sum().reindex(years).to_numpy() for m in models}
    idx = np.random.default_rng(SEED).integers(0, len(years), (n_boot, len(years)))
    N = n[idx].sum(1)
    sst = s2[idx].sum(1) - s1[idx].sum(1) ** 2 / N
    rm = {m: np.sqrt(sse[m][idx].sum(1) / N) for m in models}
    r2 = {m: 1 - sse[m][idx].sum(1) / sst for m in models}
    ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
    rows = []
    for m in models:
        lo, hi = ci(rm[m]); rlo, rhi = ci(r2[m])
        row = {"model": m, "RMSE": rm[m].mean(), "RMSE_lo95": lo, "RMSE_hi95": hi,
               "R2": r2[m].mean(), "R2_lo95": rlo, "R2_hi95": rhi}
        if m != base:
            d = rm[base] - rm[m]
            row.update({"dRMSE_vs_base": d.mean(), "dRMSE_lo95": ci(d)[0], "dRMSE_hi95": ci(d)[1]})
        rows.append(row)
    return pd.DataFrame(rows)


def calibration(pred, model, bins=5):
    d = pred[pred["model"] == model].copy()
    d["bin"] = pd.qcut(d["pred"].rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(n=("obs", "size"), mean_pred=("pred", "mean"), mean_obs=("obs", "mean")).round(3)


# ------------------------------------------------------------------ one target
def do_target(df, target, ttype, a, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    section(f"TARGET = {target}  ({ttype}) | rows with target {int(df[target].notna().sum())} | "
            f"mean {df[target].mean():.2f} | sd {df[target].std():.2f}")
    years = sorted(df["Year"].unique())
    first_val = years[MIN_TRAIN_YEARS]
    specs = make_specs(df, ttype, a.with_gbm)
    roll = run(df, target, ttype, [v for v in years if v >= first_val], "rolling", specs)
    loyo = run(df, target, ttype, years, "loyo", specs)
    sets = {"rolling_test": roll[roll["Year"] >= SPLIT_YEAR], "rolling_all": roll, "loyo_all": loyo}
    show = ["model", "n", "RMSE", "MAE", "bias", "R2", "R2_within_year", "spearman", "year_level_r", "skill_vs_clim"]
    summ = {}
    for tag, pr in sets.items():
        summ[tag] = summary_table(pr)
        summ[tag].to_csv(os.path.join(out_dir, f"summary_{tag}.csv"), index=False)
        print(f"\n--- {tag} (years {pr['Year'].min()}..{pr['Year'].max()}, {pr['Year'].nunique()} years) ---")
        print(summ[tag][show].round(3).to_string(index=False))
    print("\nR2 = pooled (includes hot-year vs cool-year differences); R2_within_year = year means removed (honest).")

    cands = [m for m in specs if m not in ("mean", "clim", BASELINE)]
    KIND_RANK = {"ols": 1, "poisson": 2, "ridge": 3, "ridge_log": 4, "gbm": 5}
    cx = {m: (len(specs[m][1]), KIND_RANK.get(specs[m][0], 9)) for m in specs}      # (n features, model-type rank)
    rl = summ["rolling_all"].set_index("model")["RMSE"]
    section(f"{target}: STEPWISE SELECTION - a candidate replaces the current model only if its year-bootstrap "
            f"interval for the RMSE gain is above 0 on rolling_all AND loyo_all (simplest winner wins)")
    current, pool, trail = BASELINE, list(cands), [BASELINE]
    for step in range(1, 6):
        models = [current] + [m for m in pool if m != current]
        wins = {}
        for tag in ("rolling_all", "loyo_all"):
            sub = sets[tag][sets[tag]["model"].isin(models)]
            b = year_bootstrap(sub, models, current, a.n_boot)
            b.to_csv(os.path.join(out_dir, f"bootstrap_step{step}_vs_{current}_{tag}.csv"), index=False)
            print(f"\n--- step {step}: candidates vs {current} | {tag} ---")
            print(b.rename(columns={"RMSE": "RMSE_boot_mean", "R2": "R2_boot_mean"}).round(3).to_string(index=False))
            wins[tag] = set(b.loc[b["dRMSE_lo95"] > 0, "model"].dropna())
        both = sorted(wins["rolling_all"] & wins["loyo_all"], key=lambda m: (cx[m], rl[m]))
        if not both:
            print(f"\nstep {step}: no candidate beats {current} on both schemes -> stop.")
            break
        print(f"\nstep {step}: significant improvements over {current}: {both} -> take the simplest: {both[0]}")
        current = both[0]
        trail.append(current)
        pool = [m for m in pool if m != current]
    best = current

    section(f"{target}: VERDICT")
    for tag in ("rolling_all", "loyo_all"):
        t = summ[tag].set_index("model")
        print(f"[{tag}] {BASELINE}: RMSE {t.loc[BASELINE, 'RMSE']:.3f}, R2 {t.loc[BASELINE, 'R2']:.3f} | "
              f"{best}: RMSE {t.loc[best, 'RMSE']:.3f}, R2 {t.loc[best, 'R2']:.3f}, "
              f"R2_within_year {t.loc[best, 'R2_within_year']:.3f} | climatology RMSE {t.loc['clim', 'RMSE']:.3f}")
    print(f"selection path: {' -> '.join(trail)}")
    print(f"-> recommended model for {target}: {best}" +
          ("  (the simplest linear model; added features/models do not help)" if best == BASELINE else ""))
    both = trail[1:]
    cal = calibration(sets["rolling_all"], best)
    cal.to_csv(os.path.join(out_dir, "calibration_recommended.csv"))
    print(f"\nCalibration of {best} (rolling_all): predicted bins vs observed mean")
    print(cal.to_string())
    py = sets["rolling_test"][sets["rolling_test"]["model"] == best]
    byy = py.groupby("Year").apply(lambda g: pd.Series({
        "n": len(g), "obs_mean": g["obs"].mean(), "pred_mean": g["pred"].mean(),
        "RMSE": np.sqrt(((g["obs"] - g["pred"]) ** 2).mean())}))
    byy.to_csv(os.path.join(out_dir, "test_by_year_recommended.csv"))
    print(f"\nTest years, {best}:")
    print(byy.round(2).to_string())
    sets["rolling_all"][sets["rolling_all"]["model"] == best].to_csv(os.path.join(out_dir, "predictions_recommended.csv"), index=False)
    return {"target": target, "recommended": best, "improved_over_baseline": bool(both),
            "RMSE_rolling_all": float(summ["rolling_all"].set_index("model").loc[best, "RMSE"]),
            "R2_rolling_all": float(summ["rolling_all"].set_index("model").loc[best, "R2"]),
            "R2_within_year_rolling_all": float(summ["rolling_all"].set_index("model").loc[best, "R2_within_year"]),
            "RMSE_loyo_all": float(summ["loyo_all"].set_index("model").loc[best, "RMSE"]),
            "RMSE_clim_rolling_all": float(summ["rolling_all"].set_index("model").loc["clim", "RMSE"])}


# ------------------------------------------------------------------ selftest / main
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
                r = {"State": st, "District": f"{st[:2]}_{i}", "Year": yr, "n35": n35,
                     "hdd35": float(max(0, T - 35) * 3 + 0.2 * n35), "tmax_mean_W": T - 4 + rng.normal(0, .3),
                     "sat_lst_day_mean": lstp - 3 + rng.normal(0, .5), "sat_lst_day_max": lstp + 1 + rng.normal(0, .5),
                     "sat_lst_day_p90": lstp, "sat_lst_day_std": abs(rng.normal(2, .5)),
                     "sat_lst_night_mean": 12 + 0.3 * (T - 33) + rng.normal(0, 1), "sat_lst_dtr": 20 + rng.normal(0, 2),
                     "sat_lst_day_pre": 30 + rng.normal(0, 2), "sat_lst_rise": 5 + rng.normal(0, 2),
                     "sat_ndvi_W": 0.5 + rng.normal(0, .05), "sat_ndvi_change": rng.normal(0, .05),
                     "sat_ndmi_W": rng.normal(0.2, .05), "sat_evi_W": 0.3 + rng.normal(0, .03)}
                rows.append(r)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="B4: RMSE/R2 baseline for continuous heat-exposure targets")
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--target", default="all", choices=["all"] + list(TARGETS))
    ap.add_argument("--with-gbm", action="store_true", help="also fit gradient boosting (an ablation, run last)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--split-year", type=int, default=None)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    global SPLIT_YEAR
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if not a.selftest and os.path.isdir(os.path.join(root, "data")) and os.path.abspath(os.getcwd()) != root:
        os.chdir(root)
    out_base = OUT_BASE
    if a.selftest:
        a.table, out_base = "/tmp/b4_selftest/exposure_table.csv", "/tmp/b4_selftest/out"
        make_synthetic(a.table)
        print("[selftest] synthetic table:", a.table)
    meta = os.path.join(os.path.dirname(a.table), "exposure_table_meta.json")
    if a.split_year is not None:
        SPLIT_YEAR = a.split_year
    elif os.path.exists(meta):
        SPLIT_YEAR = int(json.load(open(meta))["split_year"])
    if not os.path.exists(a.table):
        sys.exit(f"[ERROR] {a.table} not found - run B1 first")
    df = pd.read_csv(a.table)
    miss = {"State", "District", "Year", "sat_lst_day_p90"} - set(df.columns)
    if miss:
        sys.exit(f"[ERROR] table lacks {sorted(miss)} - rerun B1")
    if a.exclude:
        df = df[~df["District"].isin(a.exclude)]
    df = df.sort_values(["State", "District", "Year"]).reset_index(drop=True)
    os.makedirs(out_base, exist_ok=True)
    logf = open(os.path.join(out_base, "run_B4.log"), "w")
    old = sys.stdout
    sys.stdout = Tee(old, logf)
    try:
        print(f"B4 | rows {len(df)} | districts {df['District'].nunique()} | years {df['Year'].min()}-{df['Year'].max()} "
              f"| test years >= {SPLIT_YEAR}")
        targets = TARGETS if a.target == "all" else {a.target: TARGETS[a.target]}
        res = [do_target(df, t, tt, a, os.path.join(out_base, t)) for t, tt in targets.items() if t in df.columns]
        section("SUMMARY - recommended model per target (rolling origin over 2008..2024 unless the split says otherwise)")
        sm = pd.DataFrame(res)
        sm.to_csv(os.path.join(out_base, "summary_all_targets.csv"), index=False)
        print(sm.round(3).to_string(index=False))
        print("\nRead: RMSE is in the units of the target (days, degree-days, C). Compare it with RMSE_clim: the gain over\n"
              "district climatology is what the satellite adds. R2_within_year is the number to quote for district ranking.\n"
              "Do not tune further: if the recommended model is ols_p90, that is the production regressor.")
    finally:
        sys.stdout = old
        logf.close()


if __name__ == "__main__":
    main()
    