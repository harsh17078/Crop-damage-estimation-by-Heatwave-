"""
OPTION A — STEP A2: WHICH LABEL? (signal x baseline x threshold)
================================================================

z <= -1 is only a CANDIDATE. This script tests every combination of

    signal     S_ndvi | S_dndvi | S_evi | S_gpp
    baseline   expanding | rolling | trend           (all PAST-ONLY, see proxy_label_lib)
    threshold  z <= -0.5 (mild) | -1.0 | -1.5 | -2.0 (extreme)

and reports, per combination:

  CLASS BALANCE        pos_rate, n_pos, n_pos_train (years < SPLIT_YEAR), pos_rate_test
  POSITIVE YEARS       years_with_pos (>=3 positive districts), top3_share (share of all
                       positives that sit in only 3 years: high = "label = a few events")
  SPATIAL DISTRIBUTION per-state positive rate, districts_with_pos, coherence
                       (share of positives inside state-years where >= 25% of districts are
                       positive. Real heat/rain/hail events are regional; isolated positives
                       are more likely cloud / mask / noise artefacts)
  STABILITY            kappa between baselines and between signals at the same threshold
  VALIDITY vs DES      spearman(z, yield z), AUC of -z for yield loss (yield z <= -1),
                       kappa with the yield-loss label at the same threshold.
                       DES is used ONLY to judge the proxy - never as a training label.
  (F1 / recall of the models is measured next, in A3, with the same labels.)

Every check becomes a PASS / FAIL / n.a. line; the candidate with most passes (then best
yield validity) is written to recommended_label.json for A3.

Also printed: a year-shift check of the proxy against DES (shift 0 must be strongest; if
not, the DES year key is mis-aligned - see script 07a).

Input : data/processed/A_features_signals.csv  (A1)
        data/processed/des_yield_clean.csv     (optional; HARVEST-year labels)
Output: reports/proxy_labels/label_candidates.csv, recommended_label.json, *.png
Run   : python scripts/A2_label_threshold_analysis.py
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_label_lib import kappa, label_from_z, past_z

warnings.filterwarnings("ignore")

try:
    from project_config import SPLIT_YEAR
except ModuleNotFoundError:
    SPLIT_YEAR = 2019

IN_PATH = "data/processed/A_features_signals.csv"
DES_PATH = "data/processed/des_yield_clean.csv"
OUT_DIR = "reports/proxy_labels"

SIGNALS = ["S_ndvi", "S_dndvi", "S_evi", "S_peak", "S_int", "S_late", "S_p20", "S_gpp"]
METHODS = ["expanding", "rolling", "trend"]
THRESHOLDS = [-0.5, -1.0, -1.5, -2.0]
MIN_HIST = 5
DROP_FALLBACK = True      # exclude districts flagged P_fallback=1 by A1 (peak not wheat-like)
EXCLUDE_DISTRICTS = []    # extra districts to leave out for a sensitivity test, e.g. ["Jhajjar"]

# pass criteria (edit to taste)
CRIT = {
    "balance": (0.08, 0.35),        # positive rate: "damage" should be the minority class
    "n_pos_train": 40,              # ~ 4 positives per 10 districts per year is the lower limit for learning
    "years_with_pos": 6,
    "top3_share_max": 0.60,
    "coherence_min": 0.50,
    "spearman_min": 0.25,           # z vs yield z (district-year)
    "stateyear_min": 0.40,          # state-year means: regional events average out district noise
    "auc_min": 0.60,
    "kappa_min": 0.50,
}


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--des", default=DES_PATH, help="path to des_yield_clean.csv (District, Year, Yield; HARVEST year)")
    ap.add_argument("--exclude", nargs="*", default=[], help="extra districts to leave out, e.g. --exclude Jhajjar")
    args = ap.parse_args()
    des_path = args.des
    excl = list(EXCLUDE_DISTRICTS) + list(args.exclude)

    section("A2 — LABEL CANDIDATE ANALYSIS")
    df = pd.read_csv(IN_PATH)
    if DROP_FALLBACK and "P_fallback" in df.columns and df["P_fallback"].sum() > 0:
        bad = sorted(df.loc[df["P_fallback"] == 1, "District"].unique())
        print(f"Excluding {len(bad)} district(s) whose NDVI peak is not wheat-like (mustard/gram/other): {bad}")
        df = df[df["P_fallback"] == 0]
    if excl:
        print(f"Excluding by request: {excl}")
        df = df[~df["District"].isin(excl)]
    df = df.reset_index(drop=True)
    signals = [s for s in SIGNALS if s in df.columns and df[s].notna().sum() > 100]
    print(f"rows {len(df)} | signals {signals} | split year {SPLIT_YEAR}")

    # ---------------- z-scores ----------------
    Z = {(s, m): past_z(df, s, m, MIN_HIST) for s in signals for m in METHODS}

    # ---------------- DES (validation only) ----------------
    YZ = None
    if os.path.exists(des_path):
        des = pd.read_csv(des_path)[["District", "Year", "Yield"]]
        tmp = df[["State", "District", "Year"]].merge(des, on=["District", "Year"], how="left")
        tmp["logy"] = np.log(tmp["Yield"].where(tmp["Yield"] > 0))
        YZ = {m: past_z(tmp, "logy", m, MIN_HIST) for m in METHODS}
        print(f"DES yield matched for {tmp['Yield'].notna().sum()} district-years (validation only).")
        print("  DES 'Year' must be the HARVEST year (2014-15 season = 2015).")
    else:
        print(f"\n[WARNING] DES file not found at: {os.path.abspath(des_path)}")
        print("  Validity checks (correlation with yield) are SKIPPED. Without them this ranking only reflects")
        print("  class balance and stability - DO NOT choose the final threshold from this run.")
        print("  Fix: pass the file with  --des path/to/des_yield_clean.csv  (columns District, Year, Yield).")

    # ---------------- labels ----------------
    LAB = {(s, m, t): label_from_z(Z[(s, m)], t) for s in signals for m in METHODS for t in THRESHOLDS}

    # ---------------- per-candidate metrics ----------------
    rows = []
    for s in signals:
        for m in METHODS:
            for t in THRESHOLDS:
                lab = LAB[(s, m, t)]
                d = df[["State", "District", "Year"]].assign(z=Z[(s, m)], lab=lab).dropna(subset=["lab"])
                if len(d) < 50:
                    continue
                pos = int(d["lab"].sum())
                yearly = d.groupby("Year")["lab"].agg(["sum", "mean"])
                cell = d.groupby(["State", "Year"])["lab"].transform("mean")
                r = {
                    "signal": s, "baseline": m, "thr": t, "n": len(d), "n_pos": pos,
                    "pos_rate": pos / len(d),
                    "n_pos_train": int(d.loc[d["Year"] < SPLIT_YEAR, "lab"].sum()),
                    "pos_rate_test": float(d.loc[d["Year"] >= SPLIT_YEAR, "lab"].mean())
                    if (d["Year"] >= SPLIT_YEAR).any() else np.nan,
                    "years_with_pos": int((yearly["sum"] >= 3).sum()),
                    "top3_share": float(yearly["sum"].nlargest(3).sum() / max(pos, 1)),
                    "yearly_rate_std": float(yearly["mean"].std()),
                    "districts_with_pos": int(d.loc[d["lab"] == 1, "District"].nunique()),
                    "top_years": ",".join(str(int(y)) for y in yearly["sum"].nlargest(3).index),
                    "coherence": float(((cell >= 0.25) & (d["lab"] == 1)).sum() / pos) if pos else np.nan,
                }
                for st, v in d.groupby("State")["lab"].mean().items():
                    r[f"rate_{st}"] = float(v)

                if YZ is not None:
                    v = d.assign(yz=YZ[m].reindex(d.index)).dropna(subset=["yz"])
                    if len(v) > 50:
                        r["n_valid"] = len(v)
                        r["spearman_yield"] = float(spearmanr(v["z"], v["yz"])[0])
                        yl = (v["yz"] <= -1.0).astype(int)
                        r["auc_yield_loss"] = float(roc_auc_score(yl, -v["z"])) if yl.nunique() == 2 else np.nan
                        r["kappa_yield"] = kappa(v["lab"], (v["yz"] <= t).astype(float))
                        r["yz_gap"] = float(v.loc[v["lab"] == 1, "yz"].mean() - v.loc[v["lab"] == 0, "yz"].mean())
                        for nm, sub in (("train", v[v["Year"] < SPLIT_YEAR]), ("test", v[v["Year"] >= SPLIT_YEAR])):
                            if len(sub) > 30:
                                r[f"spearman_yield_{nm}"] = float(spearmanr(sub["z"], sub["yz"])[0])
                        sy = v.groupby(["State", "Year"])[["z", "yz"]].mean()
                        if len(sy) >= 20:
                            r["spearman_stateyear"] = float(spearmanr(sy["z"], sy["yz"])[0])
                rows.append(r)

    res = pd.DataFrame(rows)

    # ---------------- stability (kappa) ----------------
    kb, ks = [], []
    for _, r in res.iterrows():
        s, m, t = r["signal"], r["baseline"], r["thr"]
        others_m = [kappa(LAB[(s, m, t)], LAB[(s, m2, t)]) for m2 in METHODS if m2 != m]
        others_s = [kappa(LAB[(s, m, t)], LAB[(s2, m, t)]) for s2 in signals if s2 != s]
        kb.append(np.nanmean(others_m) if others_m and not np.all(np.isnan(others_m)) else np.nan)
        ks.append(np.nanmean(others_s) if others_s and not np.all(np.isnan(others_s)) else np.nan)
    res["kappa_baselines"], res["kappa_signals"] = kb, ks

    # ---------------- checklist ----------------
    def chk(r):
        out = {}
        out["balance"] = CRIT["balance"][0] <= r["pos_rate"] <= CRIT["balance"][1]
        out["n_pos_train"] = r["n_pos_train"] >= CRIT["n_pos_train"]
        out["years_with_pos"] = r["years_with_pos"] >= CRIT["years_with_pos"]
        out["top3_share"] = r["top3_share"] <= CRIT["top3_share_max"]
        out["coherence"] = (r["coherence"] >= CRIT["coherence_min"]) if r["coherence"] == r["coherence"] else None
        out["stability"] = (r["kappa_baselines"] >= CRIT["kappa_min"]) if r["kappa_baselines"] == r["kappa_baselines"] else None
        sp, au = r.get("spearman_yield", np.nan), r.get("auc_yield_loss", np.nan)
        out["validity_spearman"] = (sp >= CRIT["spearman_min"]) if sp == sp else None
        out["validity_auc"] = (au >= CRIT["auc_min"]) if au == au else None
        sy_ = r.get("spearman_stateyear", np.nan)
        out["validity_stateyear"] = (sy_ >= CRIT["stateyear_min"]) if sy_ == sy_ else None
        return {k: (None if v is None else bool(v)) for k, v in out.items()}

    checks = res.apply(chk, axis=1, result_type="expand")
    res["checks_passed"] = checks.apply(lambda c: sum(1 for v in c if v is True), axis=1)
    res["checks_total"] = checks.apply(lambda c: sum(1 for v in c if v is not None), axis=1)
    res["failed"] = checks.apply(lambda c: ",".join(k for k, v in c.items() if v is False), axis=1)
    # ranking: regional validity vs DES (state-year Spearman) -> checks passed
    #          -> label-level agreement with DES yield loss at THIS threshold (breaks threshold ties)
    res["_sp"] = res["spearman_stateyear"].round(2).fillna(-1) if "spearman_stateyear" in res else 0.0
    res["_kp"] = res["kappa_yield"].fillna(-1) if "kappa_yield" in res else 0.0
    res = (res.sort_values(["_sp", "checks_passed", "_kp"], ascending=False)
              .drop(columns=["_sp", "_kp"]).reset_index(drop=True))

    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(os.path.join(OUT_DIR, "label_candidates.csv"), index=False)

    # ---------------- report ----------------
    section("TOP CANDIDATES")
    cols = ["signal", "baseline", "thr", "pos_rate", "n_pos_train", "years_with_pos", "top3_share",
            "coherence", "kappa_baselines", "spearman_yield", "spearman_yield_train", "spearman_yield_test",
            "spearman_stateyear", "auc_yield_loss", "checks_passed",
            "checks_total", "failed"]
    cols = [c for c in cols if c in res.columns]
    print(res[cols].head(12).round(3).to_string(index=False))

    if "spearman_stateyear" in res.columns:
        section("SIGNAL VALIDITY SUMMARY (one row per signal x baseline; threshold-independent)")
        sv = (res[res["thr"] == -1.0]
              [["signal", "baseline", "spearman_yield", "spearman_yield_train", "spearman_yield_test",
                "spearman_stateyear", "auc_yield_loss", "top_years", "pos_rate"]]
              .sort_values("spearman_stateyear", ascending=False))
        print(sv.round(3).to_string(index=False))
        print("\nRead: a usable signal needs spearman_stateyear >= ~0.4 AND a similar value in train and test; "
              "top_years (z <= -1) should contain known bad seasons (e.g. 2015 rain/hail, 2022 heat wave).")

    best = res.iloc[0]
    section("THRESHOLD LADDER FOR THE BEST SIGNAL / BASELINE")
    lad = res[(res["signal"] == best["signal"]) & (res["baseline"] == best["baseline"])].sort_values("thr", ascending=False)
    state_cols = [c for c in lad.columns if c.startswith("rate_")]
    lad_cols = [c for c in ["thr", "n_pos", "pos_rate", "pos_rate_test", "years_with_pos", "top3_share", "top_years",
                            "coherence", "spearman_yield", "spearman_stateyear", "auc_yield_loss", "failed"] if c in lad.columns]
    print(lad[lad_cols + state_cols].round(3).to_string(index=False))
    print("\nReading guide: z <= -0.5 is usually too frequent (noise-level), z <= -2 too rare to learn from; "
          "pick the threshold where balance, spatial coherence and DES validity are all acceptable.")

    # ---------------- DES year-shift check ----------------
    if YZ is not None:
        section("DES YEAR-ALIGNMENT CHECK (best signal)")
        z = Z[(best["signal"], best["baseline"])]
        zdf = df[["District", "Year"]].assign(z=z)
        ydf = df[["District", "Year"]].assign(yz=YZ[best["baseline"]])
        for sh in (-1, 0, 1):
            y2 = ydf.assign(Year=ydf["Year"] - sh)
            m = zdf.merge(y2, on=["District", "Year"]).dropna()
            print(f"  shift {sh:+d}: spearman(z, yield z) = {spearmanr(m['z'], m['yz'])[0]:+.3f}  (n={len(m)})"
                  if len(m) > 30 else f"  shift {sh:+d}: too few pairs")
        print("  Expected: shift +0 is the largest. Otherwise the DES labels are stored under the wrong year.")

    # ---------------- year-level table (is the label dominated by common-year effects?) ----------------
    section("YEAR-LEVEL TABLE (best signal / baseline / threshold)")
    yt = df[["State", "District", "Year"]].assign(z=Z[(best["signal"], best["baseline"])])
    yt["lab"] = (yt["z"] <= best["thr"]).astype(float).where(yt["z"].notna())
    aggs = {"n": ("z", "count"), "mean_z": ("z", "mean"), "pos_rate": ("lab", "mean")}
    if YZ is not None:
        yt["yz"] = YZ[best["baseline"]]
        aggs["mean_yield_z"] = ("yz", "mean")
    tab = yt.groupby("Year").agg(**aggs).dropna(subset=["mean_z"]).round(3)
    print(tab.to_string())
    tab.to_csv(os.path.join(OUT_DIR, "year_level_table.csv"))
    print("\nRead: years where almost every district is positive (pos_rate near 1) while mean_yield_z is NOT "
          "negative are common-year artefacts of the satellite signal (fog, composites, sensor), not damage.")

    # ---------------- plots ----------------
    d0 = df[["State", "District", "Year"]].copy()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for t in THRESHOLDS:
        d = d0.assign(lab=LAB[(best["signal"], best["baseline"], t)]).dropna(subset=["lab"])
        ax[0].plot(d.groupby("Year")["lab"].mean(), marker="o", label=f"z<={t}")
    ax[0].set_title(f"Positive rate per year ({best['signal']}, {best['baseline']})")
    ax[0].set_ylabel("share of districts"); ax[0].legend(); ax[0].axvline(SPLIT_YEAR - 0.5, color="grey", ls="--")
    w = 0.2
    states = sorted(d0["State"].dropna().unique())
    for i, t in enumerate(THRESHOLDS):
        d = d0.assign(lab=LAB[(best["signal"], best["baseline"], t)]).dropna(subset=["lab"])
        rates = d.groupby("State")["lab"].mean().reindex(states)
        ax[1].bar(np.arange(len(states)) + i * w, rates.values, w, label=f"z<={t}")
    ax[1].set_xticks(np.arange(len(states)) + 1.5 * w); ax[1].set_xticklabels(states, rotation=20)
    ax[1].set_title("Positive rate by state"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "label_balance.png"), dpi=130); plt.close(fig)

    # ---------------- recommendation ----------------
    rec = {
        "signal": best["signal"], "baseline": best["baseline"], "threshold": float(best["thr"]),
        "validated_against_DES": bool(YZ is not None),
        "checks_passed": f"{int(best['checks_passed'])}/{int(best['checks_total'])}",
        "failed_checks": best["failed"],
        "alternatives": [
            {"signal": r["signal"], "baseline": r["baseline"], "threshold": float(r["thr"]),
             "checks_passed": int(r["checks_passed"])}
            for _, r in res.iloc[1:5].iterrows()],
    }
    with open(os.path.join(OUT_DIR, "recommended_label.json"), "w") as f:
        json.dump(rec, f, indent=2)
    section("RECOMMENDATION")
    print(json.dumps(rec, indent=2))
    print("\nFiles in", OUT_DIR, ": label_candidates.csv, recommended_label.json, label_balance.png")
    print("Next: A3_train_classifiers.py  (add --sweep to compare F1/recall across all four thresholds)")


if __name__ == "__main__":
    main()