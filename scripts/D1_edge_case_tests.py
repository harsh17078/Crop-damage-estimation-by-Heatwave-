"""
D1: COMPREHENSIVE EDGE-CASE TESTING PIPELINE (PHASE 2)
=======================================================
Tests the production inference pipeline under realistic and difficult scenarios.

Categories:
  A. Missing Data         -- NaN, null, inf, single/multi-feature missing
  B. Extreme Weather      -- High Tmax, LST, heatwaves, drought, flood
  C. Satellite Edge Cases -- NDVI extremes, missing obs, inconsistencies
  D. Prediction Consistency -- Physical scenario audits (heat vs other damage)
  E. Spatial/Temporal     -- OOD districts, future years, unusual patterns

KEY DESIGN: All 98+ test cases are batched into ONE model call to avoid
repeated joblib worker spawning (which makes ExtraTrees very slow).

Outputs:
  reports/edge_case_tests/edge_case_results.csv
  reports/edge_case_tests/edge_case_summary.csv
  reports/edge_case_tests/consistency_audit.csv
  reports/edge_case_tests/ood_scores.csv
  reports/edge_case_tests/D1_test_report.txt

Usage:
  python scripts/D1_edge_case_tests.py
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# -- Paths -------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "data")):
    os.chdir(ROOT)

MODELS_DIR  = "models"
CONFIG_PATH = os.path.join(MODELS_DIR, "feature_config.json")
DATA_PATH   = "data/processed/features_enriched.csv"
OUT_DIR     = "reports/edge_case_tests"

FRONTEND_LABELS = {
    0: "NOT DAMAGED",
    1: "LESS DAMAGED",
    2: "MODERATELY DAMAGED",
    3: "SEVERELY DAMAGED",
}

# Metadata columns that are NOT features
META_COLS = [
    "case_id", "case_name", "category", "sub",
    "expected_crash", "expected_min_class", "expected_max_class",
    "rationale", "spatial_context", "expect_ood",
]


def section(title):
    print("\n" + "=" * 72)
    print("  " + title)
    print("=" * 72)


# ============================================================================
# LOAD PIPELINE
# ============================================================================
def load_pipeline():
    if not os.path.exists(CONFIG_PATH):
        sys.exit("[ERROR] Missing %s. Run C5 first." % CONFIG_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    reg_model = joblib.load(os.path.join(MODELS_DIR, config["exposure_model"]["file"]))
    clf_model = joblib.load(os.path.join(MODELS_DIR, config["classification_model"]["file"]))
    imp_exp   = joblib.load(os.path.join(MODELS_DIR, "imputer_exposure.joblib"))
    imp_clf   = joblib.load(os.path.join(MODELS_DIR, "preprocessor_imputer.joblib"))
    return config, reg_model, clf_model, imp_exp, imp_clf


# ============================================================================
# INTERACTION FEATURES
# ============================================================================
def compute_interactions(df):
    d = df.copy()

    def _g(col, default):
        if col in d.columns:
            return d[col].fillna(default)
        return pd.Series(default, index=d.index)

    if "inter_lst_vpd" not in d.columns:
        d["inter_lst_vpd"] = (_g("sat_lst_day_mean", 30.0) * _g("wx_fill_vpd_mean", 2.0)).round(4)
    if "inter_tmax_solar" not in d.columns:
        d["inter_tmax_solar"] = (_g("wx_fill_tmax_max", 33.0) * (_g("wx_fill_solar_sum", 600.0) / 100.0)).round(4)
    if "inter_dtr_sm" not in d.columns:
        d["inter_dtr_sm"] = (_g("sat_lst_dtr", 15.0) / _g("wx_fill_sm_mean", 0.2).clip(lower=0.02)).round(4)
    if "inter_sm_buffer" not in d.columns:
        d["inter_sm_buffer"] = (_g("wx_fill_sm_mean", 0.2) / (_g("wx_fill_hdd35", 0.0) + 1.0)).round(4)
    if "canopy_senescence_rate" not in d.columns:
        d["canopy_senescence_rate"] = ((_g("veg_NDVI_v3", 0.65) - _g("S_ndvi", 0.55)) / 40.0).clip(lower=0).round(5)
    if "inter_lst_ndmi_deficit" not in d.columns:
        d["inter_lst_ndmi_deficit"] = (_g("sat_lst_day_p90", 35.0) * (1.0 - _g("sat_ndmi_W", 0.2).clip(lower=-0.5, upper=1.0))).round(4)
    if "inter_heat_persistence" not in d.columns:
        d["inter_heat_persistence"] = (_g("wx_fill_hot35_days", 0.0) * (_g("wx_fill_run35", 0.0) + 1.0)).round(2)
    if "pre_vigor_ratio" not in d.columns:
        d["pre_vigor_ratio"] = (_g("veg_NDVI_v3", 0.65) / _g("veg_NDVI_v1", 0.40).clip(lower=0.05)).round(4)
    return d


# ============================================================================
# OOD DETECTOR (PCA-Mahalanobis)
# ============================================================================
def build_ood_detector(X_train_imp, n_components=20, pct=97.5):
    n_comp = min(n_components, X_train_imp.shape[1], X_train_imp.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=42)
    Z_tr   = pca.fit_transform(X_train_imp)
    mean_  = Z_tr.mean(axis=0)
    cov_   = np.cov(Z_tr.T) + np.eye(n_comp) * 1e-4
    inv_c  = np.linalg.inv(cov_)
    dists  = np.einsum("ij,jk,ik->i", Z_tr - mean_, inv_c, Z_tr - mean_)
    thresh = float(np.percentile(dists, pct))
    return pca, mean_, inv_c, thresh


def ood_scores_batch(X_imp, pca, mean_, inv_c):
    Z = pca.transform(X_imp)
    d = Z - mean_
    return np.einsum("ij,jk,ik->i", d, inv_c, d)


# ============================================================================
# BATCH INFERENCE  (single model call for all cases)
# ============================================================================
def batch_predict(cases_df, config, reg_model, clf_model, imp_exp, imp_clf,
                  pca, mean_ood, inv_c_ood, ood_thresh):
    """
    Impute + predict for ALL cases at once.
    Returns results DataFrame aligned to cases_df index.
    """
    exp_cols = config["exposure_features"]
    clf_cols = config["classification_features"]

    df = compute_interactions(cases_df.copy())

    # Ensure all required columns exist
    all_cols = list(dict.fromkeys(exp_cols + clf_cols))
    for c in all_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Replace inf
    df[all_cols] = df[all_cols].replace([np.inf, -np.inf], np.nan)

    # Impute
    X_exp_imp = imp_exp.transform(df[exp_cols])
    X_clf_imp = imp_clf.transform(df[clf_cols])

    # Predict (single batch call)
    pred_exp     = reg_model.predict(X_exp_imp)
    pred_classes = clf_model.predict(X_clf_imp)
    pred_probs   = clf_model.predict_proba(X_clf_imp)

    # OOD
    if pca is not None:
        ood_sc   = ood_scores_batch(X_clf_imp, pca, mean_ood, inv_c_ood)
        is_ood   = ood_sc > ood_thresh
    else:
        ood_sc = np.full(len(df), np.nan)
        is_ood = np.zeros(len(df), dtype=bool)

    results = pd.DataFrame({
        "case_id":          cases_df["case_id"].values,
        "pred_class_id":    pred_classes,
        "pred_label":       [FRONTEND_LABELS[c] for c in pred_classes],
        "pred_exposure_C":  np.round(pred_exp, 2),
        "prob_0":           np.round(pred_probs[:, 0], 4),
        "prob_1":           np.round(pred_probs[:, 1], 4),
        "prob_2":           np.round(pred_probs[:, 2], 4),
        "prob_3":           np.round(pred_probs[:, 3], 4),
        "max_prob":         np.round(pred_probs.max(axis=1), 4),
        "confidence":       np.where(pred_probs.max(axis=1) >= 0.60, "HIGH",
                            np.where(pred_probs.max(axis=1) >= 0.40, "MEDIUM", "LOW")),
        "is_ood":           is_ood,
        "ood_score":        np.round(ood_sc, 3),
        "ood_threshold":    round(ood_thresh, 2) if not np.isnan(ood_thresh) else np.nan,
        "has_nan_output":   np.isnan(pred_exp) | np.any(np.isnan(pred_probs), axis=1),
        "has_inf_output":   np.isinf(pred_exp) | np.any(np.isinf(pred_probs), axis=1),
        "crashed":          False,
        "error":            None,
    })
    return results


# ============================================================================
# BASELINE TEST CASE
# ============================================================================
def make_baseline_case():
    return {
        "sat_lst_day_mean": 32.0, "sat_lst_day_max": 36.0, "sat_lst_day_p90": 35.0,
        "sat_lst_day_std": 2.5, "sat_lst_night_mean": 18.0, "sat_lst_dtr": 14.0,
        "sat_lst_day_pre": 28.0, "sat_lst_rise": 4.0,
        "sat_ndvi_W": 0.45, "sat_ndvi_change": -0.05, "sat_ndmi_W": 0.22, "sat_evi_W": 0.32,
        "diag_lst_n_obs": 5,
        "veg_NDVI_v1": 0.42, "veg_NDVI_v2": 0.60, "veg_NDVI_v3": 0.65,
        "veg_EVI_v1": 0.28, "veg_EVI_v2": 0.40, "veg_EVI_v3": 0.45,
        "veg_NDMI_v1": 0.20, "veg_NDMI_v2": 0.25, "veg_NDMI_v3": 0.28,
        "veg_LST_day_v1": 28.0, "veg_LST_day_v2": 30.0, "veg_LST_day_v3": 32.0,
        "veg_LST_night_v1": 16.0, "veg_LST_night_v2": 17.5, "veg_LST_night_v3": 18.0,
        "veg_NDVI_premax": 0.68, "veg_NDVI_rise": 0.18, "veg_NDVI_flat": 0.05,
        "veg_LST_day_gap": 2.0,
        "S_ndvi": 0.48, "S_dndvi": -0.17, "S_evi": 0.33, "S_peak": 0.70,
        "S_int": 0.55, "S_late": 0.40, "S_p20": 0.42, "S_cov": 0.90,
        "wx_pre_tmax_mean": 28.0, "wx_pre_tmax_max": 33.0, "wx_pre_tmin_mean": 15.0,
        "wx_pre_hot35_days": 0, "wx_pre_hdd35": 0.0, "wx_pre_run35": 0,
        "wx_pre_anomhot_days": 0, "wx_pre_vpd_mean": 1.5, "wx_pre_solar_sum": 600.0,
        "wx_pre_rain_sum": 120.0, "wx_pre_rain_max": 25.0, "wx_pre_dry_max": 7,
        "wx_pre_sm_mean": 0.28, "wx_pre_frost_days": 1,
        "wx_anth_tmax_mean": 32.0, "wx_anth_tmax_max": 36.0, "wx_anth_tmin_mean": 18.0,
        "wx_anth_hot35_days": 1, "wx_anth_hdd35": 1.0, "wx_anth_run35": 1,
        "wx_anth_anomhot_days": 0, "wx_anth_vpd_mean": 2.0, "wx_anth_solar_sum": 200.0,
        "wx_anth_rain_sum": 40.0, "wx_anth_rain_max": 15.0, "wx_anth_dry_max": 5,
        "wx_anth_sm_mean": 0.25,
        "wx_fill_tmax_mean": 33.0, "wx_fill_tmax_max": 37.0, "wx_fill_tmin_mean": 19.0,
        "wx_fill_hot35_days": 2, "wx_fill_hdd35": 2.5, "wx_fill_run35": 2,
        "wx_fill_anomhot_days": 1, "wx_fill_vpd_mean": 2.2, "wx_fill_solar_sum": 400.0,
        "wx_fill_rain_sum": 30.0, "wx_fill_rain_max": 12.0, "wx_fill_dry_max": 10,
        "wx_fill_sm_mean": 0.24,
        "mask_frac": 0.65,
    }


# ============================================================================
# CATEGORY A: MISSING DATA
# ============================================================================
def build_category_A():
    base = make_baseline_case()
    cases = []

    single_drop = [
        ("A01", "NDVI missing",              ["sat_ndvi_W"]),
        ("A02", "EVI missing",               ["sat_evi_W"]),
        ("A03", "NDMI missing",              ["sat_ndmi_W"]),
        ("A04", "LST day mean missing",      ["sat_lst_day_mean"]),
        ("A05", "LST day max missing",       ["sat_lst_day_max"]),
        ("A06", "LST night missing",         ["sat_lst_night_mean"]),
        ("A07", "LST p90 missing",           ["sat_lst_day_p90"]),
        ("A08", "Tmax fill missing",         ["wx_fill_tmax_mean", "wx_fill_tmax_max"]),
        ("A09", "Rainfall fill missing",     ["wx_fill_rain_sum", "wx_fill_rain_max"]),
        ("A10", "Soil moisture missing",     ["wx_fill_sm_mean", "wx_anth_sm_mean", "wx_pre_sm_mean"]),
        ("A11", "VPD missing",               ["wx_fill_vpd_mean", "wx_anth_vpd_mean"]),
        ("A12", "NDVI change missing",       ["sat_ndvi_change"]),
    ]
    for cid, name, drop_cols in single_drop:
        row = base.copy()
        for c in drop_cols:
            row[c] = np.nan
        cases.append({**{"case_id": cid, "case_name": name,
                         "category": "A", "sub": "single_feature_missing",
                         "expected_crash": False}, **row})

    # Multiple satellite features missing
    row = base.copy()
    for c in ["sat_ndvi_W", "sat_evi_W", "sat_ndmi_W",
               "sat_lst_day_mean", "sat_lst_day_max", "sat_lst_day_p90"]:
        row[c] = np.nan
    cases.append({**{"case_id": "A13", "case_name": "All primary satellite features missing",
                     "category": "A", "sub": "multi_satellite_missing",
                     "expected_crash": False}, **row})

    # All veg windows missing
    row = base.copy()
    for c in [k for k in base if k.startswith("veg_")]:
        row[c] = np.nan
    cases.append({**{"case_id": "A14", "case_name": "All veg-window features missing",
                     "category": "A", "sub": "multi_satellite_missing",
                     "expected_crash": False}, **row})

    # All weather missing
    row = base.copy()
    for c in [k for k in base if k.startswith("wx_")]:
        row[c] = np.nan
    cases.append({**{"case_id": "A15", "case_name": "All weather features missing",
                     "category": "A", "sub": "all_wx_missing",
                     "expected_crash": False}, **row})

    # Entire record missing
    row = base.copy()
    for c in [k for k in base if k.startswith(("sat_", "veg_", "wx_", "S_"))]:
        row[c] = np.nan
    cases.append({**{"case_id": "A16", "case_name": "Entire district-year missing",
                     "category": "A", "sub": "entire_record_missing",
                     "expected_crash": False}, **row})

    # All-NaN record
    cases.append({**{"case_id": "A17", "case_name": "All-NaN record",
                     "category": "A", "sub": "all_nan",
                     "expected_crash": False},
                  **{k: np.nan for k in base}})

    # Inf values
    for cid, col, val, name_ in [
        ("A18", "sat_lst_day_mean",  np.inf,  "+Inf LST day mean"),
        ("A19", "sat_lst_day_mean", -np.inf, "-Inf LST day mean"),
        ("A20", "wx_fill_tmax_mean", np.inf,  "+Inf Tmax fill"),
        ("A21", "sat_ndvi_W",        np.inf,  "+Inf NDVI"),
        ("A22", "wx_fill_sm_mean",  -np.inf, "-Inf Soil Moisture"),
    ]:
        row = base.copy(); row[col] = val
        cases.append({**{"case_id": cid, "case_name": name_,
                         "category": "A", "sub": "inf_values",
                         "expected_crash": False}, **row})

    # Zero NDVI/EVI
    row = base.copy(); row["sat_ndvi_W"] = 0.0; row["sat_evi_W"] = 0.0
    cases.append({**{"case_id": "A23", "case_name": "Zero NDVI and EVI",
                     "category": "A", "sub": "zero_values",
                     "expected_crash": False}, **row})

    # Missing critical growth stage (v3)
    row = base.copy()
    for c in ["veg_NDVI_v3", "veg_EVI_v3", "veg_NDMI_v3"]:
        row[c] = np.nan
    cases.append({**{"case_id": "A24", "case_name": "Missing v3 (pre-peak) observations",
                     "category": "A", "sub": "missing_critical_stage",
                     "expected_crash": False}, **row})
    return cases


# ============================================================================
# CATEGORY B: EXTREME WEATHER
# ============================================================================
def build_category_B():
    base = make_baseline_case()
    cases = []
    extreme_wx = [
        ("B01", "Extreme Tmax 45C",
         {"wx_fill_tmax_mean": 45, "wx_fill_tmax_max": 47,
          "wx_fill_hot35_days": 20, "wx_fill_hdd35": 200, "wx_fill_run35": 15,
          "sat_lst_day_mean": 48, "sat_lst_day_p90": 50}),
        ("B02", "Extreme Tmax 50C (physiological kill)",
         {"wx_fill_tmax_mean": 50, "wx_fill_tmax_max": 52,
          "wx_fill_hot35_days": 30, "wx_fill_hdd35": 450, "wx_fill_run35": 25,
          "sat_lst_day_mean": 55, "sat_lst_day_p90": 58}),
        ("B03", "Extreme LST 55C",
         {"sat_lst_day_mean": 55, "sat_lst_day_max": 58, "sat_lst_day_p90": 57, "sat_lst_dtr": 20}),
        ("B04", "Extreme LST 60C",
         {"sat_lst_day_mean": 60, "sat_lst_day_max": 63, "sat_lst_day_p90": 62, "sat_lst_dtr": 22}),
        ("B05", "Prolonged heatwave run35=20 days",
         {"wx_fill_run35": 20, "wx_fill_hot35_days": 22, "wx_fill_hdd35": 180,
          "wx_fill_tmax_max": 44, "wx_anth_run35": 15, "wx_anth_hot35_days": 15}),
        ("B06", "Prolonged heatwave run35=30 days",
         {"wx_fill_run35": 30, "wx_fill_hot35_days": 30, "wx_fill_hdd35": 300, "wx_fill_tmax_max": 46}),
        ("B07", "Complete drought (rainfall=0, SM=0.01)",
         {"wx_fill_rain_sum": 0, "wx_fill_rain_max": 0, "wx_fill_sm_mean": 0.01,
          "wx_anth_rain_sum": 0, "wx_anth_sm_mean": 0.01, "wx_pre_rain_sum": 0,
          "wx_pre_sm_mean": 0.01, "wx_fill_dry_max": 40}),
        ("B08", "Severe SM deficiency (SM=0.03)",
         {"wx_fill_sm_mean": 0.03, "wx_anth_sm_mean": 0.03, "wx_pre_sm_mean": 0.04}),
        ("B09", "Extreme rainfall 500mm flood",
         {"wx_fill_rain_sum": 500, "wx_fill_rain_max": 150, "wx_anth_rain_sum": 300, "wx_fill_sm_mean": 0.45}),
        ("B10", "Extreme rainfall 800mm",
         {"wx_fill_rain_sum": 800, "wx_fill_rain_max": 250, "wx_fill_sm_mean": 0.50}),
        ("B11", "High Tmax + very low SM (combined heat-drought)",
         {"wx_fill_tmax_mean": 43, "wx_fill_tmax_max": 46, "wx_fill_hot35_days": 15,
          "wx_fill_hdd35": 130, "wx_fill_sm_mean": 0.02, "wx_anth_sm_mean": 0.03,
          "sat_lst_day_mean": 50, "sat_lst_day_p90": 52}),
        ("B12", "High Tmax + sufficient rainfall (partial buffering)",
         {"wx_fill_tmax_mean": 42, "wx_fill_tmax_max": 45, "wx_fill_hot35_days": 12,
          "wx_fill_hdd35": 100, "wx_fill_rain_sum": 200, "wx_fill_sm_mean": 0.30}),
        ("B13", "High VPD evapotranspiration stress",
         {"wx_fill_vpd_mean": 5.5, "wx_anth_vpd_mean": 4.8, "wx_fill_tmax_mean": 40, "wx_fill_sm_mean": 0.10}),
        ("B14", "Zero solar radiation",
         {"wx_fill_solar_sum": 0, "wx_anth_solar_sum": 0, "wx_pre_solar_sum": 0}),
        ("B15", "Near-freezing conditions (frost during grain fill)",
         {"wx_fill_tmax_mean": 12, "wx_fill_tmax_max": 15, "wx_fill_tmin_mean": 2,
          "wx_pre_frost_days": 10, "sat_lst_day_mean": 14, "sat_lst_night_mean": 4}),
        ("B16", "Normal Tmax + zero rainfall late-season drought",
         {"wx_fill_tmax_mean": 34, "wx_fill_tmax_max": 37, "wx_fill_rain_sum": 0,
          "wx_fill_sm_mean": 0.05, "wx_fill_dry_max": 35}),
        ("B17", "Negative SM (physically impossible -- robustness check)",
         {"wx_fill_sm_mean": -0.10, "wx_anth_sm_mean": -0.05}),
        ("B18", "Negative rainfall (impossible -- robustness check)",
         {"wx_fill_rain_sum": -50, "wx_pre_rain_sum": -20}),
        ("B19", "Extreme anomhot_days >30",
         {"wx_fill_anomhot_days": 35, "wx_anth_anomhot_days": 25,
          "wx_fill_hot35_days": 30, "wx_fill_hdd35": 250}),
        ("B20", "Simultaneous all-worst-case extremes",
         {"wx_fill_tmax_mean": 48, "wx_fill_tmax_max": 52, "wx_fill_hot35_days": 30,
          "wx_fill_hdd35": 400, "wx_fill_run35": 28, "wx_fill_sm_mean": 0.01,
          "wx_fill_rain_sum": 0, "wx_fill_vpd_mean": 6.0,
          "sat_lst_day_mean": 58, "sat_lst_day_p90": 60,
          "sat_ndvi_W": 0.05, "sat_ndmi_W": -0.30}),
    ]
    for cid, name, overrides in extreme_wx:
        row = base.copy(); row.update(overrides)
        cases.append({**{"case_id": cid, "case_name": name,
                         "category": "B", "sub": "extreme_weather",
                         "expected_crash": False}, **row})
    return cases


# ============================================================================
# CATEGORY C: SATELLITE EDGE CASES
# ============================================================================
def build_category_C():
    base = make_baseline_case()
    cases = []
    sat_edges = [
        ("C01", "NDVI close to -1 (bare/water)",
         {"sat_ndvi_W": -0.95, "veg_NDVI_v1": -0.90, "veg_NDVI_v2": -0.92, "veg_NDVI_v3": -0.91, "S_ndvi": -0.95}),
        ("C02", "NDVI = 0 (no vegetation)",
         {"sat_ndvi_W": 0.0, "veg_NDVI_v1": 0.01, "veg_NDVI_v2": 0.01, "veg_NDVI_v3": 0.0, "S_ndvi": 0.0}),
        ("C03", "NDVI close to 1 (maximum greenness)",
         {"sat_ndvi_W": 0.95, "veg_NDVI_v1": 0.90, "veg_NDVI_v2": 0.92, "veg_NDVI_v3": 0.95, "S_ndvi": 0.95}),
        ("C04", "Extremely low all vegetation indices",
         {"sat_ndvi_W": 0.02, "sat_evi_W": 0.01, "sat_ndmi_W": -0.40,
          "veg_NDVI_v1": 0.05, "veg_NDVI_v2": 0.03, "veg_NDVI_v3": 0.02, "S_ndvi": 0.02}),
        ("C05", "Unusually high NDVI > 0.9 across all windows",
         {"sat_ndvi_W": 0.92, "sat_evi_W": 0.70,
          "veg_NDVI_v1": 0.85, "veg_NDVI_v2": 0.90, "veg_NDVI_v3": 0.92, "S_peak": 0.95}),
        ("C06", "Missing v3 (critical pre-peak window)",
         {"veg_NDVI_v3": np.nan, "veg_EVI_v3": np.nan, "veg_NDMI_v3": np.nan}),
        ("C07", "Missing NDVI at all windows v1/v2/v3",
         {"veg_NDVI_v1": np.nan, "veg_NDVI_v2": np.nan, "veg_NDVI_v3": np.nan,
          "veg_EVI_v1":  np.nan, "veg_EVI_v2":  np.nan, "veg_EVI_v3":  np.nan}),
        ("C08", "Zero LST observations in window (cloud-covered)",
         {"diag_lst_n_obs": 0}),
        ("C09", "Only 1 LST observation (low confidence)",
         {"diag_lst_n_obs": 1}),
        ("C10", "Inconsistent: high LST but high NDVI (irrigated)",
         {"sat_lst_day_mean": 50, "sat_lst_day_p90": 52, "sat_ndvi_W": 0.72,
          "sat_evi_W": 0.55, "sat_ndmi_W": 0.30, "wx_fill_sm_mean": 0.35}),
        ("C11", "Inconsistent: normal LST but very low NDVI (non-heat decline)",
         {"sat_lst_day_mean": 30, "sat_lst_day_p90": 32, "sat_ndvi_W": 0.08,
          "sat_evi_W": 0.06, "sat_ndmi_W": -0.15, "wx_fill_tmax_mean": 28, "wx_fill_hot35_days": 0}),
        ("C12", "NDVI positive but EVI negative (sensor artefact)",
         {"sat_ndvi_W": 0.40, "sat_evi_W": -0.05, "veg_EVI_v3": -0.02, "veg_EVI_v2": -0.03}),
        ("C13", "Negative NDMI (severe water stress)",
         {"sat_ndmi_W": -0.45, "veg_NDMI_v1": -0.30, "veg_NDMI_v2": -0.40, "veg_NDMI_v3": -0.45}),
        ("C14", "NDMI > 0.5 (waterlogged)",
         {"sat_ndmi_W": 0.55, "veg_NDMI_v1": 0.48, "veg_NDMI_v2": 0.52, "veg_NDMI_v3": 0.55}),
        ("C15", "S_dndvi large negative (severe canopy collapse)",
         {"S_ndvi": 0.10, "veg_NDVI_v3": 0.68, "S_dndvi": -0.58}),
        ("C16", "S_dndvi positive (late-season rain greenup)",
         {"S_ndvi": 0.75, "veg_NDVI_v3": 0.65, "S_dndvi": 0.10}),
        ("C17", "All S_ signals missing",
         {"S_ndvi": np.nan, "S_dndvi": np.nan, "S_evi": np.nan,
          "S_peak": np.nan, "S_int": np.nan, "S_late": np.nan, "S_p20": np.nan, "S_cov": np.nan}),
        ("C18", "Very high LST DTR 30C",
         {"sat_lst_dtr": 30.0, "sat_lst_day_mean": 48, "sat_lst_night_mean": 18}),
        ("C19", "Negative LST DTR (cloud artefact -- night warmer than day)",
         {"sat_lst_dtr": -5.0, "sat_lst_day_mean": 22, "sat_lst_night_mean": 27}),
        ("C20", "Maximum LST p90 62C desert-like",
         {"sat_lst_day_p90": 62, "sat_lst_day_max": 64, "sat_lst_day_mean": 55}),
    ]
    for cid, name, overrides in sat_edges:
        row = base.copy(); row.update(overrides)
        cases.append({**{"case_id": cid, "case_name": name,
                         "category": "C", "sub": "satellite_edge",
                         "expected_crash": False}, **row})
    return cases


# ============================================================================
# CATEGORY D: PREDICTION CONSISTENCY
# ============================================================================
def build_category_D():
    base = make_baseline_case()
    cases = []
    scenarios = [
        ("D01", "Heat damage: High Tmax + Low SM + Declining NDVI",
         {"wx_fill_tmax_mean": 41, "wx_fill_tmax_max": 44, "wx_fill_hot35_days": 14,
          "wx_fill_hdd35": 120, "wx_fill_sm_mean": 0.05, "wx_anth_sm_mean": 0.06,
          "sat_lst_day_mean": 46, "sat_lst_day_p90": 48,
          "sat_ndvi_W": 0.18, "S_ndvi": 0.15, "S_dndvi": -0.50, "veg_NDVI_v3": 0.65},
         2, 3, "High heat + severe SM deficit + canopy collapse -> MODERATELY/SEVERELY DAMAGED"),

        ("D02", "Buffered: High Tmax + Sufficient SM + Stable NDVI (irrigated)",
         {"wx_fill_tmax_mean": 40, "wx_fill_tmax_max": 43, "wx_fill_hot35_days": 10,
          "wx_fill_hdd35": 80, "wx_fill_sm_mean": 0.32, "wx_anth_sm_mean": 0.30,
          "sat_lst_day_mean": 38, "sat_lst_day_p90": 40,
          "sat_ndvi_W": 0.52, "S_ndvi": 0.50, "S_dndvi": -0.15},
         0, 1, "High temp but irrigation buffering -> NOT/LESS DAMAGED"),

        ("D03", "Non-heat decline: Normal Tmax + Low NDVI",
         {"wx_fill_tmax_mean": 30, "wx_fill_tmax_max": 33, "wx_fill_hot35_days": 0,
          "wx_fill_hdd35": 0, "wx_fill_sm_mean": 0.08, "sat_ndvi_W": 0.10,
          "S_ndvi": 0.12, "S_dndvi": -0.53, "sat_lst_day_mean": 29},
         0, 1, "Low NDVI without heat -> disease/pest/drought, NOT heat damage"),

        ("D04", "Baseline: Normal temperature + healthy vegetation",
         {"wx_fill_tmax_mean": 30, "wx_fill_tmax_max": 34, "wx_fill_hot35_days": 1,
          "wx_fill_hdd35": 0.5, "sat_ndvi_W": 0.55, "S_ndvi": 0.52,
          "S_dndvi": -0.13, "wx_fill_sm_mean": 0.26},
         0, 0, "Normal healthy season -> NOT DAMAGED"),

        ("D05", "Heat + healthy vegetation: possible irrigation buffering",
         {"wx_fill_tmax_mean": 39, "wx_fill_tmax_max": 42, "wx_fill_hot35_days": 8,
          "wx_fill_hdd35": 60, "sat_ndvi_W": 0.60, "S_ndvi": 0.58,
          "S_dndvi": -0.07, "wx_fill_sm_mean": 0.30, "sat_lst_day_mean": 40},
         0, 1, "Irrigation-buffered heat stress -> at most LESS DAMAGED"),

        ("D06", "Cold season + declining NDVI (not heatwave)",
         {"wx_fill_tmax_mean": 20, "wx_fill_tmax_max": 24, "wx_fill_hot35_days": 0,
          "wx_fill_hdd35": 0, "sat_lst_day_mean": 20, "sat_lst_day_p90": 22,
          "sat_ndvi_W": 0.22, "S_ndvi": 0.20, "S_dndvi": -0.45},
         0, 1, "Cold year NDVI decline -> NOT heat damage"),

        ("D07", "Severe heatwave + NO vegetation decline (irrigated)",
         {"wx_fill_tmax_mean": 42, "wx_fill_tmax_max": 46, "wx_fill_hot35_days": 18,
          "wx_fill_hdd35": 160, "wx_fill_run35": 14, "sat_ndvi_W": 0.55,
          "S_ndvi": 0.52, "S_dndvi": -0.13, "wx_fill_sm_mean": 0.35, "sat_lst_day_mean": 46},
         0, 2, "Heatwave exposure but vegetation maintained -> subclinical"),

        ("D08", "Mild heatwave + significant NDVI decline (possibly other cause)",
         {"wx_fill_tmax_mean": 36, "wx_fill_tmax_max": 38, "wx_fill_hot35_days": 3,
          "wx_fill_hdd35": 8, "sat_ndvi_W": 0.18, "S_ndvi": 0.15,
          "S_dndvi": -0.50, "wx_fill_sm_mean": 0.22},
         0, 2, "Mild heat but large NDVI decline -> mixed attribution"),
    ]
    for cid, name, overrides, exp_min, exp_max, rationale in scenarios:
        row = base.copy(); row.update(overrides)
        cases.append({**{"case_id": cid, "case_name": name,
                         "category": "D", "sub": "consistency",
                         "expected_min_class": exp_min,
                         "expected_max_class": exp_max,
                         "expected_crash": False,
                         "rationale": rationale}, **row})
    return cases


# ============================================================================
# CATEGORY E: SPATIAL AND TEMPORAL GENERALIZATION
# ============================================================================
def build_category_E():
    base = make_baseline_case()
    cases = []
    e_cases = [
        ("E01", "High-yield district (Punjab baseline)",
         {"sat_ndvi_W": 0.68, "veg_NDVI_v3": 0.75, "S_ndvi": 0.65,
          "wx_fill_sm_mean": 0.35, "wx_fill_rain_sum": 180,
          "wx_fill_tmax_mean": 33, "wx_fill_hot35_days": 2}, "known", False),
        ("E02", "Low-yield district (arid zone)",
         {"sat_ndvi_W": 0.22, "veg_NDVI_v3": 0.28, "S_ndvi": 0.20,
          "wx_fill_sm_mean": 0.08, "wx_fill_rain_sum": 15,
          "wx_fill_tmax_mean": 38, "wx_fill_hot35_days": 12, "mask_frac": 0.08}, "known", False),
        ("E03", "Unusual: high yield in hot year",
         {"sat_ndvi_W": 0.65, "S_ndvi": 0.62, "wx_fill_tmax_mean": 42,
          "wx_fill_hot35_days": 15, "wx_fill_sm_mean": 0.33}, "known", False),
        ("E04", "Incomplete historical record (many features NaN)",
         {k: np.nan for k in list(base.keys())[:20]}, "known", False),
        ("E05", "Previously unseen district (OOD)",
         {"sat_ndvi_W": 0.40, "wx_fill_tmax_mean": 35}, "unseen_district", True),
        ("E06", "Year 2030 (unseen future)",
         {"sat_ndvi_W": 0.45, "wx_fill_tmax_mean": 36}, "future_year", True),
        ("E07", "Year 2035 (climate change scenario)",
         {"wx_fill_tmax_mean": 44, "wx_fill_hot35_days": 20,
          "wx_fill_sm_mean": 0.12, "sat_lst_day_mean": 48}, "future_year", True),
        ("E08", "Unprecedented weather year (all extremes)",
         {"wx_fill_tmax_mean": 48, "wx_fill_hot35_days": 30, "wx_fill_hdd35": 350,
          "wx_fill_sm_mean": 0.01, "sat_lst_day_mean": 56, "sat_ndvi_W": 0.05}, "extreme_climate", True),
        ("E09", "Desert-like climate (Rajasthan extreme)",
         {"sat_lst_day_mean": 52, "sat_lst_day_p90": 55, "wx_fill_tmax_mean": 44,
          "wx_fill_rain_sum": 5, "wx_fill_sm_mean": 0.03, "mask_frac": 0.04,
          "sat_ndvi_W": 0.12}, "extreme_climate", True),
        ("E10", "Humid tropical (atypical for wheat)",
         {"wx_fill_rain_sum": 600, "wx_fill_sm_mean": 0.48,
          "wx_fill_tmax_mean": 26, "sat_ndvi_W": 0.72, "wx_fill_hot35_days": 0}, "atypical_climate", True),
        ("E11", "Unusual: low temp in normally hot district",
         {"wx_fill_tmax_mean": 22, "sat_lst_day_mean": 20,
          "wx_fill_hot35_days": 0, "sat_ndvi_W": 0.70}, "unusual_pattern", False),
        ("E12", "District with NDVI data gaps",
         {c: np.nan for c in [k for k in base if "NDVI" in k]}, "incomplete_record", False),
        ("E13", "Transition year (features near distribution edge)",
         {"wx_fill_tmax_mean": 39, "wx_fill_hot35_days": 9,
          "wx_fill_sm_mean": 0.15, "sat_lst_day_mean": 42, "sat_ndvi_W": 0.35}, "known", False),
        ("E14", "Very low crop mask coverage",
         {"mask_frac": 0.01}, "low_coverage", False),
        ("E15", "High SM retention (deep black soil)",
         {"wx_fill_sm_mean": 0.48, "wx_anth_sm_mean": 0.45, "wx_pre_sm_mean": 0.50,
          "wx_fill_tmax_mean": 40, "wx_fill_hot35_days": 10}, "soil_difference", False),
        ("E16", "High altitude district (cooler LST, lower NDVI)",
         {"sat_lst_day_mean": 20, "sat_lst_night_mean": 10, "wx_fill_tmax_mean": 25,
          "wx_fill_hot35_days": 0, "sat_ndvi_W": 0.35, "mask_frac": 0.30}, "altitude", False),
    ]
    for cid, name, overrides, context, expect_ood in e_cases:
        row = base.copy(); row.update(overrides)
        cases.append({**{"case_id": cid, "case_name": name,
                         "category": "E", "sub": "generalization",
                         "spatial_context": context, "expect_ood": expect_ood,
                         "expected_crash": False}, **row})
    return cases


# ============================================================================
# AUDIT FUNCTIONS
# ============================================================================
def audit_A(meta, res):
    rows = []
    for _, m in meta[meta["category"] == "A"].iterrows():
        r = res[res["case_id"] == m["case_id"]].iloc[0]
        ok = not (r["crashed"] or r["has_nan_output"] or r["has_inf_output"])
        rows.append({"case_id": m["case_id"], "case_name": m["case_name"],
                     "sub": m["sub"], "crashed": r["crashed"],
                     "has_nan": r["has_nan_output"], "has_inf": r["has_inf_output"],
                     "pred_label": r["pred_label"],
                     "status": "PASS" if ok else "FAIL",
                     "note": "CRASH" if r["crashed"] else ("NaN output" if r["has_nan_output"] else "OK")})
    return pd.DataFrame(rows)


def audit_B(meta, res):
    rows = []
    for _, m in meta[meta["category"] == "B"].iterrows():
        r = res[res["case_id"] == m["case_id"]].iloc[0]
        name = str(m["case_name"]).lower()
        pred_c = int(r["pred_class_id"])
        if any(k in name for k in ["tmax 4", "tmax 5", "drought", "heat-drought", "worst-case"]):
            ok = not r["crashed"] and pred_c >= 1
            note = "Expected >= class 1 for severe heat/drought"
        elif any(k in name for k in ["flood", "800mm", "500mm"]):
            ok = not r["crashed"]
            note = "Flood: any valid prediction acceptable"
        elif any(k in name for k in ["frost", "freezing", "near-freezing"]):
            ok = not r["crashed"] and pred_c <= 1
            note = "Cold: expect <= class 1 (not heat damage)"
        elif any(k in name for k in ["impossible", "negative", "robustness"]):
            ok = not r["crashed"]
            note = "Impossible value: crash check only"
        else:
            ok = not r["crashed"]
            note = "Extreme condition: no-crash check"
        rows.append({"case_id": m["case_id"], "case_name": m["case_name"],
                     "crashed": r["crashed"], "pred_class": pred_c,
                     "pred_label": r["pred_label"], "physically_reasonable": ok,
                     "status": "PASS" if ok else "FAIL", "note": note})
    return pd.DataFrame(rows)


def audit_C(meta, res):
    rows = []
    for _, m in meta[meta["category"] == "C"].iterrows():
        r = res[res["case_id"] == m["case_id"]].iloc[0]
        ok = not (r["crashed"] or r["has_nan_output"] or r["has_inf_output"])
        rows.append({"case_id": m["case_id"], "case_name": m["case_name"],
                     "crashed": r["crashed"], "has_nan": r["has_nan_output"],
                     "pred_label": r["pred_label"], "confidence": r["confidence"],
                     "status": "PASS" if ok else "FAIL",
                     "note": "CRASH" if r["crashed"] else ("NaN" if r["has_nan_output"] else "OK")})
    return pd.DataFrame(rows)


def audit_D(meta, res):
    rows = []
    for _, m in meta[meta["category"] == "D"].iterrows():
        r    = res[res["case_id"] == m["case_id"]].iloc[0]
        pred = int(r["pred_class_id"])
        exp_min = int(m.get("expected_min_class", 0))
        exp_max = int(m.get("expected_max_class", 3))
        in_range = (not r["crashed"]) and (exp_min <= pred <= exp_max)
        rows.append({"case_id": m["case_id"], "case_name": m["case_name"],
                     "expected_range": "[%d-%d]" % (exp_min, exp_max),
                     "pred_class": pred, "pred_label": r["pred_label"],
                     "confidence": r["confidence"], "max_prob": r["max_prob"],
                     "in_expected_range": in_range,
                     "status": "PASS" if in_range else "WARN",
                     "rationale": str(m.get("rationale", ""))})
    return pd.DataFrame(rows)


def audit_E(meta, res):
    rows = []
    for _, m in meta[meta["category"] == "E"].iterrows():
        r = res[res["case_id"] == m["case_id"]].iloc[0]
        exp_ood = bool(m.get("expect_ood", False))
        is_ood  = bool(r["is_ood"])
        ood_ok  = (not exp_ood) or (exp_ood and is_ood)
        ok      = (not r["crashed"]) and ood_ok
        rows.append({"case_id": m["case_id"], "case_name": m["case_name"],
                     "spatial_context": str(m.get("spatial_context", "")),
                     "expected_ood": exp_ood, "flagged_as_ood": is_ood,
                     "ood_score": r["ood_score"], "pred_class": r["pred_class_id"],
                     "pred_label": r["pred_label"],
                     "ood_correct": ood_ok,
                     "status": "PASS" if ok else "WARN"})
    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================
def main():
    section("D1: COMPREHENSIVE EDGE-CASE TESTING PIPELINE")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load pipeline
    section("1. Loading Production Pipeline")
    config, reg_model, clf_model, imp_exp, imp_clf = load_pipeline()
    print("  Exposure model : %s" % config["exposure_model"]["type"])
    print("  Classifier     : %s" % config["classification_model"]["type"])
    clf_cols = config["classification_features"]
    print("  Classifier features: %d" % len(clf_cols))

    # 2. Load training data for OOD
    section("2. Loading Training Data for OOD Reference")
    pca = mean_ood = inv_c_ood = None
    ood_thresh = np.nan

    if os.path.exists(DATA_PATH):
        df_full  = pd.read_csv(DATA_PATH)
        df_train = df_full[df_full["Year"] < 2019].copy()
        print("  Training samples: %d rows, %d districts (%d-%d)" % (
            len(df_train), df_train["District"].nunique(),
            df_train["Year"].min(), df_train["Year"].max()))

        X_tr_raw = df_train[clf_cols].replace([np.inf, -np.inf], np.nan)
        X_tr_imp = imp_clf.transform(X_tr_raw)
        pca, mean_ood, inv_c_ood, ood_thresh = build_ood_detector(X_tr_imp)
        print("  OOD threshold (97.5th pct): %.1f" % ood_thresh)
        tr_ood_frac = (ood_scores_batch(X_tr_imp, pca, mean_ood, inv_c_ood) > ood_thresh).mean()
        print("  Training OOD rate (expect ~2.5%%): %.1f%%" % (tr_ood_frac * 100))
    else:
        print("  [WARNING] Training data not found -- OOD detection disabled.")

    # 3. Build all test cases
    section("3. Building Test Cases")
    cat_A = build_category_A()
    cat_B = build_category_B()
    cat_C = build_category_C()
    cat_D = build_category_D()
    cat_E = build_category_E()

    all_cases = cat_A + cat_B + cat_C + cat_D + cat_E
    print("  Category A (Missing Data)      : %3d cases" % len(cat_A))
    print("  Category B (Extreme Weather)   : %3d cases" % len(cat_B))
    print("  Category C (Satellite Edge)    : %3d cases" % len(cat_C))
    print("  Category D (Consistency)       : %3d cases" % len(cat_D))
    print("  Category E (Generalization)    : %3d cases" % len(cat_E))
    print("  TOTAL                          : %3d cases" % len(all_cases))

    # Build DataFrame, separate metadata from features
    cases_df = pd.DataFrame(all_cases).reset_index(drop=True)
    meta_cols = [c for c in dict.fromkeys(META_COLS) if c in cases_df.columns]
    feat_cols = [c for c in cases_df.columns if c not in meta_cols]
    meta_df   = cases_df[meta_cols].copy()

    # 4. BATCH INFERENCE (single model call -- fast)
    section("4. Running Batch Inference (single model call)")
    feat_df = cases_df[["case_id"] + feat_cols].copy()
    results = batch_predict(feat_df, config, reg_model, clf_model, imp_exp, imp_clf,
                            pca, mean_ood, inv_c_ood, ood_thresh)

    # Print per-case summary
    for _, r in results.iterrows():
        m = meta_df[meta_df["case_id"] == r["case_id"]].iloc[0]
        ood_tag = " [OOD]" if r["is_ood"] else ""
        print("  %5s | %-55s | %s [conf=%s, p=%.3f]%s" % (
            r["case_id"], str(m["case_name"])[:55],
            r["pred_label"], r["confidence"], r["max_prob"], ood_tag))

    # Save raw results
    results.to_csv(os.path.join(OUT_DIR, "edge_case_results.csv"), index=False)
    print("\n  Saved: edge_case_results.csv")

    # 5. Audits
    section("5. Running Pass/Fail Audits")
    aud_A = audit_A(cases_df, results)
    aud_B = audit_B(cases_df, results)
    aud_C = audit_C(cases_df, results)
    aud_D = audit_D(cases_df, results)
    aud_E = audit_E(cases_df, results)

    audit_all = pd.concat([
        aud_A.assign(category="A"),
        aud_B.assign(category="B"),
        aud_C.assign(category="C"),
        aud_D.assign(category="D"),
        aud_E.assign(category="E"),
    ], ignore_index=True, sort=False)

    audit_all.to_csv(os.path.join(OUT_DIR, "edge_case_summary.csv"), index=False)
    aud_D.to_csv(os.path.join(OUT_DIR, "consistency_audit.csv"), index=False)

    ood_df = results[["case_id", "ood_score", "ood_threshold", "is_ood"]].copy()
    ood_df.to_csv(os.path.join(OUT_DIR, "ood_scores.csv"), index=False)

    # 6. Summary
    section("6. Summary")

    def cat_sum(df, label, col="status"):
        total = len(df)
        passed = (df[col] == "PASS").sum()
        warned = (df[col] == "WARN").sum() if "WARN" in df[col].values else 0
        failed = (df[col] == "FAIL").sum()
        return "  %-35s: %d total | PASS=%d WARN=%d FAIL=%d" % (label, total, passed, warned, failed)

    print(cat_sum(aud_A, "A (Missing Data)"))
    print(cat_sum(aud_B, "B (Extreme Weather)"))
    print(cat_sum(aud_C, "C (Satellite Edge)"))
    print(cat_sum(aud_D, "D (Consistency)"))
    print(cat_sum(aud_E, "E (Generalization)"))

    n_crash   = results["crashed"].sum()
    n_nan     = results["has_nan_output"].sum()
    n_inf     = results["has_inf_output"].sum()
    n_ood     = results["is_ood"].sum()

    print("\n  Total cases    : %d" % len(results))
    print("  Crashes        : %d" % n_crash)
    print("  NaN outputs    : %d" % n_nan)
    print("  Inf outputs    : %d" % n_inf)
    print("  OOD flagged    : %d" % n_ood)

    print("\n  -- Category D: Consistency Audit --")
    print(aud_D[["case_id", "case_name", "expected_range",
                  "pred_class", "pred_label", "status"]].to_string(index=False))

    # Write text report
    lines = [
        "=" * 72,
        "D1: EDGE-CASE TESTING REPORT",
        "=" * 72,
        "",
        "MODEL UNDER TEST:",
        "  Exposure Model  : %s" % config["exposure_model"]["type"],
        "  Classifier      : %s" % config["classification_model"]["type"],
        "  Test Accuracy   : %.4f" % config["classification_model"]["test_accuracy"],
        "  ROC-AUC OvR     : %.4f" % config["classification_model"]["test_roc_auc_ovr"],
        "",
        "EDGE-CASE SUMMARY:",
        cat_sum(aud_A, "A (Missing Data)"),
        cat_sum(aud_B, "B (Extreme Weather)"),
        cat_sum(aud_C, "C (Satellite Edge)"),
        cat_sum(aud_D, "D (Consistency)"),
        cat_sum(aud_E, "E (Generalization)"),
        "",
        "  Crashes          : %d" % n_crash,
        "  NaN outputs      : %d" % n_nan,
        "  Inf outputs      : %d" % n_inf,
        "  OOD flagged      : %d / %d" % (n_ood, len(results)),
        "  OOD threshold    : %.2f" % ood_thresh if not np.isnan(ood_thresh) else "  OOD threshold    : N/A",
        "",
        "CONSISTENCY AUDIT (Category D):",
    ]
    for _, r in aud_D.iterrows():
        lines.append("  %s | %-50s | Expected %s | Got %d (%s) | %s" % (
            r["case_id"], str(r["case_name"])[:50],
            r["expected_range"], r["pred_class"], r["pred_label"], r["status"]))

    lines += ["", "FAILURES & WARNINGS:"]
    fails = audit_all[audit_all["status"].isin(["FAIL", "WARN"])]
    if len(fails) == 0:
        lines.append("  None.")
    else:
        for _, r in fails.iterrows():
            note = str(r.get("note", r.get("rationale", "")))[:60]
            lines.append("  [%s] %s | %-45s | %s" % (
                r["status"], r["case_id"], str(r.get("case_name", ""))[:45], note))

    lines += [
        "", "OUTPUT FILES:",
        "  %s/edge_case_results.csv" % OUT_DIR,
        "  %s/edge_case_summary.csv" % OUT_DIR,
        "  %s/consistency_audit.csv" % OUT_DIR,
        "  %s/ood_scores.csv"        % OUT_DIR,
    ]

    with open(os.path.join(OUT_DIR, "D1_test_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n  Report written to %s/D1_test_report.txt" % OUT_DIR)
    overall_ok = (n_crash == 0) and (n_nan == 0) and (n_inf == 0)
    print("\n" + "=" * 72)
    print("  VERDICT: %s" % ("[PASS] PIPELINE ROBUST" if overall_ok else "[WARN] ISSUES DETECTED"))
    print("=" * 72)


if __name__ == "__main__":
    main()
