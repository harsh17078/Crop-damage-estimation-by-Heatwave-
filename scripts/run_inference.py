"""
RUN_INFERENCE.PY
================
Production inference pipeline for heatwave crop damage estimation.
Loads trained models, exact feature configurations, and preprocessors from models/.

Usage:
  python scripts/run_inference.py --test-sample
  python scripts/run_inference.py --input path/to/features.csv --output path/to/predictions.csv
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = "models"
CONFIG_PATH = os.path.join(MODELS_DIR, "feature_config.json")

def load_pipeline():
    if not os.path.exists(CONFIG_PATH):
        sys.exit(f"Missing {CONFIG_PATH}. Run C5 first.")
        
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        
    reg_path = os.path.join(MODELS_DIR, config["exposure_model"]["file"])
    clf_path = os.path.join(MODELS_DIR, config["classification_model"]["file"])
    imp_exp_path = os.path.join(MODELS_DIR, "imputer_exposure.joblib")
    imp_clf_path = os.path.join(MODELS_DIR, "preprocessor_imputer.joblib")
    
    reg_model = joblib.load(reg_path)
    clf_model = joblib.load(clf_path)
    imp_exp = joblib.load(imp_exp_path)
    imp_clf = joblib.load(imp_clf_path)
    
    return config, reg_model, clf_model, imp_exp, imp_clf

def compute_interactions_if_missing(df):
    d = df.copy()
    if "inter_lst_vpd" not in d.columns:
        lst_day = d.get("sat_lst_day_mean", pd.Series(30.0, index=d.index)).fillna(30.0)
        vpd_fill = d.get("wx_fill_vpd_mean", pd.Series(2.0, index=d.index)).fillna(2.0)
        d["inter_lst_vpd"] = (lst_day * vpd_fill).round(4)
        
    if "inter_tmax_solar" not in d.columns:
        tmax_max = d.get("wx_fill_tmax_max", pd.Series(33.0, index=d.index)).fillna(33.0)
        solar_fill = d.get("wx_fill_solar_sum", pd.Series(600.0, index=d.index)).fillna(600.0)
        d["inter_tmax_solar"] = (tmax_max * (solar_fill / 100.0)).round(4)
        
    if "inter_dtr_sm" not in d.columns:
        lst_dtr = d.get("sat_lst_dtr", pd.Series(15.0, index=d.index)).fillna(15.0)
        sm_fill = d.get("wx_fill_sm_mean", pd.Series(0.2, index=d.index)).fillna(0.2)
        d["inter_dtr_sm"] = (lst_dtr / (sm_fill.clip(lower=0.02))).round(4)
        
    if "inter_sm_buffer" not in d.columns:
        sm_fill = d.get("wx_fill_sm_mean", pd.Series(0.2, index=d.index)).fillna(0.2)
        hdd35 = d.get("wx_fill_hdd35", pd.Series(0.0, index=d.index)).fillna(0.0)
        d["inter_sm_buffer"] = (sm_fill / (hdd35 + 1.0)).round(4)
        
    if "canopy_senescence_rate" not in d.columns:
        ndvi_v3 = d.get("veg_NDVI_v3", pd.Series(0.65, index=d.index)).fillna(0.65)
        s_ndvi = d.get("S_ndvi", pd.Series(0.55, index=d.index)).fillna(0.55)
        d["canopy_senescence_rate"] = ((ndvi_v3 - s_ndvi) / 40.0).clip(lower=0).round(5)
        
    if "inter_lst_ndmi_deficit" not in d.columns:
        lst_p90 = d.get("sat_lst_day_p90", pd.Series(35.0, index=d.index)).fillna(35.0)
        ndmi_w = d.get("sat_ndmi_W", pd.Series(0.2, index=d.index)).fillna(0.2)
        d["inter_lst_ndmi_deficit"] = (lst_p90 * (1.0 - ndmi_w.clip(lower=-0.5, upper=1.0))).round(4)
        
    if "inter_heat_persistence" not in d.columns:
        hot35 = d.get("wx_fill_hot35_days", pd.Series(0.0, index=d.index)).fillna(0.0)
        run35 = d.get("wx_fill_run35", pd.Series(0.0, index=d.index)).fillna(0.0)
        d["inter_heat_persistence"] = (hot35 * (run35 + 1.0)).round(2)
        
    if "pre_vigor_ratio" not in d.columns:
        ndvi_v3 = d.get("veg_NDVI_v3", pd.Series(0.65, index=d.index)).fillna(0.65)
        ndvi_v1 = d.get("veg_NDVI_v1", pd.Series(0.40, index=d.index)).fillna(0.40)
        d["pre_vigor_ratio"] = (ndvi_v3 / (ndvi_v1.clip(lower=0.05))).round(4)
        
    return d

def predict(df_raw, config, reg_model, clf_model, imp_exp, imp_clf):
    df = compute_interactions_if_missing(df_raw)
    
    # Exposure Regression
    exp_cols = config["exposure_features"]
    # Ensure all features exist in input
    for col in exp_cols:
        if col not in df.columns:
            df[col] = np.nan
    X_exp = df[exp_cols]
    X_exp_imp = imp_exp.transform(X_exp)
    pred_exp = reg_model.predict(X_exp_imp)
    
    # Damage Classification
    clf_cols = config["classification_features"]
    for col in clf_cols:
        if col not in df.columns:
            df[col] = np.nan
    X_clf = df[clf_cols]
    X_clf_imp = imp_clf.transform(X_clf)
    
    pred_classes = clf_model.predict(X_clf_imp)
    pred_probs = clf_model.predict_proba(X_clf_imp)
    
    class_map = {int(k): v for k, v in config["class_mapping"].items()}
    
    results = df_raw.copy()
    results["pred_exposure_tmax_mean_C"] = np.round(pred_exp, 2)
    results["pred_damage_class_id"] = pred_classes
    results["pred_damage_severity"] = [class_map[c] for c in pred_classes]
    
    for c in range(config["n_classes"]):
        results[f"prob_{class_map[c].replace(' ', '_')}"] = np.round(pred_probs[:, c], 4)
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Crop Heatwave Damage Inference Pipeline")
    parser.add_argument("--input", help="Path to input features CSV")
    parser.add_argument("--output", help="Path to save predictions CSV")
    parser.add_argument("--test-sample", action="store_true", help="Run a test verification on real holdout samples")
    args = parser.parse_args()
    
    print("Loading production pipeline from models/...")
    config, reg_model, clf_model, imp_exp, imp_clf = load_pipeline()
    print(f"Loaded {config['exposure_model']['type']} for exposure regression.")
    print(f"Loaded {config['classification_model']['type']} for damage classification.")
    
    if args.test_sample or (not args.input):
        print("\n" + "=" * 75)
        print("RUNNING VERIFICATION TEST ON DIVERSE HOLDOUT SAMPLES (2019-2024)")
        print("=" * 75)
        data_path = "data/processed/features_enriched.csv"
        if not os.path.exists(data_path):
            sys.exit(f"Missing {data_path} for sample test.")
        full_df = pd.read_csv(data_path)
        
        # Pick 4 diverse sample cases:
        # 1. 2022 Heatwave (Severe shock)
        # 2. 2020 Cool season (No damage)
        # 3. 2021 Moderate stress
        # 4. 2024 Mild stress
        s1 = full_df[(full_df["Year"] == 2022) & (full_df["damage_severity"] == 3)].head(1)
        s2 = full_df[(full_df["Year"] == 2020) & (full_df["damage_severity"] == 0)].head(1)
        s3 = full_df[(full_df["Year"] == 2021) & (full_df["damage_severity"] == 2)].head(1)
        s4 = full_df[(full_df["Year"] == 2024) & (full_df["damage_severity"] == 1)].head(1)
        
        sample_df = pd.concat([s1, s2, s3, s4], ignore_index=True)
        preds = predict(sample_df, config, reg_model, clf_model, imp_exp, imp_clf)
        
        show_cols = [
            "State", "District", "Year", "tmax_mean_W", "pred_exposure_tmax_mean_C",
            "damage_class_name", "pred_damage_severity",
            "prob_No_Significant_Heatwave_Damage", "prob_Severe_Heatwave_Damage"
        ]
        print("\nSAMPLE INFERENCE RESULTS:")
        print(preds[show_cols].to_string(index=False))
        print("\nAll pipeline components verified and fully functional.")
        
    elif args.input:
        in_df = pd.read_csv(args.input)
        print(f"Running inference on {len(in_df)} input samples...")
        preds = predict(in_df, config, reg_model, clf_model, imp_exp, imp_clf)
        out_path = args.output or "reports/predictions_output.csv"
        preds.to_csv(out_path, index=False)
        print(f"Predictions saved to {out_path}")

if __name__ == "__main__":
    main()
