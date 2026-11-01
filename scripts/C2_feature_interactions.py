"""
C2: PHYSIOLOGICAL & REMOTE SENSING FEATURE INTERACTIONS
======================================================
Engineers domain-grounded interaction terms capturing:
  1. Thermal-radiative stress (LST x VPD, Tmax x Solar)
  2. Moisture buffering / Evaporative cooling (LST DTR / Soil Moisture, SM / (HDD35 + 1))
  3. Canopy desiccation dynamics (NDVI drop rate, LST x NDMI deficit)
  4. Heat persistence factor (hot35_days x run35)

Input : data/processed/damage_ground_truth.csv
Output: data/processed/features_enriched.csv
        reports/damage_ground_truth/feature_correlation_ranking.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IN_PATH = "data/processed/damage_ground_truth.csv"
OUT_PATH = "data/processed/features_enriched.csv"
REPORT_DIR = "reports/damage_ground_truth"

def main():
    print("=" * 75)
    print("C2: PHYSIOLOGICAL & REMOTE SENSING FEATURE INTERACTIONS")
    print("=" * 75)
    
    if not os.path.exists(IN_PATH):
        sys.exit(f"Missing {IN_PATH}. Run C1 first.")
        
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} rows from {IN_PATH}")
    
    # Fill safe defaults for numerical interaction stability
    lst_day = df["sat_lst_day_mean"].fillna(df["sat_lst_day_mean"].median())
    lst_p90 = df["sat_lst_day_p90"].fillna(df["sat_lst_day_p90"].median())
    lst_dtr = df["sat_lst_dtr"].fillna(df["sat_lst_dtr"].median())
    vpd_fill = df["wx_fill_vpd_mean"].fillna(df["wx_fill_vpd_mean"].median())
    solar_fill = df["wx_fill_solar_sum"].fillna(df["wx_fill_solar_sum"].median())
    tmax_max = df["wx_fill_tmax_max"].fillna(df["wx_fill_tmax_max"].median())
    sm_fill = df["wx_fill_sm_mean"].fillna(df["wx_fill_sm_mean"].median())
    hdd35 = df["wx_fill_hdd35"].fillna(0)
    hot35 = df["wx_fill_hot35_days"].fillna(0)
    run35 = df["wx_fill_run35"].fillna(0)
    ndmi_w = df["sat_ndmi_W"].fillna(df["sat_ndmi_W"].median())
    ndvi_v3 = df["veg_NDVI_v3"].fillna(df["veg_NDVI_v3"].median())
    ndvi_v1 = df["veg_NDVI_v1"].fillna(df["veg_NDVI_v1"].median())
    s_ndvi = df["S_ndvi"].fillna(df["S_ndvi"].median())
    
    # 1. Thermal-radiative loading
    df["inter_lst_vpd"] = (lst_day * vpd_fill).round(4)
    df["inter_tmax_solar"] = (tmax_max * (solar_fill / 100.0)).round(4)
    
    # 2. Moisture buffering and evaporative cooling
    # High DTR combined with low soil moisture signifies lack of evaporative cooling
    df["inter_dtr_sm"] = (lst_dtr / (sm_fill.clip(lower=0.02))).round(4)
    # Soil moisture buffer ratio against accumulated degree days
    df["inter_sm_buffer"] = (sm_fill / (hdd35 + 1.0)).round(4)
    
    # 3. Canopy desiccation dynamics
    # Accelerated senescence: rate of greenness loss per unit time
    df["canopy_senescence_rate"] = ((ndvi_v3 - s_ndvi) / 40.0).clip(lower=0).round(5)
    # Thermal extreme weighted by canopy moisture deficit
    df["inter_lst_ndmi_deficit"] = (lst_p90 * (1.0 - ndmi_w.clip(lower=-0.5, upper=1.0))).round(4)
    
    # 4. Heatwave persistence (consecutive duration multiplier)
    df["inter_heat_persistence"] = (hot35 * (run35 + 1.0)).round(2)
    
    # 5. Pre-anthesis canopy buffer
    df["pre_vigor_ratio"] = (ndvi_v3 / (ndvi_v1.clip(lower=0.05))).round(4)
    
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved enriched feature dataset: {OUT_PATH} ({len(df)} rows, {df.shape[1]} columns)")
    
    # Evaluate rank correlations of newly engineered features with Damage Severity
    new_features = [
        "inter_lst_vpd", "inter_tmax_solar", "inter_dtr_sm", "inter_sm_buffer",
        "canopy_senescence_rate", "inter_lst_ndmi_deficit", "inter_heat_persistence", "pre_vigor_ratio"
    ]
    
    corrs = []
    for feat in new_features:
        valid = df[[feat, "damage_severity"]].dropna()
        if len(valid) > 50:
            rho, pval = spearmanr(valid[feat], valid["damage_severity"])
            corrs.append({"feature": feat, "spearman_rho_with_damage": round(rho, 4), "p_value": pval})
            
    corr_df = pd.DataFrame(corrs).sort_values("spearman_rho_with_damage", ascending=False)
    corr_df.to_csv(os.path.join(REPORT_DIR, "feature_correlation_ranking.csv"), index=False)
    
    print("\nCORRELATION OF NEW INTERACTION FEATURES WITH DAMAGE SEVERITY:")
    print(corr_df.to_string(index=False))
    print("\nFeature interaction engineering completed successfully.")

if __name__ == "__main__":
    main()
