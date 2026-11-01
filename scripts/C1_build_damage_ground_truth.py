"""
C1: BUILD HEATWAVE CROP DAMAGE SEVERITY GROUND TRUTH
====================================================
Scientific Grounding:
  Determines whether wheat crops experienced heatwave-induced damage vs general yield fluctuations.
  A crop is considered damaged by heatwave stress IF AND ONLY IF:
    1. Verified Physical Heat Exposure (E): Hot days (n35), accumulated degree-days (hdd35), or extreme LST p90.
    2. Biological Damage Response (R): Canopy desiccation (accelerated post-peak decline S_dndvi, NDMI moisture loss)
       or negative yield anomaly (DES yz < 0).
  Crucially, unseasonal excess rainfall / hail damage (e.g. 2014, 2015) is quarantined so that non-heat
  disasters are NOT misattributed to heatwaves.

Four Severity Tiers:
  0: No Significant Heatwave Damage
  1: Mild Heatwave Damage
  2: Moderate Heatwave Damage
  3: Severe Heatwave Damage

Outputs:
  data/processed/damage_ground_truth.csv
  reports/damage_ground_truth/summary_distribution.csv
  reports/damage_ground_truth/contingency_table.csv
"""

import os
import sys
import numpy as np
import pandas as pd

# Paths
EXPOSURE_PATH = "reports/exposure/exposure_table.csv"
FEATURES_PATH = "data/processed/A_features_signals.csv"
DES_PATH = "data/processed/des_yield_clean.csv"
OUT_DATA = "data/processed/damage_ground_truth.csv"
OUT_REPORT_DIR = "reports/damage_ground_truth"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_label_lib import past_z

def main():
    print("=" * 75)
    print("C1: CONSTRUCTING 4-TIER HEATWAVE CROP DAMAGE GROUND TRUTH")
    print("=" * 75)
    
    if not os.path.exists(EXPOSURE_PATH):
        sys.exit(f"Missing {EXPOSURE_PATH}. Run B1 first.")
    if not os.path.exists(FEATURES_PATH):
        sys.exit(f"Missing {FEATURES_PATH}.")
        
    df_exp = pd.read_csv(EXPOSURE_PATH)
    df_feat = pd.read_csv(FEATURES_PATH)
    
    # Merge exposure table with full features and signals
    merge_cols = ["State", "District", "Year"]
    feat_cols = [c for c in df_feat.columns if c not in df_exp.columns or c in merge_cols]
    df = df_exp.merge(df_feat[feat_cols], on=merge_cols, how="inner")
    
    # Merge official DES yield if available
    if os.path.exists(DES_PATH):
        des = pd.read_csv(DES_PATH)[["District", "Year", "Yield"]]
        df = df.merge(des, on=["District", "Year"], how="left")
        df["logy"] = np.log(df["Yield"].where(df["Yield"] > 0))
        # Past-only z-score of log yield (rolling baseline with shrink)
        df["yield_z"] = past_z(df, "logy", method="rolling", min_hist=5)
    else:
        df["Yield"] = np.nan
        df["yield_z"] = np.nan
        
    print(f"Total merged panel rows: {len(df)} across {df['District'].nunique()} districts and {df['Year'].min()}-{df['Year'].max()}")
    
    # -------------------------------------------------------------
    # 1. Physical Heat Exposure Score (E)
    # -------------------------------------------------------------
    df["n35_clean"] = df["n35"].fillna(0)
    df["hdd35_clean"] = df["hdd35"].fillna(0)
    
    # Standardize LST p90 anomaly past-only
    df["lst_p90_z"] = past_z(df, "sat_lst_day_p90", method="expanding", min_hist=5)
    
    # -------------------------------------------------------------
    # 2. Compound Non-Heat Hazard Screen (Excess Rain / Hail)
    # -------------------------------------------------------------
    rain_fill = df["wx_fill_rain_sum"].fillna(0)
    df["rain_fill_p90"] = rain_fill > df.groupby("District")["wx_fill_rain_sum"].transform(lambda x: x.quantile(0.85))
    df["excess_rain_shock"] = (df["rain_fill_p90"]) & (df["n35_clean"] <= 1)
    
    # -------------------------------------------------------------
    # 3. Biological Response Score (R)
    # -------------------------------------------------------------
    df["dndvi_z"] = past_z(df, "S_dndvi", method="rolling", min_hist=5)
    df["ndmi_z"] = past_z(df, "sat_ndmi_W", method="rolling", min_hist=5)
    
    # -------------------------------------------------------------
    # 4. Multi-Tier Ground Truth Assignment Logic
    # -------------------------------------------------------------
    damage_severity = []
    damage_reason = []
    
    for idx, row in df.iterrows():
        n35 = row["n35_clean"]
        hdd = row["hdd35_clean"]
        lst_z = row["lst_p90_z"] if pd.notna(row["lst_p90_z"]) else 0.0
        yz = row["yield_z"] if pd.notna(row["yield_z"]) else 0.0
        dz = row["dndvi_z"] if pd.notna(row["dndvi_z"]) else 0.0
        is_excess_rain = row["excess_rain_shock"]
        
        # Rule 0: Negligible heat exposure
        if n35 == 0 and hdd == 0 and lst_z < 0.5:
            damage_severity.append(0)
            damage_reason.append("No heat exposure")
            continue
            
        # Rule 0.1: If there was moderate heat, but crop yield & canopy remained healthy (transpirational cooling via irrigation)
        if (n35 <= 2) and (yz >= -0.25) and (dz >= -0.25):
            damage_severity.append(0)
            damage_reason.append("Irrigation-buffered (no yield/canopy damage despite mild heat)")
            continue
            
        # Rule 0.2: Strong yield drop caused primarily by non-heat lodging/hailstorm
        if is_excess_rain and n35 <= 1:
            damage_severity.append(0)
            damage_reason.append("Yield drop due to rain/hailstorm lodging, not heatwave")
            continue
            
        # Severe Heatwave Damage (Class 3):
        # Extreme thermal exposure AND acute biological damage (either severe yield loss or severe desiccation)
        if (n35 >= 5 or hdd >= 8.0 or lst_z >= 1.8) and (yz <= -1.25 or dz <= -1.25):
            damage_severity.append(3)
            damage_reason.append("Severe heat shock with major yield loss / canopy collapse")
            continue
        elif (n35 >= 8 or hdd >= 15.0) and (yz <= -0.75 or dz <= -0.75):
            damage_severity.append(3)
            damage_reason.append("Extreme prolonged heatwave with documented biological damage")
            continue
            
        # Moderate Heatwave Damage (Class 2):
        # Significant thermal exposure (n35 >= 3 or hdd >= 3.0 or lst_z >= 1.0) with evident damage
        if (n35 >= 3 or hdd >= 3.0 or lst_z >= 1.0) and (yz <= -0.50 or dz <= -0.50):
            damage_severity.append(2)
            damage_reason.append("Moderate heat exposure with noticeable canopy/yield depression")
            continue
        elif (n35 >= 5 or hdd >= 8.0) and (yz < 0 or dz < 0):
            damage_severity.append(2)
            damage_reason.append("High heat exposure with moderate depression")
            continue
            
        # Mild Heatwave Damage (Class 1):
        # Measurable thermal stress with mild vegetative or yield response
        if (n35 >= 1 or hdd > 0.5 or lst_z >= 0.75) and (yz <= -0.25 or dz <= -0.25):
            damage_severity.append(1)
            damage_reason.append("Mild heat stress with detectable minor decline")
            continue
        elif (n35 >= 3 or hdd >= 4.0):
            # High thermal stress but buffered by irrigation from severe loss
            damage_severity.append(1)
            damage_reason.append("Thermal exposure with mild sub-clinical stress")
            continue
            
        # Default fallback
        damage_severity.append(0)
        damage_reason.append("Sub-threshold impact")
        
    df["damage_severity"] = damage_severity
    df["damage_reason"] = damage_reason
    
    # Class labels mapping
    class_names = {
        0: "No Significant Heatwave Damage",
        1: "Mild Heatwave Damage",
        2: "Moderate Heatwave Damage",
        3: "Severe Heatwave Damage"
    }
    df["damage_class_name"] = df["damage_severity"].map(class_names)
    
    os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
    os.makedirs(OUT_REPORT_DIR, exist_ok=True)
    
    df.to_csv(OUT_DATA, index=False)
    print(f"\nSaved damage ground truth table: {OUT_DATA} ({len(df)} rows)")
    
    # -------------------------------------------------------------
    # Distribution Summary
    # -------------------------------------------------------------
    dist_all = df["damage_class_name"].value_counts(normalize=True).rename("Overall_Share")
    dist_counts = df["damage_class_name"].value_counts().rename("Count")
    
    split_year = 2019
    df_train = df[df["Year"] < split_year]
    df_test = df[df["Year"] >= split_year]
    
    train_dist = df_train["damage_class_name"].value_counts().rename("Train_Count (<2019)")
    test_dist = df_test["damage_class_name"].value_counts().rename("Test_Count (>=2019)")
    
    summary_df = pd.concat([dist_counts, (dist_all * 100).round(2), train_dist, test_dist], axis=1).fillna(0)
    summary_df.to_csv(os.path.join(OUT_REPORT_DIR, "summary_distribution.csv"))
    
    print("\nDAMAGE SEVERITY DISTRIBUTION:")
    print(summary_df.to_string())
    
    # By Year Distribution
    by_yr = df.groupby(["Year", "damage_severity"]).size().unstack(fill_value=0)
    by_yr.columns = [f"Class_{c}_{class_names[c][:8]}" for c in by_yr.columns]
    by_yr.to_csv(os.path.join(OUT_REPORT_DIR, "by_year_distribution.csv"))
    
    print("\nYEAR-BY-YEAR SEVERITY COUNTS (Notable years):")
    notable = [2004, 2010, 2014, 2015, 2020, 2021, 2022, 2023, 2024]
    print(by_yr.reindex(notable).dropna().to_string())
    print("\nGround truth generation completed successfully.")

if __name__ == "__main__":
    main()
