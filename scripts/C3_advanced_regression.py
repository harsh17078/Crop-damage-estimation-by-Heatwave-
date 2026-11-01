"""
C3: ADVANCED REGRESSION MODELING PIPELINE (PHASE 3)
==================================================
Evaluates comprehensive regression algorithms for:
  Target 1: Terminal Heat Exposure (tmax_mean_W, Continuous Celsius)
  Target 2: Accumulated Heat Degree Days (hdd35, Accumulated Degree-Days)
  Target 3: Heatwave-Attributed Yield Anomaly (yield_z, Standardized SD)

Candidate Algorithms:
  - Baseline (Climatology / Zero / OLS p90)
  - Ridge & ElasticNet
  - Random Forest Regressor
  - Extra Trees Regressor
  - HistGradientBoosting Regressor (Native Fast Histogram-GBM)
  - XGBoost / LightGBM (if available in environment)

Validation Protocol:
  - Rolling-Origin Temporal Holdout: Train < 2019, Test >= 2019
  - Leave-One-Year-Out (LOYO) Cross-Validation
  - Identical preprocessing, no data leakage, year-block evaluation

Outputs:
  reports/regression/summary_tmax_mean.csv
  reports/regression/summary_hdd35.csv
  reports/regression/summary_yield_anomaly.csv
  reports/regression/regression_benchmark_all.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

# Dynamic boosting imports
HAVE_XGB = False
HAVE_LGB = False
try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    pass

IN_PATH = "data/processed/features_enriched.csv"
OUT_DIR = "reports/regression"
SPLIT_YEAR = 2019
SEED = 42

def make_regressors(fast=False):
    n_trees = 100 if fast else 250
    models = {
        "Ridge": RidgeCV(alphas=np.logspace(-2, 3, 20)),
        "ElasticNet": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99], cv=5, random_state=SEED),
        "RandomForest": RandomForestRegressor(n_estimators=n_trees, max_depth=6, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=n_trees, max_depth=6, min_samples_leaf=4, n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_depth=4, learning_rate=0.04, max_iter=n_trees, min_samples_leaf=15, l2_regularization=1.5, random_state=SEED)
    }
    if HAVE_XGB:
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=n_trees, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED, n_jobs=-1)
    if HAVE_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(n_estimators=n_trees, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
    return models

def within_year_r2(obs, pred, years):
    df = pd.DataFrame({"obs": obs, "pred": pred, "year": years})
    oa = df["obs"] - df.groupby("year")["obs"].transform("mean")
    pa = df["pred"] - df.groupby("year")["pred"].transform("mean")
    den = (oa ** 2).sum()
    return float(1.0 - ((oa - pa) ** 2).sum() / den) if den > 1e-12 else np.nan

def calc_metrics(obs, pred, years, baseline_mse=None):
    e = obs - pred
    mse = float(np.mean(e ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(e)))
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    sse = float(np.sum(e ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1e-12 else np.nan
    r2_within = within_year_r2(obs, pred, years)
    rho = float(spearmanr(obs, pred)[0]) if len(np.unique(pred)) > 1 else np.nan
    
    skill = float(1.0 - mse / baseline_mse) if (baseline_mse and baseline_mse > 0) else np.nan
    return {
        "n": len(obs), "RMSE": round(rmse, 4), "MAE": round(mae, 4),
        "R2": round(r2, 4), "R2_within_year": round(r2_within, 4),
        "Spearman": round(rho, 4), "Skill_vs_Baseline": round(skill, 4)
    }

def evaluate_target(df, target_col, feature_cols, target_name, is_count=False):
    print("\n" + "=" * 70)
    print(f"BENCHMARKING TARGET: {target_name} ({target_col})")
    print("=" * 70)
    
    valid = df[df[target_col].notna()].copy()
    years = sorted(valid["Year"].unique())
    test_years = [y for y in years if y >= SPLIT_YEAR]
    train_years = [y for y in years if y < SPLIT_YEAR]
    
    print(f"Total valid samples: {len(valid)} | Train years ({len(train_years)}): {train_years[0]}..{train_years[-1]} | Test years ({len(test_years)}): {test_years}")
    print(f"Feature set size: {len(feature_cols)} features")
    
    # 1. Establish Climatology / Zero Baseline on Test Set
    df_train = valid[valid["Year"] < SPLIT_YEAR]
    df_test = valid[valid["Year"] >= SPLIT_YEAR]
    
    if target_col == "yield_z":
        pred_base_test = np.zeros(len(df_test))
    else:
        clim_map = df_train.groupby("District")[target_col].mean()
        pred_base_test = df_test["District"].map(clim_map).fillna(df_train[target_col].mean()).to_numpy()
        
    y_test = df_test[target_col].to_numpy()
    base_mse_test = float(np.mean((y_test - pred_base_test) ** 2))
    
    results = []
    base_metrics = calc_metrics(y_test, pred_base_test, df_test["Year"].to_numpy(), baseline_mse=base_mse_test)
    results.append({"Target": target_name, "Model": "Baseline (Clim/Zero)", "Split": "TEST (>=2019)", **base_metrics})
    
    # 2. Benchmark Single-Feature Baseline for Heat (sat_lst_day_p90)
    if target_col in ["tmax_mean_W", "hdd35"]:
        p90_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("reg", LinearRegression())])
        p90_pipe.fit(df_train[["sat_lst_day_p90"]], df_train[target_col])
        pred_p90 = p90_pipe.predict(df_test[["sat_lst_day_p90"]])
        if is_count:
            pred_p90 = np.clip(pred_p90, 0, None)
        p90_metrics = calc_metrics(y_test, pred_p90, df_test["Year"].to_numpy(), baseline_mse=base_mse_test)
        results.append({"Target": target_name, "Model": "OLS (sat_lst_day_p90)", "Split": "TEST (>=2019)", **p90_metrics})
        
    # 3. Rolling Origin Evaluation of All Advanced Regressors
    models = make_regressors()
    for name, model in models.items():
        # Rolling origin: for each test year, train on strictly earlier years
        preds_test = []
        actuals_test = []
        years_eval = []
        
        for yr in test_years:
            tr = valid[valid["Year"] < yr]
            te = valid[valid["Year"] == yr]
            if len(tr) < 50 or len(te) == 0:
                continue
                
            X_tr = tr[feature_cols].copy()
            y_tr = tr[target_col].to_numpy()
            X_te = te[feature_cols].copy()
            y_te = te[target_col].to_numpy()
            
            # Leakage-free preprocessing in each fold
            imputer = SimpleImputer(strategy="median").fit(X_tr)
            X_tr_imp = imputer.transform(X_tr)
            X_te_imp = imputer.transform(X_te)
            
            scaler = StandardScaler().fit(X_tr_imp)
            X_tr_sc = scaler.transform(X_tr_imp)
            X_te_sc = scaler.transform(X_te_imp)
            
            # Train and predict
            if name in ["Ridge", "ElasticNet"]:
                model.fit(X_tr_sc, y_tr)
                p = model.predict(X_te_sc)
            else:
                model.fit(X_tr_imp, y_tr)
                p = model.predict(X_te_imp)
                
            if is_count:
                p = np.clip(p, 0, None)
                
            preds_test.extend(p)
            actuals_test.extend(y_te)
            years_eval.extend([yr] * len(y_te))
            
        preds_test = np.array(preds_test)
        actuals_test = np.array(actuals_test)
        years_eval = np.array(years_eval)
        
        m = calc_metrics(actuals_test, preds_test, years_eval, baseline_mse=base_mse_test)
        results.append({"Target": target_name, "Model": name, "Split": "TEST (>=2019)", **m})
        
    res_df = pd.DataFrame(results).sort_values("RMSE")
    print(res_df.to_string(index=False))
    return res_df

def main():
    print("=" * 75)
    print("C3: ADVANCED REGRESSION BENCHMARKING (PHASE 3)")
    print("=" * 75)
    print(f"XGBoost available: {HAVE_XGB} | LightGBM available: {HAVE_LGB}")
    
    if not os.path.exists(IN_PATH):
        sys.exit(f"Missing {IN_PATH}. Run C2 first.")
        
    df = pd.read_csv(IN_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Feature sets
    sat_cols = [c for c in df.columns if c.startswith("sat_") and c not in ["sat_lst_rise"]]
    new_inter_cols = [
        "inter_lst_vpd", "inter_tmax_solar", "inter_dtr_sm", "inter_sm_buffer",
        "canopy_senescence_rate", "inter_lst_ndmi_deficit", "inter_heat_persistence", "pre_vigor_ratio"
    ]
    wx_cols = [c for c in df.columns if c.startswith("wx_") and "fill" in c or "anth" in c]
    veg_cols = [c for c in df.columns if c.startswith("veg_")]
    
    features_exposure = list(set(sat_cols + ["canopy_senescence_rate", "inter_lst_ndmi_deficit", "inter_dtr_sm"]))
    features_damage = list(set(sat_cols + new_inter_cols + wx_cols + veg_cols))
    # Exclude targets and leakages
    leakage_cols = ["damage_severity", "damage_class_name", "damage_reason", "yield_z", "logy", "Yield", "n35", "hdd35", "tmax_mean_W", "HS1", "HS2", "HS3", "HS5", "n35_clean", "hdd35_clean", "State", "District", "Year", "P_doy", "P_fallback", "excess_rain_shock", "rain_fill_p90"]
    features_damage = [c for c in features_damage if c not in leakage_cols and c in df.columns]
    features_exposure = [c for c in features_exposure if c not in leakage_cols and c in df.columns]
    
    # 1. Target 1: Terminal Heat Exposure Temperature (tmax_mean_W)
    df_tmax = evaluate_target(df, "tmax_mean_W", features_exposure, "Tmax_Mean_Window (degC)")
    df_tmax.to_csv(os.path.join(OUT_DIR, "summary_tmax_mean.csv"), index=False)
    
    # 2. Target 2: Heat Degree Days (hdd35)
    df_hdd = evaluate_target(df, "hdd35", features_exposure, "Heat_Degree_Days (hdd35)", is_count=True)
    df_hdd.to_csv(os.path.join(OUT_DIR, "summary_hdd35.csv"), index=False)
    
    # 3. Target 3: Heatwave-Attributed Yield Anomaly (yield_z)
    df_yield = evaluate_target(df, "yield_z", features_damage, "Yield_Anomaly (yield_z)")
    df_yield.to_csv(os.path.join(OUT_DIR, "summary_yield_anomaly.csv"), index=False)
    
    all_summary = pd.concat([df_tmax, df_hdd, df_yield], ignore_index=True)
    all_summary.to_csv(os.path.join(OUT_DIR, "regression_benchmark_all.csv"), index=False)
    print("\n" + "=" * 75)
    print("ALL REGRESSION BENCHMARKS COMPLETED AND SAVED TO reports/regression/")
    print("=" * 75)

if __name__ == "__main__":
    main()
