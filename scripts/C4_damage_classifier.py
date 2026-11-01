"""
C4: MULTI-CLASS HEATWAVE CROP DAMAGE CLASSIFIER (PHASE 4)
=========================================================
Determines whether wheat crops suffered heatwave-induced damage:
  Class 0: No Significant Heatwave Damage
  Class 1: Mild Heatwave Damage
  Class 2: Moderate Heatwave Damage
  Class 3: Severe Heatwave Damage

Algorithms Evaluated:
  - Baseline (Stratified Prior / Most Frequent)
  - Balanced Multinomial Logistic Regression
  - Balanced Random Forest Classifier
  - Balanced Extra Trees Classifier
  - HistGradientBoosting Classifier (with class-weighted loss)
  - Gradient Boosting Classifier

Optimization & Objectives:
  - Macro F1, Weighted F1, Balanced Accuracy, Accuracy, Multi-class ROC-AUC (OvR)
  - Class-wise Precision, Recall, and F1
  - Minimizing False Negatives on Damaged Crops (Classes 1, 2, 3)
  - Full Confusion Matrices
  - Zero data leakage via rolling-origin validation (Train < 2019, Test >= 2019)

Outputs:
  reports/classification/classification_benchmark_all.csv
  reports/classification/class_wise_performance.csv
  reports/classification/confusion_matrices/
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, GradientBoostingClassifier)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             roc_auc_score, cohen_kappa_score)

IN_PATH = "data/processed/features_enriched.csv"
OUT_DIR = "reports/classification"
CM_DIR = os.path.join(OUT_DIR, "confusion_matrices")
SPLIT_YEAR = 2019
SEED = 42

CLASS_NAMES = [
    "No Significant Damage",
    "Mild Damage",
    "Moderate Damage",
    "Severe Damage"
]

def make_classifiers():
    return {
        "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=SEED),
        "LogisticRegression (Balanced)": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
        "RandomForest (Balanced)": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                                           class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "ExtraTrees (Balanced)": ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2,
                                                       class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=5, learning_rate=0.04, max_iter=250,
                                                               min_samples_leaf=12, l2_regularization=2.0,
                                                               class_weight="balanced", random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=180, max_depth=4, learning_rate=0.04,
                                                       subsample=0.85, random_state=SEED)
    }

def main():
    print("=" * 75)
    print("C4: MULTI-CLASS HEATWAVE CROP DAMAGE CLASSIFICATION (PHASE 4)")
    print("=" * 75)
    
    if not os.path.exists(IN_PATH):
        sys.exit(f"Missing {IN_PATH}. Run C2 first.")
        
    df = pd.read_csv(IN_PATH)
    target_col = "damage_severity"
    
    # Identify feature columns
    leakage_cols = [
        "damage_severity", "damage_class_name", "damage_reason", "yield_z", "logy", "Yield",
        "n35", "hdd35", "tmax_mean_W", "HS1", "HS2", "HS3", "HS5", "n35_clean", "hdd35_clean",
        "State", "District", "Year", "P_doy", "P_fallback", "excess_rain_shock", "rain_fill_p90",
        "dndvi_z", "ndmi_z", "lst_p90_z"
    ]
    feature_cols = [c for c in df.columns if c not in leakage_cols]
    
    print(f"Total rows: {len(df)} | Features: {len(feature_cols)}")
    print("Class counts overall:", df[target_col].value_counts().sort_index().to_dict())
    
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CM_DIR, exist_ok=True)
    
    test_years = sorted([y for y in df["Year"].unique() if y >= SPLIT_YEAR])
    print(f"Rolling evaluation over test years ({len(test_years)}): {test_years}")
    
    models = make_classifiers()
    benchmark_rows = []
    class_wise_rows = []
    
    for name, model in models.items():
        preds_all = []
        actuals_all = []
        probs_all = []
        
        for yr in test_years:
            tr = df[df["Year"] < yr]
            te = df[df["Year"] == yr]
            if len(tr) < 80 or len(te) == 0:
                continue
                
            X_tr = tr[feature_cols].copy()
            y_tr = tr[target_col].to_numpy(int)
            X_te = te[feature_cols].copy()
            y_te = te[target_col].to_numpy(int)
            
            # Leakage-free preprocessing strictly within fold
            imputer = SimpleImputer(strategy="median").fit(X_tr)
            X_tr_imp = imputer.transform(X_tr)
            X_te_imp = imputer.transform(X_te)
            
            scaler = StandardScaler().fit(X_tr_imp)
            X_tr_sc = scaler.transform(X_tr_imp)
            X_te_sc = scaler.transform(X_te_imp)
            
            if name.startswith("LogisticRegression"):
                model.fit(X_tr_sc, y_tr)
                p = model.predict(X_te_sc)
                prob = model.predict_proba(X_te_sc)
            else:
                model.fit(X_tr_imp, y_tr)
                p = model.predict(X_te_imp)
                prob = model.predict_proba(X_te_imp)
                
            preds_all.extend(p)
            actuals_all.extend(y_te)
            probs_all.extend(prob)
            
        y_true = np.array(actuals_all)
        y_pred = np.array(preds_all)
        y_prob = np.array(probs_all)
        
        # Calculate Comprehensive Metrics
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        # Multi-class ROC AUC (OvR)
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc = np.nan
            
        # Severe & Moderate Damage False Negative Rate (Critical Objective)
        # Any sample with true class 2 or 3 predicted as class 0
        severe_mod_mask = (y_true >= 2)
        fn_severe_mod = np.sum((y_pred == 0) & severe_mod_mask)
        fn_rate = fn_severe_mod / np.sum(severe_mod_mask) if np.sum(severe_mod_mask) > 0 else 0.0
        
        # Severe class recall
        sev_mask = (y_true == 3)
        sev_recall = np.sum((y_pred == 3) & sev_mask) / np.sum(sev_mask) if np.sum(sev_mask) > 0 else 0.0
        
        benchmark_rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Balanced_Accuracy": round(bal_acc, 4),
            "Cohen_Kappa": round(kappa, 4),
            "Macro_F1": round(macro_f1, 4),
            "Weighted_F1": round(weighted_f1, 4),
            "ROC_AUC_OvR": round(auc, 4),
            "Severe_Recall": round(sev_recall, 4),
            "Severe_Mod_FN_Rate": round(fn_rate, 4)
        })
        
        # Class-wise performance
        prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for c in range(4):
            class_wise_rows.append({
                "Model": name,
                "Class_ID": c,
                "Class_Name": CLASS_NAMES[c],
                "Support": int(np.sum(y_true == c)),
                "Precision": round(prec_per_class[c], 4) if c < len(prec_per_class) else 0.0,
                "Recall": round(rec_per_class[c], 4) if c < len(rec_per_class) else 0.0,
                "F1": round(f1_per_class[c], 4) if c < len(f1_per_class) else 0.0
            })
            
        # Save Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASS_NAMES],
                             columns=[f"Pred_{c}" for c in CLASS_NAMES])
        safe_name = name.split(" ")[0].lower()
        cm_df.to_csv(os.path.join(CM_DIR, f"cm_{safe_name}.csv"))
        
    bench_df = pd.DataFrame(benchmark_rows).sort_values("Macro_F1", ascending=False)
    bench_df.to_csv(os.path.join(OUT_DIR, "classification_benchmark_all.csv"), index=False)
    
    class_df = pd.DataFrame(class_wise_rows)
    class_df.to_csv(os.path.join(OUT_DIR, "class_wise_performance.csv"), index=False)
    
    print("\n" + "=" * 75)
    print("OVERALL CLASSIFIER BENCHMARK (TEST YEARS 2019-2024, N=222):")
    print("=" * 75)
    print(bench_df.to_string(index=False))
    
    top_model = bench_df.iloc[0]["Model"]
    print(f"\nTop Model: {top_model}")
    print("\nClass-Wise Performance for Top Models:")
    print(class_df[class_df["Model"] == top_model][["Class_Name", "Support", "Precision", "Recall", "F1"]].to_string(index=False))
    
    # Print Confusion Matrix of the Top Model
    top_safe = top_model.split(" ")[0].lower()
    top_cm = pd.read_csv(os.path.join(CM_DIR, f"cm_{top_safe}.csv"), index_col=0)
    print("\nConfusion Matrix for Top Model:")
    print(top_cm.to_string())

if __name__ == "__main__":
    main()
