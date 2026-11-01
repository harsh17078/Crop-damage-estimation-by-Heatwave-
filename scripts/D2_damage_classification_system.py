"""
D2: CROP DAMAGE CLASSIFICATION SYSTEM (PHASE 3)
================================================
Validates the frontend-ready 4-class damage classification with:
  - Frontend label mapping (NOT DAMAGED / LESS / MODERATELY / SEVERELY DAMAGED)
  - Confidence scoring (HIGH / MEDIUM / LOW)
  - Out-of-Distribution (OOD) detection via Mahalanobis distance
  - Rule-based verification layer (warning flags, not hard overrides)
  - Holdout validation on 2019–2024 test years
  - Calibration curve analysis
  - Edge-case handling verification

Outputs:
  reports/classification_system/damage_classification_report.md
  reports/classification_system/holdout_validation.csv
  reports/classification_system/calibration_curve.csv
  reports/classification_system/confusion_matrix_frontend.csv
  reports/classification_system/per_class_metrics.csv
  models/damage_classifier_v2.py

Usage:
  python scripts/D2_damage_classification_system.py
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_auc_score, cohen_kappa_score, classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "data")):
    os.chdir(ROOT)

MODELS_DIR  = "models"
CONFIG_PATH = os.path.join(MODELS_DIR, "feature_config.json")
DATA_PATH   = "data/processed/features_enriched.csv"
OUT_DIR     = "reports/classification_system"
SPLIT_YEAR  = 2019

# ── Label Mapping ─────────────────────────────────────────────────────────────
FRONTEND_LABELS = {
    0: "NOT DAMAGED",
    1: "LESS DAMAGED",
    2: "MODERATELY DAMAGED",
    3: "SEVERELY DAMAGED",
}
INTERNAL_LABELS = {
    0: "No Significant Heatwave Damage",
    1: "Mild Heatwave Damage",
    2: "Moderate Heatwave Damage",
    3: "Severe Heatwave Damage",
}
DAMAGE_COLORS = {  # For reporting
    0: "🟢",
    1: "🟡",
    2: "🟠",
    3: "🔴",
}


def section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ── Load Pipeline ─────────────────────────────────────────────────────────────
def load_pipeline():
    if not os.path.exists(CONFIG_PATH):
        sys.exit(f"[ERROR] Missing {CONFIG_PATH}. Run C5 first.")

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    clf_model = joblib.load(os.path.join(MODELS_DIR, config["classification_model"]["file"]))
    imp_clf   = joblib.load(os.path.join(MODELS_DIR, "preprocessor_imputer.joblib"))

    return config, clf_model, imp_clf


# ── OOD Detector ─────────────────────────────────────────────────────────────
def build_ood_detector_pca(X_train_imp, n_components=20, threshold_percentile=97.5):
    """
    PCA-based Mahalanobis OOD detector.
    More stable than full-feature Mahalanobis for high-dim data.
    """
    n_comp = min(n_components, X_train_imp.shape[1], X_train_imp.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=42)
    Z_tr   = pca.fit_transform(X_train_imp)

    mean_  = Z_tr.mean(axis=0)
    cov_   = np.cov(Z_tr.T) + np.eye(n_comp) * 1e-4
    inv_c  = np.linalg.inv(cov_)

    # Compute training distances
    dists  = np.array([float((z - mean_) @ inv_c @ (z - mean_)) for z in Z_tr])
    thresh = float(np.percentile(dists, threshold_percentile))

    return pca, mean_, inv_c, thresh


def compute_ood_scores(X_imp, pca, mean_, inv_c):
    Z = pca.transform(X_imp)
    return np.array([float((z - mean_) @ inv_c @ (z - mean_)) for z in Z])


# ── Rule-Based Verification Layer ─────────────────────────────────────────────
# These rules add WARNING FLAGS; they never hard-override the model.
RULE_CHECKS = [
    {
        "name": "RULE_01_SEVERE_NO_HEAT",
        "description": "Predicted SEVERELY DAMAGED but no measurable heat exposure",
        "condition": lambda row, pred_cls: (
            pred_cls == 3
            and row.get("wx_fill_hot35_days", np.nan) == 0
            and row.get("wx_anth_hot35_days", np.nan) == 0
            and row.get("sat_lst_day_p90", np.nan) is not None
            and not np.isnan(row.get("sat_lst_day_p90", np.nan))
            and row.get("sat_lst_day_p90", 40) < 35
        ),
        "warning": "⚠️ Model predicts SEVERELY DAMAGED but no hot days detected. "
                   "Check for data quality issues or non-heat causes.",
        "suggested_action": "Review satellite data for cloud contamination",
    },
    {
        "name": "RULE_02_NO_DAMAGE_HIGH_HEAT",
        "description": "Predicted NOT DAMAGED but extreme heat + moisture deficit",
        "condition": lambda row, pred_cls: (
            pred_cls == 0
            and row.get("wx_fill_hot35_days", 0) >= 15
            and row.get("wx_fill_sm_mean", 0.5) < 0.05
            and row.get("sat_lst_day_p90", 30) > 45
        ),
        "warning": "⚠️ Model predicts NOT DAMAGED but extreme heat + severe SM deficit. "
                   "Possible data inconsistency or strong irrigation buffering.",
        "suggested_action": "Verify irrigation records for this district-year",
    },
    {
        "name": "RULE_03_HIGH_UNCERTAINTY",
        "description": "Very low confidence (max_prob < 0.35)",
        "condition": lambda row, pred_cls: False,  # handled separately via max_prob
        "warning": "⚠️ Very low prediction confidence. Result should be treated as uncertain.",
        "suggested_action": "Consider this an inconclusive result",
    },
    {
        "name": "RULE_04_IMPOSSIBLE_LST",
        "description": "LST values physically impossible (>65°C or <-10°C)",
        "condition": lambda row, pred_cls: (
            row.get("sat_lst_day_mean", 30) > 65
            or row.get("sat_lst_day_mean", 30) < -10
        ),
        "warning": "⚠️ LST values outside physically plausible range. "
                   "Possible cloud contamination or sensor artefact.",
        "suggested_action": "Recheck satellite QA flags for this composite",
    },
    {
        "name": "RULE_05_NEGATIVE_NDVI_DAMAGE",
        "description": "NDVI < 0 — non-vegetation surface",
        "condition": lambda row, pred_cls: (
            row.get("sat_ndvi_W", 0.4) < 0
            and pred_cls >= 2
        ),
        "warning": "⚠️ NDVI < 0 suggests bare soil/water. "
                   "Damage classification may not apply to this land cover.",
        "suggested_action": "Verify crop mask coverage (mask_frac)",
    },
]


def apply_rules(row_dict, pred_class, max_prob):
    """Apply all rule checks. Returns list of triggered rule names + warnings."""
    triggered = []
    for rule in RULE_CHECKS:
        try:
            if rule["name"] == "RULE_03_HIGH_UNCERTAINTY":
                if max_prob < 0.35:
                    triggered.append({
                        "rule": rule["name"],
                        "warning": rule["warning"],
                        "action": rule["suggested_action"],
                    })
            elif rule["condition"](row_dict, pred_class):
                triggered.append({
                    "rule": rule["name"],
                    "warning": rule["warning"],
                    "action": rule["suggested_action"],
                })
        except Exception:
            pass
    return triggered


# ── Core Classifier ───────────────────────────────────────────────────────────
def classify_batch(df_raw, config, clf_model, imp_clf,
                   pca=None, mean_ood=None, inv_cov_ood=None, ood_thresh=np.nan):
    """
    Run classification on a batch DataFrame.
    Returns enriched DataFrame with frontend labels, confidence, OOD flags, rule warnings.
    """
    df = df_raw.copy()
    clf_cols = config["classification_features"]

    # Ensure all columns exist
    for col in clf_cols:
        if col not in df.columns:
            df[col] = np.nan

    X_raw = df[clf_cols].replace([np.inf, -np.inf], np.nan)
    X_imp = imp_clf.transform(X_raw)

    pred_classes = clf_model.predict(X_imp)
    pred_probs   = clf_model.predict_proba(X_imp)

    # OOD scores
    if pca is not None:
        ood_scores = compute_ood_scores(X_imp, pca, mean_ood, inv_cov_ood)
        is_ood     = ood_scores > ood_thresh
    else:
        ood_scores = np.full(len(df), np.nan)
        is_ood     = np.zeros(len(df), dtype=bool)

    results = df_raw.copy()
    results["damage_class_id"]     = pred_classes
    results["damage_label"]        = [FRONTEND_LABELS[c] for c in pred_classes]
    results["damage_internal_name"]= [INTERNAL_LABELS[c] for c in pred_classes]
    results["confidence_score"]    = pred_probs.max(axis=1).round(4)
    results["confidence_level"]    = np.where(
        pred_probs.max(axis=1) >= 0.60, "HIGH",
        np.where(pred_probs.max(axis=1) >= 0.40, "MEDIUM", "LOW")
    )
    results["prob_not_damaged"]        = pred_probs[:, 0].round(4)
    results["prob_less_damaged"]       = pred_probs[:, 1].round(4)
    results["prob_moderately_damaged"] = pred_probs[:, 2].round(4)
    results["prob_severely_damaged"]   = pred_probs[:, 3].round(4)
    results["ood_score"]          = np.round(ood_scores, 3)
    results["ood_threshold"]      = round(ood_thresh, 3) if not np.isnan(ood_thresh) else np.nan
    results["is_ood"]             = is_ood

    # Rule warnings (row-by-row)
    rule_flags = []
    for i, (_, row) in enumerate(df_raw.iterrows()):
        row_d    = row.to_dict()
        rules    = apply_rules(row_d, int(pred_classes[i]), float(pred_probs[i].max()))
        rule_strs = "; ".join([r["rule"] for r in rules]) if rules else None
        rule_flags.append(rule_strs)
    results["rule_warnings"] = rule_flags

    return results


# ── Calibration Curve ─────────────────────────────────────────────────────────
def compute_calibration(y_true, y_prob, n_bins=10):
    """
    Compute calibration per class (one-vs-rest).
    Returns DataFrame with fraction_positives and mean_predicted per bin per class.
    """
    rows = []
    for c in range(4):
        y_bin = (y_true == c).astype(int)
        prob_c = y_prob[:, c]
        try:
            frac_pos, mean_pred = calibration_curve(
                y_bin, prob_c, n_bins=n_bins, strategy="uniform")
            for fp, mp in zip(frac_pos, mean_pred):
                rows.append({
                    "class_id":          c,
                    "frontend_label":    FRONTEND_LABELS[c],
                    "mean_predicted":    round(float(mp), 4),
                    "fraction_positive": round(float(fp), 4),
                })
        except Exception:
            pass
    return pd.DataFrame(rows)


# ── Holdout Validation ────────────────────────────────────────────────────────
def run_holdout_validation(df, config, clf_model, imp_clf,
                           pca, mean_ood, inv_cov_ood, ood_thresh):
    """Evaluate classification on the 2019–2024 holdout set."""
    leakage_cols = [
        "damage_severity", "damage_class_name", "damage_reason", "yield_z", "logy",
        "Yield", "n35", "hdd35", "tmax_mean_W", "HS1", "HS2", "HS3", "HS5",
        "n35_clean", "hdd35_clean", "State", "District", "Year", "P_doy",
        "P_fallback", "excess_rain_shock", "rain_fill_p90", "dndvi_z",
        "ndmi_z", "lst_p90_z",
    ]
    clf_cols = config["classification_features"]

    df_test  = df[df["Year"] >= SPLIT_YEAR].copy()
    df_train = df[df["Year"] < SPLIT_YEAR].copy()

    print(f"  Holdout set: {len(df_test)} samples "
          f"({df_test['Year'].min()}–{df_test['Year'].max()})")

    # Impute
    X_test     = df_test[clf_cols].replace([np.inf, -np.inf], np.nan)
    X_test_imp = imp_clf.transform(X_test)

    y_true = df_test["damage_severity"].to_numpy(int)
    y_pred = clf_model.predict(X_test_imp)
    y_prob = clf_model.predict_proba(X_test_imp)

    # OOD
    ood_scores = compute_ood_scores(X_test_imp, pca, mean_ood, inv_cov_ood)

    # Core metrics
    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    kappa    = cohen_kappa_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    wtd_f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan

    # Severe/Moderate false-negative rate
    sev_mod_mask = y_true >= 2
    fn_sev_mod   = np.sum((y_pred == 0) & sev_mod_mask)
    fn_rate      = fn_sev_mod / sev_mod_mask.sum() if sev_mod_mask.sum() > 0 else 0.0
    sev_mask     = y_true == 3
    sev_recall   = (np.sum((y_pred == 3) & sev_mask) / sev_mask.sum()
                    if sev_mask.sum() > 0 else 0.0)

    # Per-class metrics
    prec_c = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_c  = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_c   = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class_rows = []
    for c in range(4):
        per_class_rows.append({
            "class_id":       c,
            "frontend_label": FRONTEND_LABELS[c],
            "internal_label": INTERNAL_LABELS[c],
            "support":        int(np.sum(y_true == c)),
            "precision":      round(float(prec_c[c]) if c < len(prec_c) else 0.0, 4),
            "recall":         round(float(rec_c[c]) if c < len(rec_c) else 0.0, 4),
            "f1_score":       round(float(f1_c[c]) if c < len(f1_c) else 0.0, 4),
        })
    per_class_df = pd.DataFrame(per_class_rows)

    # Confusion matrix
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_idx = [FRONTEND_LABELS[i] for i in range(4)]
    cm_df  = pd.DataFrame(cm, index=[f"True_{l}" for l in cm_idx],
                          columns=[f"Pred_{l}" for l in cm_idx])

    # Holdout prediction frame
    df_hold = df_test.copy()
    df_hold["damage_label"]          = [FRONTEND_LABELS[c] for c in y_pred]
    df_hold["pred_class_id"]         = y_pred
    df_hold["true_class_id"]         = y_true
    df_hold["true_label"]            = [FRONTEND_LABELS[c] for c in y_true]
    df_hold["correct"]               = (y_pred == y_true)
    df_hold["confidence_score"]      = y_prob.max(axis=1).round(4)
    df_hold["confidence_level"]      = np.where(
        y_prob.max(axis=1) >= 0.60, "HIGH",
        np.where(y_prob.max(axis=1) >= 0.40, "MEDIUM", "LOW"))
    df_hold["prob_not_damaged"]          = y_prob[:, 0].round(4)
    df_hold["prob_less_damaged"]         = y_prob[:, 1].round(4)
    df_hold["prob_moderately_damaged"]   = y_prob[:, 2].round(4)
    df_hold["prob_severely_damaged"]     = y_prob[:, 3].round(4)
    df_hold["ood_score"]             = np.round(ood_scores, 3)
    df_hold["is_ood"]                = ood_scores > ood_thresh

    # Rule warnings on holdout
    rule_flags = []
    for _, row in df_test.iterrows():
        pred_c = int(clf_model.predict(
            imp_clf.transform(pd.DataFrame([row[clf_cols].replace(
                [np.inf, -np.inf], np.nan)])))[0])
        max_p = float(clf_model.predict_proba(
            imp_clf.transform(pd.DataFrame([row[clf_cols].replace(
                [np.inf, -np.inf], np.nan)])))[0].max())
        rules = apply_rules(row.to_dict(), pred_c, max_p)
        rule_flags.append("; ".join([r["rule"] for r in rules]) if rules else None)
    df_hold["rule_warnings"] = rule_flags

    metrics = {
        "n_test":               len(df_test),
        "accuracy":             round(acc, 4),
        "balanced_accuracy":    round(bal_acc, 4),
        "cohen_kappa":          round(kappa, 4),
        "macro_f1":             round(macro_f1, 4),
        "weighted_f1":          round(wtd_f1, 4),
        "roc_auc_ovr":          round(auc, 4) if not np.isnan(auc) else np.nan,
        "severe_recall":        round(sev_recall, 4),
        "sev_mod_fn_rate":      round(fn_rate, 4),
        "ood_flagged_count":    int((ood_scores > ood_thresh).sum()),
        "ood_threshold":        round(ood_thresh, 2),
    }

    return metrics, per_class_df, cm_df, df_hold, compute_calibration(y_true, y_prob)


# ── Report Generation ─────────────────────────────────────────────────────────
def generate_markdown_report(metrics, per_class_df, cm_df, ood_thresh, config, out_dir):
    lines = [
        "# Crop Damage Classification System — Validation Report",
        "",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Model**: {config['classification_model']['type']}  ",
        f"**Test Period**: 2019–2024 (rolling-origin holdout)  ",
        "",
        "---",
        "",
        "## 1. Label Mapping (Internal → Frontend)",
        "",
        "| Class ID | Internal Label | Frontend Label | Color |",
        "|---|---|---|---|",
    ]
    for c in range(4):
        lines.append(
            f"| {c} | {INTERNAL_LABELS[c]} | **{FRONTEND_LABELS[c]}** | {DAMAGE_COLORS[c]} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Holdout Performance (2019–2024)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Samples (test) | {metrics['n_test']} |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} |",
        f"| Cohen's Kappa | {metrics['cohen_kappa']:.4f} |",
        f"| Macro F1 | {metrics['macro_f1']:.4f} |",
        f"| Weighted F1 | {metrics['weighted_f1']:.4f} |",
        f"| ROC-AUC (OvR) | {metrics['roc_auc_ovr']:.4f} |",
        f"| Severely Damaged Recall | {metrics['severe_recall']:.4f} |",
        f"| Severe+Moderate → Not Damaged FN Rate | {metrics['sev_mod_fn_rate']:.4f} |",
        "",
        "---",
        "",
        "## 3. Per-Class Performance",
        "",
    ]

    lines.append("| Class | Frontend Label | Support | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in per_class_df.iterrows():
        lines.append(
            f"| {DAMAGE_COLORS[int(row['class_id'])]} {int(row['class_id'])} "
            f"| {row['frontend_label']} "
            f"| {int(row['support'])} "
            f"| {row['precision']:.4f} "
            f"| {row['recall']:.4f} "
            f"| {row['f1_score']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Confusion Matrix (Frontend Labels)",
        "",
        "```",
        cm_df.to_string(),
        "```",
        "",
        "---",
        "",
        "## 5. Confidence Scoring System",
        "",
        "| Confidence Level | Criterion |",
        "|---|---|",
        "| **HIGH** | Max class probability ≥ 0.60 |",
        "| **MEDIUM** | Max class probability 0.40–0.59 |",
        "| **LOW** | Max class probability < 0.40 |",
        "",
        "---",
        "",
        "## 6. OOD (Out-of-Distribution) Detection",
        "",
        "Method: PCA-reduced Mahalanobis distance from training distribution  ",
        "Threshold: 97.5th percentile of training set distances  ",
        "",
        f"- **OOD Threshold**: {ood_thresh:.2f}",
        f"- **OOD-flagged in holdout**: {metrics['ood_flagged_count']} / {metrics['n_test']}",
        "",
        "> Predictions on OOD samples are still returned but marked with `is_ood=True`.  ",
        "> Users should treat OOD predictions with extra caution.",
        "",
        "---",
        "",
        "## 7. Rule-Based Verification Layer",
        "",
        "The following rules add **warning flags** (not hard overrides) to predictions:",
        "",
    ]
    for rule in RULE_CHECKS:
        lines.append(f"- **{rule['name']}**: {rule['description']}")
        lines.append(f"  - *Warning*: {rule['warning']}")
        lines.append(f"  - *Suggested Action*: {rule['suggested_action']}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 8. Production API",
        "",
        "The `CropDamageClassifier` class (in `models/damage_classifier_v2.py`) provides:",
        "",
        "```python",
        "from models.damage_classifier_v2 import CropDamageClassifier",
        "",
        "clf = CropDamageClassifier(models_dir='models')",
        "results = clf.predict(df_features)",
        "# Returns DataFrame with columns:",
        "#   damage_label, damage_class_id, confidence_level,",
        "#   confidence_score, is_ood, ood_score, rule_warnings,",
        "#   prob_not_damaged, prob_less_damaged,",
        "#   prob_moderately_damaged, prob_severely_damaged",
        "```",
        "",
        "---",
        "",
        "## 9. Physical Interpretation Notes",
        "",
        "> [!NOTE]",
        "> The model distinguishes **heat-related crop damage** from **vegetation",
        "> decline caused by other factors**. A drop in NDVI alone does NOT imply",
        "> heatwave damage — the model requires corroborating evidence from",
        "> temperature features (LST, Tmax, HDD35) and moisture indicators (NDMI, SM).",
        "",
        "> [!WARNING]",
        "> For districts/years flagged as OOD (`is_ood=True`), predictions may be",
        "> unreliable. Exercise extra caution for future years (2025+) or unseen",
        "> districts.",
        "",
    ]

    report_md = "\n".join(lines)
    out_path  = os.path.join(out_dir, "damage_classification_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  Report saved: {out_path}")
    return report_md


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    section("D2: CROP DAMAGE CLASSIFICATION SYSTEM (PHASE 3)")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load pipeline
    section("1. Loading Production Pipeline")
    config, clf_model, imp_clf = load_pipeline()
    print(f"  Classifier : {config['classification_model']['type']}")
    print(f"  Features   : {len(config['classification_features'])}")
    clf_cols = config["classification_features"]

    # 2. Load data
    section("2. Loading and Splitting Data")
    if not os.path.exists(DATA_PATH):
        sys.exit(f"[ERROR] Missing {DATA_PATH}. Run C2 first.")

    df       = pd.read_csv(DATA_PATH)
    df_train = df[df["Year"] < SPLIT_YEAR].copy()
    df_test  = df[df["Year"] >= SPLIT_YEAR].copy()

    print(f"  Full dataset : {len(df)} rows")
    print(f"  Train (<{SPLIT_YEAR}): {len(df_train)} rows")
    print(f"  Test  (≥{SPLIT_YEAR}): {len(df_test)} rows")
    print("  Class distribution (test):",
          df_test["damage_severity"].value_counts().sort_index().to_dict())
    print("  Frontend label distribution (test):")
    for c, count in df_test["damage_severity"].value_counts().sort_index().items():
        print(f"    {FRONTEND_LABELS[int(c)]}: {count}")

    # 3. Build OOD detector from training data
    section("3. Building OOD Detector (PCA-Mahalanobis)")
    X_tr_raw = df_train[clf_cols].replace([np.inf, -np.inf], np.nan)
    X_tr_imp = imp_clf.transform(X_tr_raw)

    pca, mean_ood, inv_cov_ood, ood_thresh = build_ood_detector_pca(
        X_tr_imp, n_components=20, threshold_percentile=97.5)
    print(f"  PCA components  : {pca.n_components_}")
    print(f"  Explained variance (cumulative): "
          f"{pca.explained_variance_ratio_.cumsum()[-1]:.1%}")
    print(f"  OOD threshold (97.5th pct)     : {ood_thresh:.2f}")

    # Quick sanity: what fraction of training samples are flagged as OOD?
    tr_scores = compute_ood_scores(X_tr_imp, pca, mean_ood, inv_cov_ood)
    tr_ood_frac = (tr_scores > ood_thresh).mean()
    print(f"  Training set OOD rate (should be ≈2.5%): {tr_ood_frac:.1%}")

    # 4. Run holdout validation
    section("4. Running Holdout Validation (2019–2024)")
    metrics, per_class_df, cm_df, df_hold, calib_df = run_holdout_validation(
        df, config, clf_model, imp_clf, pca, mean_ood, inv_cov_ood, ood_thresh)

    print(f"\n  ── Overall Metrics ──")
    for k, v in metrics.items():
        print(f"    {k:35s}: {v}")

    print(f"\n  ── Per-Class Performance ──")
    print(per_class_df.to_string(index=False))

    print(f"\n  ── Confusion Matrix (Frontend Labels) ──")
    print(cm_df.to_string())

    # Save outputs
    df_hold.to_csv(os.path.join(OUT_DIR, "holdout_validation.csv"), index=False)
    per_class_df.to_csv(os.path.join(OUT_DIR, "per_class_metrics.csv"), index=False)
    cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix_frontend.csv"))
    calib_df.to_csv(os.path.join(OUT_DIR, "calibration_curve.csv"), index=False)
    print(f"\n  Holdout validation saved to {OUT_DIR}/")

    # 5. Confidence distribution analysis
    section("5. Confidence Distribution Analysis")
    conf_dist = pd.crosstab(
        df_hold["confidence_level"],
        df_hold["true_label"],
        margins=True, margins_name="Total"
    )
    acc_by_conf = df_hold.groupby("confidence_level")["correct"].mean()
    print("  Accuracy by confidence level:")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        if lvl in acc_by_conf.index:
            n = df_hold[df_hold["confidence_level"] == lvl].shape[0]
            print(f"    {lvl:6s}: {acc_by_conf[lvl]:.3f} (n={n})")
    conf_dist.to_csv(os.path.join(OUT_DIR, "confidence_distribution.csv"))

    # 6. Rule-based flag analysis
    section("6. Rule-Based Verification Layer Analysis")
    flagged = df_hold[df_hold["rule_warnings"].notna()]
    print(f"  Predictions with rule warnings: {len(flagged)} / {len(df_hold)}")
    if len(flagged) > 0:
        print("  Most common rule flags:")
        all_flags = []
        for flags in flagged["rule_warnings"].dropna():
            all_flags.extend(flags.split("; "))
        from collections import Counter
        for rule, cnt in Counter(all_flags).most_common():
            print(f"    {rule}: {cnt}")

    # 7. Year-by-year performance
    section("7. Year-by-Year Performance")
    yearly = []
    for yr in sorted(df_test["Year"].unique()):
        sub = df_hold[df_hold["Year"] == yr]
        y_t = sub["true_class_id"].to_numpy()
        y_p = sub["pred_class_id"].to_numpy()
        acc_yr = accuracy_score(y_t, y_p)
        f1_yr  = f1_score(y_t, y_p, average="macro", zero_division=0)
        yearly.append({"Year": int(yr), "N": len(sub),
                       "Accuracy": round(acc_yr, 4), "Macro_F1": round(f1_yr, 4),
                       "OOD_flagged": int(sub["is_ood"].sum())})
    yr_df = pd.DataFrame(yearly)
    print(yr_df.to_string(index=False))
    yr_df.to_csv(os.path.join(OUT_DIR, "yearly_performance.csv"), index=False)

    # 8. Generate markdown report
    section("8. Generating Validation Report")
    generate_markdown_report(metrics, per_class_df, cm_df, ood_thresh, config, OUT_DIR)

    # 9. Export production class
    section("9. Exporting Production CropDamageClassifier Class")
    export_production_class()
    print(f"  Production class exported to: models/damage_classifier_v2.py")

    # Final summary
    print("\n" + "=" * 72)
    print("  D2 CLASSIFICATION SYSTEM SUMMARY")
    print("=" * 72)
    print(f"  Model            : {config['classification_model']['type']}")
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  ROC-AUC OvR      : {metrics['roc_auc_ovr']:.4f}")
    print(f"  Severe Recall    : {metrics['severe_recall']:.4f}")
    print(f"  Sev+Mod→No FN    : {metrics['sev_mod_fn_rate']:.4f}")
    print(f"  OOD Threshold    : {ood_thresh:.2f}")
    print(f"  OOD in holdout   : {metrics['ood_flagged_count']}/{metrics['n_test']}")
    print("=" * 72)
    print("  Frontend Labels:")
    for c, lbl in FRONTEND_LABELS.items():
        row = per_class_df[per_class_df["class_id"] == c].iloc[0]
        print(f"    {DAMAGE_COLORS[c]} Class {c}: {lbl:22s} | "
              f"F1={row['f1_score']:.4f} | Recall={row['recall']:.4f}")
    print("=" * 72)


def export_production_class():
    """Write the CropDamageClassifier class to models/damage_classifier_v2.py."""
    code = '''"""
CropDamageClassifier — Production Class (PHASE 3)
==================================================
Frontend-ready 4-class wheat heatwave damage classifier.

Classes:
  0 → NOT DAMAGED
  1 → LESS DAMAGED
  2 → MODERATELY DAMAGED
  3 → SEVERELY DAMAGED

Usage:
  from models.damage_classifier_v2 import CropDamageClassifier
  clf = CropDamageClassifier(models_dir="models")
  results = clf.predict(df_features)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

FRONTEND_LABELS = {
    0: "NOT DAMAGED",
    1: "LESS DAMAGED",
    2: "MODERATELY DAMAGED",
    3: "SEVERELY DAMAGED",
}
INTERNAL_LABELS = {
    0: "No Significant Heatwave Damage",
    1: "Mild Heatwave Damage",
    2: "Moderate Heatwave Damage",
    3: "Severe Heatwave Damage",
}


class CropDamageClassifier:
    """
    Production-ready wheat heatwave crop damage classifier.

    Attributes
    ----------
    models_dir : str
        Path to the directory containing trained .joblib files and feature_config.json.
    ood_percentile : float
        Percentile of training distribution for OOD threshold (default: 97.5).
    """

    def __init__(self, models_dir: str = "models", ood_percentile: float = 97.5):
        self.models_dir     = models_dir
        self.ood_percentile = ood_percentile
        self._loaded        = False
        self._ood_built     = False

    def load(self):
        """Load all model artifacts from models_dir."""
        config_path = os.path.join(self.models_dir, "feature_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"feature_config.json not found in {self.models_dir}. Run C5 first.")

        with open(config_path) as f:
            self.config = json.load(f)

        self.clf_model  = joblib.load(
            os.path.join(self.models_dir, self.config["classification_model"]["file"]))
        self.imp_clf    = joblib.load(
            os.path.join(self.models_dir, "preprocessor_imputer.joblib"))
        self.clf_cols   = self.config["classification_features"]
        self.class_map  = {int(k): v for k, v in self.config["class_mapping"].items()}

        # Optional: exposure regressor
        try:
            self.reg_model  = joblib.load(
                os.path.join(self.models_dir, self.config["exposure_model"]["file"]))
            self.imp_exp    = joblib.load(
                os.path.join(self.models_dir, "imputer_exposure.joblib"))
            self.exp_cols   = self.config["exposure_features"]
        except Exception:
            self.reg_model = None

        self._loaded = True
        return self

    def fit_ood_detector(self, df_train: pd.DataFrame, n_pca_components: int = 20):
        """
        Fit the OOD detector from training data.
        Call this after load() and before predict().

        Parameters
        ----------
        df_train : pd.DataFrame
            Training set (Year < split_year). Must contain classification features.
        n_pca_components : int
            Number of PCA components for dimensionality reduction before Mahalanobis.
        """
        if not self._loaded:
            self.load()

        X_raw = df_train[self.clf_cols].replace([np.inf, -np.inf], np.nan)
        X_imp = self.imp_clf.transform(X_raw)

        n_comp      = min(n_pca_components, X_imp.shape[1], X_imp.shape[0] - 1)
        self._pca   = PCA(n_components=n_comp, random_state=42)
        Z_tr        = self._pca.fit_transform(X_imp)
        self._mean  = Z_tr.mean(axis=0)
        cov_        = np.cov(Z_tr.T) + np.eye(n_comp) * 1e-4
        self._inv_c = np.linalg.inv(cov_)

        dists           = np.array([float((z - self._mean) @ self._inv_c @ (z - self._mean))
                                     for z in Z_tr])
        self._ood_thresh = float(np.percentile(dists, self.ood_percentile))
        self._ood_built  = True
        return self

    def _compute_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recompute interaction features if not present."""
        d = df.copy()

        def _g(col, default):
            return d.get(col, pd.Series(default, index=d.index)).fillna(default)

        if "inter_lst_vpd" not in d.columns:
            d["inter_lst_vpd"] = (_g("sat_lst_day_mean", 30.0) *
                                   _g("wx_fill_vpd_mean", 2.0)).round(4)
        if "inter_tmax_solar" not in d.columns:
            d["inter_tmax_solar"] = (_g("wx_fill_tmax_max", 33.0) *
                                      (_g("wx_fill_solar_sum", 600.0) / 100.0)).round(4)
        if "inter_dtr_sm" not in d.columns:
            d["inter_dtr_sm"] = (_g("sat_lst_dtr", 15.0) /
                                  _g("wx_fill_sm_mean", 0.2).clip(lower=0.02)).round(4)
        if "inter_sm_buffer" not in d.columns:
            d["inter_sm_buffer"] = (_g("wx_fill_sm_mean", 0.2) /
                                     (_g("wx_fill_hdd35", 0.0) + 1.0)).round(4)
        if "canopy_senescence_rate" not in d.columns:
            d["canopy_senescence_rate"] = (
                (_g("veg_NDVI_v3", 0.65) - _g("S_ndvi", 0.55)) / 40.0
            ).clip(lower=0).round(5)
        if "inter_lst_ndmi_deficit" not in d.columns:
            d["inter_lst_ndmi_deficit"] = (
                _g("sat_lst_day_p90", 35.0) *
                (1.0 - _g("sat_ndmi_W", 0.2).clip(lower=-0.5, upper=1.0))
            ).round(4)
        if "inter_heat_persistence" not in d.columns:
            d["inter_heat_persistence"] = (
                _g("wx_fill_hot35_days", 0.0) * (_g("wx_fill_run35", 0.0) + 1.0)
            ).round(2)
        if "pre_vigor_ratio" not in d.columns:
            d["pre_vigor_ratio"] = (
                _g("veg_NDVI_v3", 0.65) / _g("veg_NDVI_v1", 0.40).clip(lower=0.05)
            ).round(4)
        return d

    def _apply_rules(self, row_dict: dict, pred_class: int, max_prob: float):
        """Return list of rule warning strings triggered for a single prediction."""
        warnings_out = []
        # R01: Severe but no heat
        if (pred_class == 3
                and row_dict.get("wx_fill_hot35_days", 1) == 0
                and row_dict.get("wx_anth_hot35_days", 1) == 0
                and not np.isnan(row_dict.get("sat_lst_day_p90", np.nan))
                and row_dict.get("sat_lst_day_p90", 40) < 35):
            warnings_out.append(
                "RULE_01: Severe predicted but no hot days — check data quality")
        # R02: No damage despite extreme heat + SM deficit
        if (pred_class == 0
                and row_dict.get("wx_fill_hot35_days", 0) >= 15
                and row_dict.get("wx_fill_sm_mean", 0.5) < 0.05
                and row_dict.get("sat_lst_day_p90", 30) > 45):
            warnings_out.append(
                "RULE_02: No damage despite extreme heat + SM deficit — verify irrigation")
        # R03: Very low confidence
        if max_prob < 0.35:
            warnings_out.append(
                "RULE_03: Very low confidence (<0.35) — treat as inconclusive")
        # R04: Physically impossible LST
        if (row_dict.get("sat_lst_day_mean", 30) > 65
                or row_dict.get("sat_lst_day_mean", 30) < -10):
            warnings_out.append(
                "RULE_04: LST outside plausible range — check satellite QA flags")
        # R05: Negative NDVI + high damage
        if row_dict.get("sat_ndvi_W", 0.4) < 0 and pred_class >= 2:
            warnings_out.append(
                "RULE_05: NDVI<0 (non-crop surface) — damage label may not apply")
        return warnings_out

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Classify crop heatwave damage for each row in df_raw.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Input features. Missing columns are filled with NaN (imputed by median).

        Returns
        -------
        pd.DataFrame
            Original data + columns:
              damage_label               : str  (frontend label)
              damage_class_id            : int  (0–3)
              damage_internal_name       : str  (scientific label)
              confidence_score           : float [0,1]
              confidence_level           : "HIGH" | "MEDIUM" | "LOW"
              prob_not_damaged           : float
              prob_less_damaged          : float
              prob_moderately_damaged    : float
              prob_severely_damaged      : float
              pred_exposure_tmax_degC    : float (if exposure model loaded)
              is_ood                     : bool
              ood_score                  : float
              rule_warnings              : str | None
        """
        if not self._loaded:
            self.load()

        df = self._compute_interactions(df_raw.copy())

        # Ensure all classification columns exist
        for col in self.clf_cols:
            if col not in df.columns:
                df[col] = np.nan

        X_raw = df[self.clf_cols].replace([np.inf, -np.inf], np.nan)
        X_imp = self.imp_clf.transform(X_raw)

        pred_classes = self.clf_model.predict(X_imp)
        pred_probs   = self.clf_model.predict_proba(X_imp)

        # OOD
        if self._ood_built:
            Z        = self._pca.transform(X_imp)
            ood_sc   = np.array([float((z - self._mean) @ self._inv_c @ (z - self._mean))
                                  for z in Z])
            is_ood   = ood_sc > self._ood_thresh
        else:
            ood_sc = np.full(len(df), np.nan)
            is_ood = np.zeros(len(df), dtype=bool)

        # Exposure regression (optional)
        if self.reg_model is not None:
            for col in self.exp_cols:
                if col not in df.columns:
                    df[col] = np.nan
            X_exp_raw = df[self.exp_cols].replace([np.inf, -np.inf], np.nan)
            X_exp_imp = self.imp_exp.transform(X_exp_raw)
            pred_exp  = self.reg_model.predict(X_exp_imp)
        else:
            pred_exp = np.full(len(df), np.nan)

        # Assemble result
        out = df_raw.copy()
        out["damage_label"]           = [FRONTEND_LABELS[c] for c in pred_classes]
        out["damage_class_id"]        = pred_classes
        out["damage_internal_name"]   = [INTERNAL_LABELS[c] for c in pred_classes]
        out["confidence_score"]       = pred_probs.max(axis=1).round(4)
        out["confidence_level"]       = np.where(
            pred_probs.max(axis=1) >= 0.60, "HIGH",
            np.where(pred_probs.max(axis=1) >= 0.40, "MEDIUM", "LOW"))
        out["prob_not_damaged"]            = pred_probs[:, 0].round(4)
        out["prob_less_damaged"]           = pred_probs[:, 1].round(4)
        out["prob_moderately_damaged"]     = pred_probs[:, 2].round(4)
        out["prob_severely_damaged"]       = pred_probs[:, 3].round(4)
        out["pred_exposure_tmax_degC"]     = np.round(pred_exp, 2)
        out["is_ood"]                      = is_ood
        out["ood_score"]                   = np.round(ood_sc, 3)
        out["ood_threshold"]               = (round(self._ood_thresh, 2)
                                               if self._ood_built else np.nan)

        # Rule warnings
        rule_flags = []
        for i, (_, row) in enumerate(df_raw.iterrows()):
            row_d = row.to_dict()
            flags = self._apply_rules(
                row_d, int(pred_classes[i]), float(pred_probs[i].max()))
            rule_flags.append("; ".join(flags) if flags else None)
        out["rule_warnings"] = rule_flags

        return out

    @property
    def label_map(self) -> dict:
        """Return the frontend label map {class_id: label_string}."""
        return FRONTEND_LABELS.copy()

    @property
    def ood_threshold(self) -> float:
        """Return the OOD detection threshold."""
        return self._ood_thresh if self._ood_built else np.nan

    def __repr__(self):
        status = "loaded" if self._loaded else "not loaded"
        ood    = f"OOD thresh={self._ood_thresh:.2f}" if self._ood_built else "OOD not fitted"
        return f"CropDamageClassifier({status}, {ood})"
'''

    out_path = os.path.join("models", "damage_classifier_v2.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code)


if __name__ == "__main__":
    main()
