# Crop Damage Classification System — Validation Report

**Generated**: 2026-11-01 22:00:04  
**Model**: ExtraTreesClassifier (Balanced)  
**Test Period**: 2019–2024 (rolling-origin holdout)  

---

## 1. Label Mapping (Internal → Frontend)

| Class ID | Internal Label | Frontend Label | Color |
|---|---|---|---|
| 0 | No Significant Heatwave Damage | **NOT DAMAGED** | 🟢 |
| 1 | Mild Heatwave Damage | **LESS DAMAGED** | 🟡 |
| 2 | Moderate Heatwave Damage | **MODERATELY DAMAGED** | 🟠 |
| 3 | Severe Heatwave Damage | **SEVERELY DAMAGED** | 🔴 |

---

## 2. Holdout Performance (2019–2024)

| Metric | Value |
|---|---|
| Samples (test) | 222 |
| Accuracy | 0.7613 |
| Balanced Accuracy | 0.5414 |
| Cohen's Kappa | 0.5694 |
| Macro F1 | 0.5459 |
| Weighted F1 | 0.7404 |
| ROC-AUC (OvR) | 0.9059 |
| Severely Damaged Recall | 0.3000 |
| Severe+Moderate → Not Damaged FN Rate | 0.0392 |

---

## 3. Per-Class Performance

| Class | Frontend Label | Support | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 🟢 0 | NOT DAMAGED | 135 | 0.9231 | 0.9778 | 0.9496 |
| 🟡 1 | LESS DAMAGED | 36 | 0.4237 | 0.6944 | 0.5263 |
| 🟠 2 | MODERATELY DAMAGED | 31 | 0.5000 | 0.1935 | 0.2791 |
| 🔴 3 | SEVERELY DAMAGED | 20 | 0.7500 | 0.3000 | 0.4286 |

---

## 4. Confusion Matrix (Frontend Labels)

```
                         Pred_NOT DAMAGED  Pred_LESS DAMAGED  Pred_MODERATELY DAMAGED  Pred_SEVERELY DAMAGED
True_NOT DAMAGED                      132                  2                        1                      0
True_LESS DAMAGED                       9                 25                        2                      0
True_MODERATELY DAMAGED                 2                 21                        6                      2
True_SEVERELY DAMAGED                   0                 11                        3                      6
```

---

## 5. Confidence Scoring System

| Confidence Level | Criterion |
|---|---|
| **HIGH** | Max class probability ≥ 0.60 |
| **MEDIUM** | Max class probability 0.40–0.59 |
| **LOW** | Max class probability < 0.40 |

---

## 6. OOD (Out-of-Distribution) Detection

Method: PCA-reduced Mahalanobis distance from training distribution  
Threshold: 97.5th percentile of training set distances  

- **OOD Threshold**: 53.72
- **OOD-flagged in holdout**: 25 / 222

> Predictions on OOD samples are still returned but marked with `is_ood=True`.  
> Users should treat OOD predictions with extra caution.

---

## 7. Rule-Based Verification Layer

The following rules add **warning flags** (not hard overrides) to predictions:

- **RULE_01_SEVERE_NO_HEAT**: Predicted SEVERELY DAMAGED but no measurable heat exposure
  - *Warning*: ⚠️ Model predicts SEVERELY DAMAGED but no hot days detected. Check for data quality issues or non-heat causes.
  - *Suggested Action*: Review satellite data for cloud contamination

- **RULE_02_NO_DAMAGE_HIGH_HEAT**: Predicted NOT DAMAGED but extreme heat + moisture deficit
  - *Warning*: ⚠️ Model predicts NOT DAMAGED but extreme heat + severe SM deficit. Possible data inconsistency or strong irrigation buffering.
  - *Suggested Action*: Verify irrigation records for this district-year

- **RULE_03_HIGH_UNCERTAINTY**: Very low confidence (max_prob < 0.35)
  - *Warning*: ⚠️ Very low prediction confidence. Result should be treated as uncertain.
  - *Suggested Action*: Consider this an inconclusive result

- **RULE_04_IMPOSSIBLE_LST**: LST values physically impossible (>65°C or <-10°C)
  - *Warning*: ⚠️ LST values outside physically plausible range. Possible cloud contamination or sensor artefact.
  - *Suggested Action*: Recheck satellite QA flags for this composite

- **RULE_05_NEGATIVE_NDVI_DAMAGE**: NDVI < 0 — non-vegetation surface
  - *Warning*: ⚠️ NDVI < 0 suggests bare soil/water. Damage classification may not apply to this land cover.
  - *Suggested Action*: Verify crop mask coverage (mask_frac)

---

## 8. Production API

The `CropDamageClassifier` class (in `models/damage_classifier_v2.py`) provides:

```python
from models.damage_classifier_v2 import CropDamageClassifier

clf = CropDamageClassifier(models_dir='models')
results = clf.predict(df_features)
# Returns DataFrame with columns:
#   damage_label, damage_class_id, confidence_level,
#   confidence_score, is_ood, ood_score, rule_warnings,
#   prob_not_damaged, prob_less_damaged,
#   prob_moderately_damaged, prob_severely_damaged
```

---

## 9. Physical Interpretation Notes

> [!NOTE]
> The model distinguishes **heat-related crop damage** from **vegetation
> decline caused by other factors**. A drop in NDVI alone does NOT imply
> heatwave damage — the model requires corroborating evidence from
> temperature features (LST, Tmax, HDD35) and moisture indicators (NDMI, SM).

> [!WARNING]
> For districts/years flagged as OOD (`is_ood=True`), predictions may be
> unreliable. Exercise extra caution for future years (2025+) or unseen
> districts.
