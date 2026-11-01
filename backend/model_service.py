"""
Model Service
=============
Manages singleton instance of CropDamageClassifier (models/damage_classifier_v2.py),
initializes PCA-Mahalanobis OOD detection, loads historical training/validation data,
and provides model metadata and performance reports.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.damage_classifier_v2 import CropDamageClassifier, FRONTEND_LABELS, INTERNAL_LABELS

MODELS_DIR = os.path.join(ROOT, "models")
CONFIG_PATH = os.path.join(MODELS_DIR, "feature_config.json")
FEATURES_PATH = os.path.join(ROOT, "data", "processed", "features_enriched.csv")
CENTROIDS_PATH = os.path.join(ROOT, "data", "external", "district_centroids.csv")


class ModelService:
    _instance = None

    def __init__(self):
        print("Initializing ModelService...")
        self.clf = CropDamageClassifier(models_dir=MODELS_DIR, ood_percentile=97.5)
        self.clf.load()
        
        # Load features dataset for OOD fitting and historical lookups
        if os.path.exists(FEATURES_PATH):
            self.df_features = pd.read_csv(FEATURES_PATH)
            # Fit OOD on training split (< 2019)
            df_train = self.df_features[self.df_features["Year"] < 2019]
            self.clf.fit_ood_detector(df_train)
            print(f"OOD detector fitted on {len(df_train)} training rows. Threshold: {self.clf.ood_threshold:.2f}")
        else:
            self.df_features = pd.DataFrame()
            print("[WARN] features_enriched.csv not found; OOD detector unfitted.")

        # Load centroids
        if os.path.exists(CENTROIDS_PATH):
            self.df_centroids = pd.read_csv(CENTROIDS_PATH)
        else:
            self.df_centroids = pd.DataFrame()

        # Load model config
        with open(CONFIG_PATH, "r") as f:
            self.config = json.load(f)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def get_metadata(self) -> Dict[str, Any]:
        archive_years = [int(y) for y in sorted(self.df_features["Year"].unique())] if not self.df_features.empty else list(range(2003, 2025))
        # Support operational, current, and upcoming harvest years (2025, 2026, 2027)
        operational_years = list(range(2003, 2028))
        all_years = sorted(list(set(archive_years + operational_years)))
        return {
            "project_title": self.config.get("project_title", "Wheat Crop Heatwave Damage Estimation"),
            "model_type": self.config.get("classification_model", {}).get("type", "ExtraTreesClassifier (Balanced)"),
            "exposure_model_type": self.config.get("exposure_model", {}).get("type", "ExtraTreesRegressor"),
            "split_year": self.config.get("split_year", 2019),
            "n_classes": 4,
            "frontend_labels": FRONTEND_LABELS,
            "internal_labels": INTERNAL_LABELS,
            "classification_features_count": len(self.clf.clf_cols),
            "exposure_features_count": len(getattr(self.clf, "exp_cols", [])),
            "ood_threshold": round(self.clf.ood_threshold, 2) if not np.isnan(self.clf.ood_threshold) else 53.72,
            "supported_states": ["Haryana", "Punjab", "Uttar Pradesh"],
            "total_districts": len(self.df_centroids["District"].unique()) if not self.df_centroids.empty else 37,
            "years_available": all_years,
            "validated_test_years": [2019, 2020, 2021, 2022, 2023, 2024],
            "operational_future_years": [2025, 2026, 2027]
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        # Exact metrics from 2019-2024 holdout validation report
        return {
            "overall": {
                "test_samples": 222,
                "test_period": "2019–2024 (rolling-origin holdout)",
                "accuracy": 0.7613,
                "balanced_accuracy": 0.5414,
                "cohen_kappa": 0.5694,
                "macro_f1": 0.5459,
                "weighted_f1": 0.7404,
                "roc_auc_ovr": 0.9059,
                "severe_recall": 0.3000,
                "sev_mod_fn_rate": 0.0392,
                "ood_flagged_count": 25,
                "ood_threshold": 53.72
            },
            "per_class": [
                {
                    "class_id": 0,
                    "frontend_label": "NOT DAMAGED",
                    "internal_label": "No Significant Heatwave Damage",
                    "color": "#16a34a", # green
                    "support": 135,
                    "precision": 0.9231,
                    "recall": 0.9778,
                    "f1_score": 0.9496
                },
                {
                    "class_id": 1,
                    "frontend_label": "LESS DAMAGED",
                    "internal_label": "Mild Heatwave Damage",
                    "color": "#eab308", # yellow
                    "support": 36,
                    "precision": 0.4237,
                    "recall": 0.6944,
                    "f1_score": 0.5263
                },
                {
                    "class_id": 2,
                    "frontend_label": "MODERATELY DAMAGED",
                    "internal_label": "Moderate Heatwave Damage",
                    "color": "#f97316", # orange
                    "support": 31,
                    "precision": 0.5000,
                    "recall": 0.1935,
                    "f1_score": 0.2791
                },
                {
                    "class_id": 3,
                    "frontend_label": "SEVERELY DAMAGED",
                    "internal_label": "Severe Heatwave Damage",
                    "color": "#dc2626", # red
                    "support": 20,
                    "precision": 0.7500,
                    "recall": 0.3000,
                    "f1_score": 0.4286
                }
            ],
            "confusion_matrix": [
                [132, 2, 1, 0],
                [9, 25, 2, 0],
                [2, 21, 6, 2],
                [0, 11, 3, 6]
            ],
            "confidence_calibration": [
                {"level": "HIGH", "min_prob": ">= 0.60", "accuracy": 0.950, "samples": 120},
                {"level": "MEDIUM", "min_prob": "0.40–0.59", "accuracy": 0.592, "samples": 76},
                {"level": "LOW", "min_prob": "< 0.40", "accuracy": 0.385, "samples": 26}
            ],
            "limitations": [
                "Strong performance on NOT DAMAGED samples (97.78% recall, 94.96% F1).",
                "Moderate performance on LESS DAMAGED samples (69.44% recall, 52.63% F1).",
                "Low sensitivity on MODERATELY DAMAGED samples (19.35% recall) due to subtle transitions between mild and moderate heat stress.",
                "Severe damage recall is 30.00% (6 of 20 severe cases captured; 11 predicted as LESS DAMAGED).",
                "High overall accuracy (76.13%) is influenced by class imbalance (135/222 holdout samples are NOT DAMAGED).",
                "Critical safety check: Severe + Moderate → Not Damaged false negative rate is low at 3.92% (only 2 out of 51 damaged seasons misclassified as undamaged).",
                "Scientific Notice: This classifier assesses wheat crop heatwave damage risk based on satellite and meteorological features. It should not be treated as a definitive ground diagnosis without field validation."
            ]
        }

    def predict_features(self, df_input: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs the CropDamageClassifier on a 1-row or multi-row DataFrame.
        Returns formatted result dictionary.
        """
        results_df = self.clf.predict(df_input)
        row = results_df.iloc[0]

        # Parse rule warnings
        rule_warnings = []
        if pd.notna(row.get("rule_warnings")) and row["rule_warnings"]:
            rule_warnings = [w.strip() for w in str(row["rule_warnings"]).split(";") if w.strip()]

        pred_res = {
            "damage_class_id": int(row["damage_class_id"]),
            "damage_label": str(row["damage_label"]),
            "damage_internal_name": str(row["damage_internal_name"]),
            "confidence_level": str(row["confidence_level"]),
            "confidence_score": float(row["confidence_score"]),
            "probabilities": {
                "not_damaged": float(row["prob_not_damaged"]),
                "less_damaged": float(row["prob_less_damaged"]),
                "moderately_damaged": float(row["prob_moderately_damaged"]),
                "severely_damaged": float(row["prob_severely_damaged"]),
            },
            "pred_exposure_tmax_degC": float(row["pred_exposure_tmax_degC"]) if pd.notna(row.get("pred_exposure_tmax_degC")) else None,
            "ood": {
                "is_ood": bool(row["is_ood"]),
                "ood_score": float(row["ood_score"]) if pd.notna(row.get("ood_score")) else 0.0,
                "threshold": float(row["ood_threshold"]) if pd.notna(row.get("ood_threshold")) else 53.72,
            },
            "rule_warnings": rule_warnings
        }
        return pred_res

    def get_historical_district(self, state: str, district: str) -> List[Dict[str, Any]]:
        """Return historical 2003-2024 time-series records for a given district."""
        if self.df_features.empty:
            return []
        
        mask = (self.df_features["State"].str.lower() == state.lower()) & \
               (self.df_features["District"].str.lower() == district.lower())
        sub = self.df_features[mask].sort_values("Year")
        if sub.empty:
            # Try district match alone
            sub = self.df_features[self.df_features["District"].str.lower() == district.lower()].sort_values("Year")
            
        records = []
        for _, r in sub.iterrows():
            records.append({
                "year": int(r["Year"]),
                "state": str(r["State"]),
                "district": str(r["District"]),
                "damage_severity": int(r["damage_severity"]) if pd.notna(r.get("damage_severity")) else 0,
                "damage_label": FRONTEND_LABELS.get(int(r["damage_severity"]), "UNKNOWN") if pd.notna(r.get("damage_severity")) else "NOT DAMAGED",
                "sat_ndvi_W": round(float(r["sat_ndvi_W"]), 3) if pd.notna(r.get("sat_ndvi_W")) else None,
                "sat_lst_day_mean": round(float(r["sat_lst_day_mean"]), 1) if pd.notna(r.get("sat_lst_day_mean")) else None,
                "sat_lst_day_p90": round(float(r["sat_lst_day_p90"]), 1) if pd.notna(r.get("sat_lst_day_p90")) else None,
                "wx_fill_tmax_max": round(float(r["wx_fill_tmax_max"]), 1) if pd.notna(r.get("wx_fill_tmax_max")) else None,
                "wx_fill_hot35_days": int(r["wx_fill_hot35_days"]) if pd.notna(r.get("wx_fill_hot35_days")) else 0,
                "wx_fill_sm_mean": round(float(r["wx_fill_sm_mean"]), 3) if pd.notna(r.get("wx_fill_sm_mean")) else None,
                "wx_fill_rain_sum": round(float(r["wx_fill_rain_sum"]), 1) if pd.notna(r.get("wx_fill_rain_sum")) else None,
                "hdd35": round(float(r["hdd35"]), 1) if pd.notna(r.get("hdd35")) else None,
                "yield_kg_ha": round(float(r["Yield"]), 1) if pd.notna(r.get("Yield")) else None
            })
        return records
