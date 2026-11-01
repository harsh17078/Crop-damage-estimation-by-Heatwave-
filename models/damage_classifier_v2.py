"""
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
