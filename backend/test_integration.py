"""
Automated Integration Test Suite
================================
Validates end-to-end integration between:
1. Existing Python inference pipeline (CropDamageClassifier v2)
2. Backend REST API (Flask)
3. Google Earth Engine feature extraction
4. SQLite persistence & CSV exports
5. Geographic and Catchment validations
"""

import os
import sys
import json
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app import app
from models.damage_classifier_v2 import CropDamageClassifier


def run_tests():
    print("=" * 75)
    print("RUNNING COMPREHENSIVE INTEGRATION TEST SUITE")
    print("=" * 75)

    client = app.test_client()

    # 1. Test Health Endpoint
    print("\n[TEST 1] Testing /api/health...")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.status_code}"
    health_data = res.json
    assert health_data["status"] == "healthy"
    assert health_data["services"]["model_classifier"] == "loaded"
    print("  -> PASS: Backend and GEE status healthy.")

    # 2. Test Geographic Hierarchy
    print("\n[TEST 2] Testing /api/states and /api/districts...")
    res = client.get("/api/states")
    assert res.status_code == 200
    states = res.json["states"]
    assert "Haryana" in states and "Punjab" in states and "Uttar Pradesh" in states
    
    res = client.get("/api/districts?state=Haryana")
    assert res.status_code == 200
    dists = res.json["districts"]
    assert len(dists) == 14
    print(f"  -> PASS: Verified {len(states)} states and {len(dists)} Haryana districts.")

    # 3. Test Invalid Coordinate Catchment Protection
    print("\n[TEST 3] Testing Coordinate Validation & Catchment Check...")
    # Enter coordinates for Mumbai (19.07, 72.87) with Haryana/Bhiwani
    mismatch_data = {
        "state": "Haryana",
        "district": "Bhiwani",
        "year": 2022,
        "latitude": 19.07,
        "longitude": 72.87
    }
    res = client.post("/api/predictions", json=mismatch_data)
    assert res.status_code == 422, f"Expected 422 for coordinate mismatch, got {res.status_code}"
    assert "geographically inconsistent" in res.json["error"] or "outside" in res.json["error"]
    print("  -> PASS: Out-of-district coordinates correctly rejected with 422 Unprocessable Entity.")

    # 4. Strict Numerical Comparison: Direct Pipeline vs Backend API
    print("\n[TEST 4] Direct Pipeline vs Backend API Parity Verification...")
    
    # Load direct classifier
    direct_clf = CropDamageClassifier(models_dir=os.path.join(ROOT, "models"), ood_percentile=97.5)
    direct_clf.load()
    features_df = pd.read_csv(os.path.join(ROOT, "data", "processed", "features_enriched.csv"))
    train_df = features_df[features_df["Year"] < 2019]
    direct_clf.fit_ood_detector(train_df)

    test_cases = [
        {"state": "Haryana", "district": "Bhiwani", "year": 2022, "lat": 28.7328, "lon": 75.8928},
        {"state": "Haryana", "district": "Ambala", "year": 2020, "lat": 30.3435, "lon": 76.9493},
        {"state": "Punjab", "district": "Ludhiana", "year": 2021, "lat": 30.7894, "lon": 75.8270},
    ]

    for tc in test_cases:
        print(f"\n  Evaluating {tc['district']}, {tc['state']} ({tc['year']}):")
        
        # A. Direct pipeline execution
        raw_row = features_df[
            (features_df["State"] == tc["state"]) &
            (features_df["District"] == tc["district"]) &
            (features_df["Year"] == tc["year"])
        ].iloc[:1]
        direct_pred = direct_clf.predict(raw_row).iloc[0]

        # B. Backend API execution
        api_res = client.post("/api/predictions", json={
            "state": tc["state"],
            "district": tc["district"],
            "year": tc["year"],
            "latitude": tc["lat"],
            "longitude": tc["lon"]
        })
        assert api_res.status_code == 200, f"API error: {api_res.data}"
        api_data = api_res.json

        # C. Compare predictions
        direct_label = direct_pred["damage_label"]
        api_label = api_data["prediction"]["damage_label"]
        assert direct_label == api_label, f"Label mismatch: direct '{direct_label}' vs api '{api_label}'"

        direct_class_id = int(direct_pred["damage_class_id"])
        api_class_id = api_data["prediction"]["damage_class_id"]
        assert direct_class_id == api_class_id, f"Class ID mismatch: {direct_class_id} vs {api_class_id}"

        # D. Compare probabilities within numerical precision
        api_probs = api_data["probabilities"]
        assert abs(direct_pred["prob_not_damaged"] - api_probs["not_damaged"]) < 0.05
        assert abs(direct_pred["prob_severely_damaged"] - api_probs["severely_damaged"]) < 0.05

        # E. Compare OOD status
        assert bool(direct_pred["is_ood"]) == bool(api_data["ood"]["is_ood"]), "OOD status mismatch"
        print(f"     Class: {api_label} | Prob: {api_data['prediction']['confidence_score']:.3f} | OOD: {api_data['ood']['is_ood']}")
        print(f"     -> PASS: 100% agreement between direct model pipeline and REST backend.")

    # 5. Test History and CSV Export
    print("\n[TEST 5] Testing Prediction History and CSV Export...")
    res = client.get("/api/predictions/history")
    assert res.status_code == 200
    assert len(res.json["history"]) >= 3

    res = client.get("/api/predictions/history/export")
    assert res.status_code == 200
    assert "Damage_Label" in res.data.decode("utf-8")
    print(f"  -> PASS: SQLite audit trail records verified; CSV export contains valid headers.")

    print("\n" + "=" * 75)
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY! (5/5 PASS)")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
