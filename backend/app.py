"""
Flask REST API Backend
======================
Connects trained ExtraTreesClassifier and GEE service to provide
real-time crop heatwave damage estimation.
"""

import os
import sys
import json
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.database import (
    init_db, save_prediction, get_prediction_by_id,
    get_history, export_history_to_csv
)
from backend.model_service import ModelService
from backend.gee_service import GEEService

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load Singletons
model_service = ModelService.get_instance()
gee_service = GEEService.get_instance()


@app.route("/api/health", methods=["GET"])
def health():
    gee_ready, gee_err = gee_service.is_gee_ready()
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": {
            "model_classifier": "loaded",
            "ood_detector": "ready",
            "google_earth_engine": "connected" if gee_ready else "error",
            "database": "sqlite_ready"
        },
        "gee_details": {
            "initialized": gee_ready,
            "error": gee_err
        }
    })


@app.route("/api/states", methods=["GET"])
def get_states():
    return jsonify({
        "success": True,
        "states": gee_service.get_states()
    })


@app.route("/api/districts", methods=["GET"])
def get_districts():
    state = request.args.get("state")
    if not state:
        return jsonify({"success": False, "error": "Query param 'state' is required"}), 400
    
    districts = gee_service.get_districts_by_state(state)
    return jsonify({
        "success": True,
        "state": state,
        "districts": districts
    })


@app.route("/api/years", methods=["GET"])
def get_years():
    metadata = model_service.get_metadata()
    return jsonify({
        "success": True,
        "years": metadata["years_available"],
        "validated_years": metadata["validated_test_years"],
        "operational_future_years": metadata.get("operational_future_years", [2025, 2026, 2027]),
        "note": "Years 2019-2024 constitute the rolling-origin holdout test period. Years 2025 onwards are operational deployment periods with live GEE extraction."
    })


@app.route("/api/predictions", methods=["POST"])
def create_prediction():
    data = request.get_json() or {}
    state = data.get("state")
    district = data.get("district")
    year = data.get("year")
    lat = data.get("latitude")
    lon = data.get("longitude")

    if not state or not district or year is None or lat is None or lon is None:
        return jsonify({
            "success": False,
            "error": "Missing required fields: state, district, year, latitude, longitude"
        }), 400

    try:
        year = int(year)
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid format for year, latitude, or longitude"}), 400

    # 1. Location Validation
    valid, msg = gee_service.validate_location(state, district, lat, lon)
    if not valid:
        return jsonify({"success": False, "error": msg}), 422

    # 2. GEE Feature Extraction
    try:
        features_df, gee_meta = gee_service.extract_features(state, district, year, lat, lon)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Google Earth Engine extraction failure: {str(e)}"
        }), 502

    # 3. Model Inference
    try:
        prediction_output = model_service.predict_features(features_df)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Model inference failure: {str(e)}"
        }), 500

    timestamp = datetime.utcnow().isoformat() + "Z"

    # 4. Save to Database
    db_record = {
        "state": state,
        "district": district,
        "year": year,
        "latitude": lat,
        "longitude": lon,
        "damage_class_id": prediction_output["damage_class_id"],
        "damage_label": prediction_output["damage_label"],
        "confidence_level": prediction_output["confidence_level"],
        "confidence_score": prediction_output["confidence_score"],
        "prob_not_damaged": prediction_output["probabilities"]["not_damaged"],
        "prob_less_damaged": prediction_output["probabilities"]["less_damaged"],
        "prob_moderately_damaged": prediction_output["probabilities"]["moderately_damaged"],
        "prob_severely_damaged": prediction_output["probabilities"]["severely_damaged"],
        "is_ood": prediction_output["ood"]["is_ood"],
        "ood_score": prediction_output["ood"]["ood_score"],
        "ood_threshold": prediction_output["ood"]["threshold"],
        "rule_warnings": prediction_output["rule_warnings"],
        "pred_exposure_tmax_degC": prediction_output.get("pred_exposure_tmax_degC"),
        "feature_status": gee_meta.get("source", "Google Earth Engine"),
        "features": gee_meta.get("features_summary", {}),
        "timestamp": timestamp
    }
    pred_id = save_prediction(db_record)

    # 5. Build Final API Response
    response = {
        "success": True,
        "prediction_id": pred_id,
        "location": {
            "state": state,
            "district": district,
            "year": year,
            "latitude": lat,
            "longitude": lon
        },
        "prediction": {
            "damage_class_id": prediction_output["damage_class_id"],
            "damage_label": prediction_output["damage_label"],
            "damage_internal_name": prediction_output["damage_internal_name"],
            "confidence_level": prediction_output["confidence_level"],
            "confidence_score": prediction_output["confidence_score"],
            "pred_exposure_tmax_degC": prediction_output.get("pred_exposure_tmax_degC")
        },
        "probabilities": prediction_output["probabilities"],
        "ood": prediction_output["ood"],
        "rule_warnings": prediction_output["rule_warnings"],
        "feature_extraction": {
            "status": gee_meta.get("status", "completed"),
            "source": gee_meta.get("source", "Google Earth Engine"),
            "extraction_time_ms": gee_meta.get("extraction_time_ms", 0),
            "features_summary": gee_meta.get("features_summary", {})
        },
        "year_validation_warning": "Warning: Selected year is outside the 2019–2024 validated holdout period." if year < 2019 or year > 2024 else None,
        "timestamp": timestamp
    }
    return jsonify(response)


@app.route("/api/predictions/<int:prediction_id>", methods=["GET"])
def get_prediction(prediction_id):
    record = get_prediction_by_id(prediction_id)
    if not record:
        return jsonify({"success": False, "error": "Prediction record not found"}), 404
    return jsonify({"success": True, "prediction": record})


@app.route("/api/predictions/history", methods=["GET"])
def list_history():
    state = request.args.get("state")
    district = request.args.get("district")
    year = request.args.get("year", type=int)
    limit = request.args.get("limit", default=100, type=int)

    history = get_history(state=state, district=district, year=year, limit=limit)
    return jsonify({
        "success": True,
        "count": len(history),
        "history": history
    })


@app.route("/api/predictions/history/export", methods=["GET"])
def export_history():
    csv_data = export_history_to_csv()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=crop_damage_predictions_history.csv"}
    )


@app.route("/api/model/metadata", methods=["GET"])
def model_metadata():
    return jsonify({
        "success": True,
        "metadata": model_service.get_metadata()
    })


@app.route("/api/model/performance", methods=["GET"])
def model_performance():
    return jsonify({
        "success": True,
        "performance": model_service.get_performance_metrics()
    })


@app.route("/api/districts/historical", methods=["GET"])
def district_historical():
    state = request.args.get("state", default="")
    district = request.args.get("district", default="")
    if not district:
        return jsonify({"success": False, "error": "Parameter 'district' is required"}), 400

    records = model_service.get_historical_district(state, district)
    return jsonify({
        "success": True,
        "state": state,
        "district": district,
        "count": len(records),
        "data": records
    })


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Crop Damage Estimation Backend on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
