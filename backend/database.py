"""
Database Service (SQLite)
=========================
Stores prediction history, coordinates, extracted features, confidence, OOD, and rule warnings.
"""

import os
import sqlite3
import json
import io
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB_PATH = os.path.join(ROOT, "data", "predictions.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT NOT NULL,
            district TEXT NOT NULL,
            year INTEGER NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            damage_class_id INTEGER NOT NULL,
            damage_label TEXT NOT NULL,
            confidence_level TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            prob_not_damaged REAL NOT NULL,
            prob_less_damaged REAL NOT NULL,
            prob_moderately_damaged REAL NOT NULL,
            prob_severely_damaged REAL NOT NULL,
            is_ood INTEGER NOT NULL,
            ood_score REAL,
            ood_threshold REAL,
            rule_warnings TEXT,
            pred_exposure_tmax_degC REAL,
            feature_status TEXT,
            features_json TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(record: Dict[str, Any]) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    
    rule_warnings = record.get("rule_warnings")
    if isinstance(rule_warnings, list):
        rule_warnings_str = json.dumps(rule_warnings)
    elif isinstance(rule_warnings, str):
        rule_warnings_str = json.dumps([rule_warnings]) if rule_warnings else "[]"
    else:
        rule_warnings_str = "[]"
        
    features_json_str = json.dumps(record.get("features", {}))
    created_at = record.get("timestamp") or datetime.utcnow().isoformat() + "Z"

    cursor.execute("""
        INSERT INTO predictions (
            state, district, year, latitude, longitude,
            damage_class_id, damage_label, confidence_level, confidence_score,
            prob_not_damaged, prob_less_damaged, prob_moderately_damaged, prob_severely_damaged,
            is_ood, ood_score, ood_threshold, rule_warnings, pred_exposure_tmax_degC,
            feature_status, features_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["state"],
        record["district"],
        record["year"],
        float(record["latitude"]),
        float(record["longitude"]),
        int(record["damage_class_id"]),
        record["damage_label"],
        record["confidence_level"],
        float(record["confidence_score"]),
        float(record["prob_not_damaged"]),
        float(record["prob_less_damaged"]),
        float(record["prob_moderately_damaged"]),
        float(record["prob_severely_damaged"]),
        1 if record.get("is_ood") else 0,
        float(record.get("ood_score", 0.0)) if record.get("ood_score") is not None else None,
        float(record.get("ood_threshold", 53.72)) if record.get("ood_threshold") is not None else 53.72,
        rule_warnings_str,
        float(record.get("pred_exposure_tmax_degC", 0.0)) if record.get("pred_exposure_tmax_degC") is not None else None,
        record.get("feature_status", "Google Earth Engine"),
        features_json_str,
        created_at
    ))
    
    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pred_id


def get_prediction_by_id(pred_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_dict(row, include_features=True)


def get_history(state: Optional[str] = None, district: Optional[str] = None, 
                year: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    
    if state:
        query += " AND state = ?"
        params.append(state)
    if district:
        query += " AND district = ?"
        params.append(district)
    if year:
        query += " AND year = ?"
        params.append(year)
        
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [_row_to_dict(r, include_features=False) for r in rows]


def export_history_to_csv() -> str:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        "ID", "State", "District", "Year", "Latitude", "Longitude",
        "Damage_Class_ID", "Damage_Label", "Confidence_Level", "Confidence_Score",
        "Prob_Not_Damaged", "Prob_Less_Damaged", "Prob_Moderately_Damaged", "Prob_Severely_Damaged",
        "Is_OOD", "OOD_Score", "Rule_Warnings", "Pred_Exposure_Tmax_C", "Feature_Source", "Created_At"
    ])
    
    for r in rows:
        writer.writerow([
            r["id"], r["state"], r["district"], r["year"], r["latitude"], r["longitude"],
            r["damage_class_id"], r["damage_label"], r["confidence_level"], r["confidence_score"],
            r["prob_not_damaged"], r["prob_less_damaged"], r["prob_moderately_damaged"], r["prob_severely_damaged"],
            bool(r["is_ood"]), r["ood_score"], r["rule_warnings"], r["pred_exposure_tmax_degC"],
            r["feature_status"], r["created_at"]
        ])
        
    return output.getvalue()


def _row_to_dict(row: sqlite3.Row, include_features: bool = False) -> Dict[str, Any]:
    try:
        rule_warnings = json.loads(row["rule_warnings"]) if row["rule_warnings"] else []
    except Exception:
        rule_warnings = [row["rule_warnings"]] if row["rule_warnings"] else []
        
    d = {
        "id": row["id"],
        "state": row["state"],
        "district": row["district"],
        "year": row["year"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "damage_class_id": row["damage_class_id"],
        "damage_label": row["damage_label"],
        "confidence_level": row["confidence_level"],
        "confidence_score": row["confidence_score"],
        "probabilities": {
            "not_damaged": row["prob_not_damaged"],
            "less_damaged": row["prob_less_damaged"],
            "moderately_damaged": row["prob_moderately_damaged"],
            "severely_damaged": row["prob_severely_damaged"],
        },
        "ood": {
            "is_ood": bool(row["is_ood"]),
            "ood_score": row["ood_score"],
            "threshold": row["ood_threshold"] or 53.72
        },
        "rule_warnings": rule_warnings,
        "pred_exposure_tmax_degC": row["pred_exposure_tmax_degC"],
        "feature_status": row["feature_status"],
        "created_at": row["created_at"]
    }
    
    if include_features:
        try:
            d["features"] = json.loads(row["features_json"]) if row["features_json"] else {}
        except Exception:
            d["features"] = {}
            
    return d


# Initialize DB immediately
init_db()
