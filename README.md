# 🌾 Machine Learning-Based Wheat Crop Damage Estimation Due to Heat Waves

### Using Multi-Source Satellite and Meteorological Data

A research-grade web application that connects a trained **ExtraTreesClassifier** with **Google Earth Engine (GEE)** to provide location-based wheat crop damage predictions across the **Indo-Gangetic Wheat Belt** (Punjab, Haryana, and Western Uttar Pradesh).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Frontend Pages](#frontend-pages)
- [Model Details](#model-details)
- [Scientific Integrity](#scientific-integrity)
- [Project Structure](#project-structure)

---

## Overview

This application enables users to:

1. **Select a location** — State → District → Year → Latitude/Longitude (via dropdown or interactive map).
2. **Extract satellite features** — MODIS (NDVI, EVI, NDMI, LST), ERA5-Land (Tmax, Soil Moisture, VPD), and CHIRPS (Precipitation) from Google Earth Engine.
3. **Run damage classification** — The pre-trained ExtraTreesClassifier predicts wheat crop damage severity (4-class: Not Damaged, Less Damaged, Moderately Damaged, Severely Damaged).
4. **View results** — Including class probabilities, confidence scores, OOD (Out-of-Distribution) detection, and rule-based verification warnings.

### Coverage

| Dimension         | Details                                             |
|-------------------|-----------------------------------------------------|
| **States**        | Punjab (13 districts), Haryana (14), UP (~10)       |
| **Total Districts** | 37                                               |
| **Year Range**    | 2003–2024 (Training: 2003–2018, Holdout: 2019–2024)|
| **Satellite Data**| MODIS/061 (MOD13Q1, MOD11A2), ERA5-Land, CHIRPS    |
| **Features**      | 89 phenology-anchored classification features       |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     REACT FRONTEND                          │
│  (Vite + Tailwind CSS + Leaflet + Recharts)                │
│                                                             │
│  Overview │ Predict │ District │ Historical │ Model │ History│
└─────────────────────────┬───────────────────────────────────┘
                          │  Axios (proxy /api → :5000)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK BACKEND (REST API)                  │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ GEE Service │  │ Model Service│  │  SQLite Database   │ │
│  │             │  │              │  │                    │ │
│  │ • MODIS     │  │ • ExtraTrees │  │ • Prediction       │ │
│  │ • ERA5-Land │  │ • OOD (PCA   │  │   History          │ │
│  │ • CHIRPS    │  │   Mahalanobis│  │ • Audit Trail      │ │
│  │ • FAO/GAUL  │  │ • Rule-Based │  │ • CSV Export       │ │
│  │   Boundaries│  │   Verifier)  │  │                    │ │
│  └─────────────┘  └──────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               GOOGLE EARTH ENGINE (Cloud)                   │
│                                                             │
│  MODIS/061/MOD13Q1  •  MODIS/061/MOD11A2                   │
│  ECMWF/ERA5_LAND/DAILY_AGGR  •  UCSB-CHG/CHIRPS/DAILY     │
│  FAO/GAUL/2015/level2 (District Boundaries)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Software            | Version    | Purpose                            |
|---------------------|------------|------------------------------------|
| **Python**          | ≥ 3.10     | Backend, ML model, GEE API         |
| **Node.js**         | ≥ 18.0     | Frontend build toolchain           |
| **npm**             | ≥ 9.0      | Package management                 |
| **GEE Account**     | Authenticated | Google Earth Engine access       |

### Google Earth Engine Authentication

Before running the backend, you must authenticate with Google Earth Engine:

```powershell
# Authenticate (opens browser for OAuth)
earthengine authenticate

# Set the GEE project
earthengine set_project crop-heatwave-minor-project
```

If you have a service account key, set the path in `.env`:
```
GEE_KEY_FILE=path/to/service-account-key.json
```

---

## Setup & Installation

### 1. Clone / Navigate to the Project

```powershell
cd "C:\Users\ACER\Downloads\MinorProject_crop (2)\MinorProject_crop\MinorProject_crop"
```

### 2. Backend Setup

```powershell
# Activate the Python virtual environment
.\.venv\Scripts\activate

# Install backend dependencies
pip install -r backend\requirements.txt
```

### 3. Frontend Setup

```powershell
# Navigate to the frontend directory
cd frontend

# Install Node dependencies
npm install

# Return to project root
cd ..
```

### 4. Environment Configuration

Copy the example environment file and adjust if needed:

```powershell
copy .env.example .env
```

Default `.env` values:
```
PORT=5000
GEE_PROJECT_ID=crop-heatwave-minor-project
```

---

## Running the Application

### Start the Backend (Terminal 1)

```powershell
cd "C:\Users\ACER\Downloads\MinorProject_crop (2)\MinorProject_crop\MinorProject_crop"
.\.venv\Scripts\activate
python backend\app.py
```

The backend starts on **http://localhost:5000**. You should see:
```
GEE successfully initialized with project 'crop-heatwave-minor-project'
Initializing ModelService...
OOD detector fitted on 592 training rows. Threshold: 53.72
Starting Crop Damage Estimation Backend on port 5000...
```

### Start the Frontend (Terminal 2)

```powershell
cd "C:\Users\ACER\Downloads\MinorProject_crop (2)\MinorProject_crop\MinorProject_crop\frontend"
npm run dev
```

The frontend starts on **http://localhost:5173**. Open this URL in your browser.

> **Note:** The Vite dev server proxies all `/api/*` requests to the Flask backend on port 5000.

---

## API Documentation

### Health Check

```
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-11-01T17:00:00Z",
  "services": {
    "model_classifier": "loaded",
    "ood_detector": "ready",
    "google_earth_engine": "connected",
    "database": "sqlite_ready"
  }
}
```

### Geographic Data

```
GET /api/states                      → List of supported states
GET /api/districts?state=Haryana     → Districts for a state (with centroids)
GET /api/years                       → Available prediction years
```

### Prediction (Core Endpoint)

```
POST /api/predictions
Content-Type: application/json

{
  "state": "Haryana",
  "district": "Bhiwani",
  "year": 2022,
  "latitude": 28.7328,
  "longitude": 75.8928
}
```

**Response:**
```json
{
  "success": true,
  "prediction_id": 1,
  "location": {
    "state": "Haryana",
    "district": "Bhiwani",
    "year": 2022,
    "latitude": 28.7328,
    "longitude": 75.8928
  },
  "prediction": {
    "damage_class_id": 3,
    "damage_label": "SEVERELY DAMAGED",
    "confidence_level": "MEDIUM",
    "confidence_score": 0.4322,
    "pred_exposure_tmax_degC": 30.45
  },
  "probabilities": {
    "not_damaged": 0.15,
    "less_damaged": 0.10,
    "moderately_damaged": 0.32,
    "severely_damaged": 0.43
  },
  "ood": {
    "is_ood": true,
    "ood_score": 61.91,
    "threshold": 53.72
  },
  "rule_warnings": [],
  "feature_extraction": {
    "status": "completed",
    "source": "archive_with_gee_overlay",
    "extraction_time_ms": 1250
  }
}
```

### Prediction History

```
GET /api/predictions/<id>                   → Single prediction details
GET /api/predictions/history                → All predictions (filterable)
GET /api/predictions/history?state=Haryana  → Filter by state
GET /api/predictions/history/export         → Download CSV export
```

### Model Metadata & Performance

```
GET /api/model/metadata      → Feature list, class labels, year ranges
GET /api/model/performance   → Accuracy, F1, per-class precision/recall, confusion matrix
```

### District Historical Data

```
GET /api/districts/historical?state=Haryana&district=Bhiwani
```

Returns all historical feature rows for the district (NDVI, LST, Tmax, rainfall, etc.) for trend analysis.

---

## Frontend Pages

| Page | Description |
|------|-------------|
| **Overview** | Hero banner, key metrics (76.13% Acc, 90.59% AUC), 5-step pipeline architecture, coverage & scientific integrity notes |
| **Predict** | Full prediction form: State/District/Year dropdowns, manual lat/lon inputs, interactive Leaflet map with draggable marker, result display with damage card, probability bars, OOD card, rule warnings, extracted features table |
| **District Analysis** | Per-district environmental indicators (NDVI, LST, Tmax, hot days, soil moisture, rainfall) with historical context |
| **Historical Trends** | Multi-decadal Recharts visualizations: thermal stress trends, heatwave frequency, NDVI evolution, soil moisture patterns |
| **Model Performance** | Holdout metrics, per-class precision/recall/F1 table, 4×4 confusion matrix heatmap, confidence calibration analysis, documented limitations |
| **Prediction History** | Filterable table of all predictions, detail modal with probabilities/OOD/rule warnings, CSV export |

---

## Model Details

### Classifier

- **Algorithm:** ExtraTreesClassifier (scikit-learn, balanced class weights)
- **Features:** 89 phenology-anchored features (classification) + 14 exposure features (regression)
- **Training Period:** 2003–2018 (592 district-year samples)
- **Holdout Period:** 2019–2024 (222 district-year samples, rolling-origin)

### Damage Classes

| Class ID | Label              | Color  |
|----------|--------------------|--------|
| 0        | NOT DAMAGED        | Green  |
| 1        | LESS DAMAGED       | Yellow |
| 2        | MODERATELY DAMAGED | Orange |
| 3        | SEVERELY DAMAGED   | Red    |

### Confidence Thresholds

| Score Range | Level  |
|-------------|--------|
| ≥ 0.60      | HIGH   |
| 0.40 – 0.59 | MEDIUM |
| < 0.40      | LOW    |

### OOD Detection

- **Method:** PCA-Mahalanobis Distance
- **Threshold:** 53.72 (97.5th percentile of training distribution)
- **Purpose:** Flag inputs whose feature profiles diverge from the training distribution

### Rule-Based Verification

Physical consistency checks that flag (but do not override) potential classification issues:
- Low thermal stress contradicting damage prediction
- NDVI anomalies inconsistent with damage severity
- Temperature-vegetation interaction warnings

---

## Scientific Integrity

> **This application transparently reports all model limitations.**

- Holdout accuracy is **76.13%** — this reflects test-period performance, not a live guarantee.
- **Severe damage recall is 30%** (6/20 holdout cases correctly identified). This limitation is prominently displayed.
- Macro F1-Score is **54.59%**, reflecting class imbalance challenges.
- **OOD detection warns** when inputs diverge from training distribution — predictions are still generated but flagged.
- Rule-based warnings **never override** the classifier's output. They are displayed as supplementary verification flags.
- Future years (2025+) trigger warnings about unseen satellite/weather conditions.

---

## Project Structure

```
MinorProject_crop/
├── backend/
│   ├── app.py                 # Flask REST API (all endpoints)
│   ├── model_service.py       # Singleton ML model wrapper
│   ├── gee_service.py         # Google Earth Engine integration
│   ├── database.py            # SQLite prediction storage
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example           # Environment template
│   └── test_integration.py    # 5-test integration suite
├── frontend/
│   ├── src/
│   │   ├── api.js             # Axios API client
│   │   ├── App.jsx            # Root app with tab navigation
│   │   ├── components/
│   │   │   ├── Navbar.jsx     # Top nav with GEE status indicator
│   │   │   ├── MapPicker.jsx  # Interactive Leaflet map
│   │   │   ├── ProbabilityBar.jsx  # 4-class probability chart
│   │   │   └── FeaturesTable.jsx   # Extracted features display
│   │   └── pages/
│   │       ├── Overview.jsx
│   │       ├── Predict.jsx
│   │       ├── DistrictAnalysis.jsx
│   │       ├── HistoricalAnalysis.jsx
│   │       ├── ModelPerformance.jsx
│   │       └── PredictionHistory.jsx
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── damage_classifier_v2.py        # CropDamageClassifier class
│   ├── feature_config.json            # 89 classification features
│   ├── crop_damage_classifier.joblib  # Trained ExtraTrees model
│   ├── heat_exposure_regressor.joblib # Tmax exposure regressor
│   ├── preprocessor_imputer.joblib    # Missing value imputer
│   ├── preprocessor_scaler.joblib     # Feature scaler
│   └── imputer_exposure.joblib        # Exposure imputer
├── data/
│   ├── processed/
│   │   ├── features_enriched.csv      # Full feature dataset (814 rows)
│   │   └── A_phenology_reference.csv  # Per-district peak DOY
│   └── external/
│       └── district_centroids.csv     # 37 district lat/lon centroids
├── scripts/                           # Training & evaluation scripts
├── reports/                           # Generated evaluation reports
├── .env.example                       # Root environment template
└── README.md                          # This file
```

---

## Running Integration Tests

```powershell
cd "C:\Users\ACER\Downloads\MinorProject_crop (2)\MinorProject_crop\MinorProject_crop"
.\.venv\Scripts\activate
python backend\test_integration.py
```

Expected output:
```
[TEST 1] Testing /api/health...                    -> PASS
[TEST 2] Testing /api/states and /api/districts... -> PASS
[TEST 3] Testing Coordinate Validation...          -> PASS
[TEST 4] Direct Pipeline vs Backend API Parity...  -> PASS (3/3 districts)
[TEST 5] Testing Prediction History and CSV...     -> PASS

ALL INTEGRATION TESTS PASSED SUCCESSFULLY! (5/5 PASS)
```

---

## License

Academic research project. For educational and research purposes only.
