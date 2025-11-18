#!/usr/bin/env python3
# app.py — Flask server for model + optional WeatherAPI proxy
"""
Flask app that:
 - Serves static frontend files from ./static (index.html, script.js, style.css, model_data.json)
 - Exposes /model_data.json for front-end fetch('model_data.json')
 - Exposes POST /predict that accepts {"features": [...]} or {"features": {"AQI": .., ...}}, optional "k"
 - Loads model_data.json on startup and uses sklearn.NearestNeighbors for KNN mean prediction
 - Returns both raw prediction (float) and ceiling integer
 - Optional: /_weather_proxy?q=<city> to proxy WeatherAPI requests using WEATHERAPI_KEY from env
"""

from flask import Flask, request, jsonify, send_from_directory, abort, make_response
from flask_cors import CORS
import os
import json
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
import requests

# Optional: load .env in development (do not commit .env)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Configuration ----------
STATIC_FOLDER = "static"
MODEL_JSON_FILENAME = "model_data.json"
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("DEBUG", "true").lower() in ("1", "true", "yes")
# -----------------------------------

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='/static')
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Globals (populated by load_model_json)
model_md = None
FEATURES = None
X_train = None
y_train = None
scaler_mean = None
scaler_scale = None
input_fill = None
k_default = 5
_nn = None
CITIES = None


def locate_model_file():
    """Return path to model_data.json in static folder or project root."""
    here = os.path.dirname(__file__)
    static_path = os.path.join(here, STATIC_FOLDER, MODEL_JSON_FILENAME)
    root_path = os.path.join(here, MODEL_JSON_FILENAME)
    if os.path.exists(static_path):
        return static_path
    if os.path.exists(root_path):
        return root_path
    return None


def load_model_json():
    """Load model JSON and prepare numpy arrays and NearestNeighbors."""
    global model_md, FEATURES, X_train, y_train, scaler_mean, scaler_scale, input_fill, k_default, _nn, CITIES

    path = locate_model_file()
    if path is None:
        logger.warning("model_data.json not found in static/ or project root")
        return False

    logger.info("Loading model JSON from: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        model_md = json.load(f)

    # -- Features
    FEATURES = model_md.get("feature_names") or model_md.get("features") or model_md.get("FEATURES")
    if not isinstance(FEATURES, list) or len(FEATURES) == 0:
        raise RuntimeError("model_data.json must include 'feature_names' (list)")

    # -- Training arrays
    if "X_train" not in model_md or "y_train" not in model_md:
        raise RuntimeError("model_data.json must include 'X_train' and 'y_train' arrays")

    X_train = np.array(model_md["X_train"], dtype=float)
    y_train = np.array(model_md["y_train"], dtype=float)
    if X_train.ndim != 2:
        raise RuntimeError("X_train should be 2D")
    if X_train.shape[0] != y_train.shape[0]:
        raise RuntimeError("X_train and y_train must have same number of rows")

    n_features = X_train.shape[1]
    scaler_mean = np.array(model_md.get("scaler_mean", [0.0] * n_features), dtype=float)
    scaler_scale = np.array(model_md.get("scaler_scale", [1.0] * n_features), dtype=float)
    input_fill = np.array(model_md.get("input_fill", [0.0] * n_features), dtype=float)

    if scaler_mean.shape[0] != n_features or scaler_scale.shape[0] != n_features or input_fill.shape[0] != n_features:
        raise RuntimeError("Length of scaler_mean/scaler_scale/input_fill must match number of features in X_train")

    k_default = int(model_md.get("k", k_default))

    # fit NearestNeighbors on (already scaled) X_train
    _nn = NearestNeighbors(metric="euclidean", n_jobs=-1)
    _nn.fit(X_train)

    # Prepare city list if present
    if isinstance(model_md.get("cities"), list):
        CITIES = model_md.get("cities")
    elif isinstance(model_md.get("city_static"), dict):
        # keys are city names
        CITIES = list(model_md.get("city_static").keys())
    else:
        # fallback list (small sample) — replace or supply via model_data.json for full top-30
        CITIES = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru", "Hyderabad", "Ahmedabad", "Pune", "Surat", "Jaipur"]

    logger.info("Model loaded: n_train=%d, n_features=%d, default_k=%d, n_cities=%d",
                X_train.shape[0], X_train.shape[1], k_default, len(CITIES))
    return True


def prepare_input_array(features):
    """
    Accepts dict mapping feature_name -> value or list/array of numbers (same order as FEATURES).
    Returns numpy array scaled the same way X_train is scaled: (x - mean) / scale
    """
    if features is None:
        raise ValueError("features payload is null")

    if isinstance(features, dict):
        arr = np.array([features.get(f, np.nan) for f in FEATURES], dtype=float)
    else:
        arr = np.array(features, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != len(FEATURES):
            raise ValueError(f"When features is a list it must have length {len(FEATURES)}")

    # fill missing (use input_fill)
    nan_mask = np.isnan(arr)
    if np.any(nan_mask):
        arr[nan_mask] = input_fill[nan_mask]

    # safe scaling (avoid division by zero)
    safe_scale = np.where(scaler_scale == 0.0, 1e-6, scaler_scale)
    arr_scaled = (arr - scaler_mean) / safe_scale
    return arr_scaled


def knn_neighbors_mean_and_indices(x_scaled, k):
    """Return tuple (mean_float, neighbors_indices) for the k nearest neighbors."""
    if _nn is None:
        raise RuntimeError("NearestNeighbors model not fitted")
    n_train = X_train.shape[0]
    k_eff = max(1, min(int(k), n_train))
    dists, idx = _nn.kneighbors(x_scaled.reshape(1, -1), n_neighbors=k_eff)
    neighbors_idx = idx[0]
    mean_val = float(np.mean(y_train[neighbors_idx]))
    return mean_val, neighbors_idx


# ----- Routes ----- #

@app.route("/")
def index():
    index_path = os.path.join(app.static_folder or STATIC_FOLDER, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder or STATIC_FOLDER, "index.html")
    return "<h3>index.html not found in ./static — place your frontend files in the static/ folder</h3>", 404


@app.route("/model_data.json")
def serve_model_json():
    static_file = os.path.join(app.static_folder or STATIC_FOLDER, MODEL_JSON_FILENAME)
    if os.path.exists(static_file):
        # send_from_directory will set proper headers
        return send_from_directory(app.static_folder or STATIC_FOLDER, MODEL_JSON_FILENAME)
    root_file = os.path.join(os.path.dirname(__file__), MODEL_JSON_FILENAME)
    if os.path.exists(root_file):
        return send_from_directory(os.path.dirname(__file__), MODEL_JSON_FILENAME)
    abort(404, description="model_data.json not found")


@app.route("/cities")
def cities():
    """Return canonical list of cities for the frontend chips."""
    if CITIES:
        return jsonify({"cities": CITIES})
    return jsonify({"cities": []})


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON:
      { "features": [...] } or { "features": {"AQI":..., "PM2.5":..., ...} }
    Optional: "k": integer to override default
    Response:
      { "prediction_raw": <float>, "prediction": <int_ceiling>, "k_used": int, "n_train": int, "neighbors": [idx...] }
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON in request body"}), 400

    if "features" not in payload:
        return jsonify({"error": "Missing 'features' field in request body"}), 400

    features = payload["features"]
    k = payload.get("k", k_default)

    try:
        x_scaled = prepare_input_array(features)
    except Exception as e:
        return jsonify({"error": f"Invalid features: {str(e)}"}), 400

    try:
        mean_val, neighbors_idx = knn_neighbors_mean_and_indices(x_scaled, k)
        pred_ceiling = int(math.ceil(mean_val)) if (mean_val is not None and not math.isnan(mean_val)) else None
        response = {
            "prediction_raw": mean_val,
            "prediction": pred_ceiling,
            "k_used": int(k),
            "n_train": int(X_train.shape[0]),
            "neighbors": [int(i) for i in neighbors_idx]
        }
        return jsonify(response)
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/model_info")
def model_info():
    if model_md is None:
        return jsonify({"loaded": False})
    return jsonify({
        "loaded": True,
        "n_train": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "features": FEATURES,
        "k_default": int(k_default),
        "cities_available": CITIES
    })


@app.route("/_weather_proxy")
def weather_proxy():
    """
    Proxy to WeatherAPI current.json endpoint.
    Query param: q (city or lat,lon)
    Requires WEATHERAPI_KEY in environment.
    Example: GET /_weather_proxy?q=Delhi
    """
    q = request.args.get("q") or request.args.get("city")
    if not q:
        return jsonify({"error": "Missing query parameter 'q' (city or lat,lon)"}), 400

    key = os.environ.get("WEATHERAPI_KEY")
    if not key:
        return jsonify({"error": "Server missing WEATHERAPI_KEY environment variable"}), 500

    url = f"https://api.weatherapi.com/v1/current.json?key={key}&q={q}&aqi=yes"
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException as e:
        logger.exception("Error contacting WeatherAPI")
        return jsonify({"error": "Failed to contact WeatherAPI", "detail": str(e)}), 502

    # Proxy response content and status code
    response = make_response(resp.content, resp.status_code)
    response.headers["Content-Type"] = resp.headers.get("Content-Type", "application/json")
    return response


# health check
@app.route("/_ping")
def ping():
    return jsonify({"ok": True})


if __name__ == "__main__":
    try:
        ok = load_model_json()
        if not ok:
            logger.warning("Model JSON not loaded — client-mode will not have model_data.json available")
    except Exception as e:
        logger.exception("Failed to load model_data.json on startup: %s", e)
        raise

    logger.info("Starting Flask server on http://%s:%s (debug=%s)", HOST, PORT, DEBUG)
    app.run(host=HOST, port=PORT, debug=DEBUG)
