# app.py
"""
Flask app that:
 - Serves static frontend files from ./static (index.html, script.js, style.css, model_data.json)
 - Exposes /model_data.json for front-end fetch('model_data.json')
 - Exposes POST /predict that accepts {"features": [...]} or {"features": {"AQI": .., ...}}, optional "k"
 - Loads model_data.json on startup and uses sklearn NearestNeighbors for KNN mean prediction

Change: the output of the prediction returned by /predict is now the CEILING (integer) of the KNN mean.
"""

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import os
import json
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math  # <-- added for ceiling

# ---------- Configuration ----------
STATIC_FOLDER = "static"                # folder where index.html, script.js, style.css, model_data.json live
MODEL_JSON_FILENAME = "model_data.json" # file name expected inside STATIC_FOLDER
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True
# -----------------------------------

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='/static')
CORS(app)  # enable CORS for testing from file:// or other origins

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals to hold model arrays and neighbors model (populated on startup)
model_md = None          # raw dict loaded from JSON
FEATURES = None          # ordered feature names (list)
X_train = None           # numpy array (n_train, n_features) — expected SCALED or as exported
y_train = None           # numpy array (n_train,)
scaler_mean = None       # numpy array (n_features,)
scaler_scale = None      # numpy array (n_features,)
input_fill = None        # numpy array (n_features,) used to impute missing values
k_default = 5            # fallback k
_nn = None               # fitted NearestNeighbors on X_train


def locate_model_file():
    """Return the path to model_data.json (first check STATIC_FOLDER, then project root)."""
    static_path = os.path.join(os.path.dirname(__file__), STATIC_FOLDER, MODEL_JSON_FILENAME)
    root_path = os.path.join(os.path.dirname(__file__), MODEL_JSON_FILENAME)
    if os.path.exists(static_path):
        return static_path
    if os.path.exists(root_path):
        return root_path
    return None


def load_model_json():
    """Load model_data.json and prepare numpy arrays and NearestNeighbors."""
    global model_md, FEATURES, X_train, y_train, scaler_mean, scaler_scale, input_fill, k_default, _nn

    path = locate_model_file()
    if path is None:
        logger.error("model_data.json not found - expected in %s or %s", STATIC_FOLDER, os.getcwd())
        return False

    logger.info("Loading model JSON from: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        model_md = json.load(f)

    # Validate expected fields and fill defaults
    FEATURES = model_md.get("feature_names") or model_md.get("features") or model_md.get("FEATURES")
    if not isinstance(FEATURES, list) or len(FEATURES) == 0:
        raise RuntimeError("model_data.json must include a 'feature_names' (list) or 'features' array")

    # X_train & y_train
    if "X_train" not in model_md or "y_train" not in model_md:
        raise RuntimeError("model_data.json must include 'X_train' and 'y_train' arrays")

    X_train = np.array(model_md["X_train"], dtype=float)
    y_train = np.array(model_md["y_train"], dtype=float)
    if X_train.ndim != 2:
        raise RuntimeError("X_train should be a 2D array")
    if X_train.shape[0] != y_train.shape[0]:
        raise RuntimeError("X_train and y_train must have the same number of rows")

    # scaler and imputer defaults
    n_features = X_train.shape[1]
    scaler_mean = np.array(model_md.get("scaler_mean", [0.0] * n_features), dtype=float)
    scaler_scale = np.array(model_md.get("scaler_scale", [1.0] * n_features), dtype=float)
    input_fill = np.array(model_md.get("input_fill", [0.0] * n_features), dtype=float)

    if scaler_mean.shape[0] != n_features or scaler_scale.shape[0] != n_features or input_fill.shape[0] != n_features:
        raise RuntimeError("Length of scaler_mean/scaler_scale/input_fill must match number of features in X_train")

    k_default = int(model_md.get("k", k_default))
    # Fit a NearestNeighbors instance on the (already scaled) X_train.
    # We can call kneighbors with different n_neighbors at request-time.
    _nn = NearestNeighbors(metric='euclidean', n_jobs=-1)
    _nn.fit(X_train)

    logger.info("Model loaded: n_train=%d, n_features=%d, default_k=%d", X_train.shape[0], X_train.shape[1], k_default)
    return True


def prepare_input_array(features):
    """
    Accepts:
      - list/tuple/array of numbers of length len(FEATURES)
      - dict mapping feature_name -> value (feature names expected to match FEATURES)
    Returns: 1D numpy array scaled to same representation as X_train (shape (n_features,))
    Replaces missing values (NaN) with input_fill, then scales using (x - mean) / scale.
    """
    if features is None:
        raise ValueError("features payload is null")

    # Build numeric array in the order of FEATURES
    if isinstance(features, dict):
        arr = np.array([features.get(f, np.nan) for f in FEATURES], dtype=float)
    else:
        arr = np.array(features, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != len(FEATURES):
            raise ValueError(f"When features is a list it must have length {len(FEATURES)} (received shape {arr.shape})")

    # Replace NaN with input_fill
    nan_mask = np.isnan(arr)
    if np.any(nan_mask):
        arr[nan_mask] = input_fill[nan_mask]

    # Scale (defensive: avoid divide-by-zero)
    safe_scale = np.where(scaler_scale == 0.0, 1e-6, scaler_scale)
    arr_scaled = (arr - scaler_mean) / safe_scale
    return arr_scaled


def knn_mean_predict_one(x_scaled, k):
    """Return the mean label among k nearest neighbors to x_scaled (1D array)."""
    if _nn is None:
        raise RuntimeError("NearestNeighbors model is not fitted")
    n_train = X_train.shape[0]
    k_eff = max(1, min(int(k), n_train))
    # kneighbors accepts n_neighbors
    dists, idx = _nn.kneighbors(x_scaled.reshape(1, -1), n_neighbors=k_eff)
    neighbors_idx = idx[0]
    return float(np.mean(y_train[neighbors_idx]))


# ----- Routes ----- #

# Serve root index.html from static folder.
@app.route("/")
def index():
    # index.html must live inside STATIC_FOLDER
    index_path = os.path.join(app.static_folder or STATIC_FOLDER, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder or STATIC_FOLDER, "index.html")
    else:
        return "<h3>index.html not found in ./static — place your frontend files in the static/ folder</h3>", 404


# Provide model_data.json at the root path so client can `fetch('model_data.json')`
@app.route("/model_data.json")
def serve_model_json():
    # Prefer the static folder copy
    static_file = os.path.join(app.static_folder or STATIC_FOLDER, MODEL_JSON_FILENAME)
    if os.path.exists(static_file):
        return send_from_directory(app.static_folder or STATIC_FOLDER, MODEL_JSON_FILENAME)
    # fallback to project root
    root_file = os.path.join(os.path.dirname(__file__), MODEL_JSON_FILENAME)
    if os.path.exists(root_file):
        return send_from_directory(os.path.dirname(__file__), MODEL_JSON_FILENAME)
    abort(404, description="model_data.json not found on server")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON body:
      { "features": [...]}  # list of length = n_features
    OR
      { "features": { "AQI": 10, "PM2.5": 20, ... } }  # dict mapping
    Optional: "k": integer to override default k
    Response: { "prediction": <integer_ceiling> } or { "error": "..." }
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
        pred = knn_mean_predict_one(x_scaled, k)
        # Return the ceiling of the predicted number as an integer
        pred_ceiling = int(math.ceil(pred))
        return jsonify({"prediction": pred_ceiling})
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# Optional: an endpoint to get model info (useful for debugging from the front-end)
@app.route("/model_info")
def model_info():
    if model_md is None:
        return jsonify({"loaded": False})
    return jsonify({
        "loaded": True,
        "n_train": X_train.shape[0],
        "n_features": X_train.shape[1],
        "features": FEATURES,
        "k_default": int(k_default)
    })


if __name__ == "__main__":
    # Load model JSON on startup (fatal if loading fails)
    try:
        ok = load_model_json()
        if not ok:
            logger.warning("Model JSON not loaded — front-end client-mode will not work until model_data.json is available")
    except Exception as e:
        logger.exception("Failed to load model_data.json on startup: %s", e)
        raise

    logger.info("Starting Flask server on http://%s:%d", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=DEBUG)
