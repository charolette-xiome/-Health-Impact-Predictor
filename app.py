from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors

# --- Config ---
MODEL_JSON = "model_data.json"   # must exist in same folder (or change path)
FEATURES = [
    'AQI','PM2.5','PM10','NO2','SO2','CO','O3',
    'temperature','humidity','wind_speed','precipitation',
    'population_density','green_cover_percentage'
]
DEFAULT_K = 1000
# ---------------

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from your front-end

# Load model_data.json produced by your training script
with open(MODEL_JSON, "r", encoding="utf-8") as f:
    md = json.load(f)

# Convert JSON arrays to numpy arrays
X_train = np.array(md["X_train"], dtype=float)        # this should be SCALED X_train (as in your exporter)
y_train = np.array(md["y_train"], dtype=float)
scaler_mean = np.array(md.get("scaler_mean", [0]*len(FEATURES)), dtype=float)
scaler_scale = np.array(md.get("scaler_scale", [1]*len(FEATURES)), dtype=float)
input_fill = np.array(md.get("input_fill", [0]*len(FEATURES)), dtype=float)
k_from_model = int(md.get("k", DEFAULT_K))

# prepare a fitted NearestNeighbors on X_train. We'll (re)create neighbours per request if k changes,
# but keep a base nn fitted for max_k = min(n_train, maybe large)
_nn = NearestNeighbors(metric='euclidean', n_jobs=-1)
_nn.fit(X_train)
n_train = X_train.shape[0]

def prepare_input(features):
    """
    Accepts either:
      - a list/array of length len(FEATURES) where order matches FEATURES
      - a dict mapping feature names -> values
    Returns: 1D numpy array of length D with NaNs replaced by input_fill, and scaled using scaler_mean/scale
    """
    if isinstance(features, dict):
        arr = np.array([features.get(f, np.nan) for f in FEATURES], dtype=float)
    else:
        arr = np.array(features, dtype=float)
        if arr.shape[0] != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} features, got {arr.shape}")

    # replace nan with input_fill
    nan_mask = np.isnan(arr)
    arr[nan_mask] = input_fill[nan_mask]

    # scale: (x - mean)/scale  -- follow same convention as your front-end / training exporter
    safe_scale = np.where(scaler_scale == 0, 1e-6, scaler_scale)
    arr_scaled = (arr - scaler_mean) / safe_scale
    return arr_scaled

def knn_mean_predict_one(x_scaled, k):
    k_eff = max(1, min(int(k), n_train))
    if k_eff != k:
        # if requested k > n_train this will cap silently
        k = k_eff
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    nbrs.fit(X_train)
    dists, idx = nbrs.kneighbors(x_scaled.reshape(1, -1))
    pred = float(np.mean(y_train[idx[0]]))
    return pred

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON"}), 400

    # Accept either { "features": [...] } or { "features": {...} }
    if "features" not in payload:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    features = payload["features"]
    # optional: payload may include 'k' to override
    k = int(payload.get("k", k_from_model))

    try:
        x_scaled = prepare_input(features)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # predict
    pred = knn_mean_predict_one(x_scaled, k)
    return jsonify({"prediction": pred})

import json
import numpy as np

# We’ll assume all variables from your training script still exist:
# X_train, y_train, imputer, scaler, etc.

# Prepare data for saving
model_data = {
    "k": 1000,
    "feature_names": FEATURES,
    "X_train": np.round(X_train, 6).tolist(),
    "y_train": np.round(y_train, 6).tolist(),
    "input_fill": imputer.statistics_.round(6).tolist(),
    "scaler_mean": scaler.mean_.round(6).tolist(),
    "scaler_scale": scaler.scale_.round(6).tolist(),
}

# Save model JSON
OUTPUT_MODEL_JSON = "model_data.json"
with open(OUTPUT_MODEL_JSON, "w", encoding="utf-8") as f:
    json.dump(model_data, f, indent=2)

print(f"✅ Saved model_data.json successfully to {OUTPUT_MODEL_JSON}")

with open("model_data.json", "r") as f:
    model_data = json.load(f)

X_train = np.array(model_data["X_train"])
y_train = np.array(model_data["y_train"])
scaler_mean = np.array(model_data["scaler_mean"])
scaler_scale = np.array(model_data["scaler_scale"])
input_fill = np.array(model_data["input_fill"])
FEATURES = model_data["feature_names"]
k = model_data["k"]

def prepare_input(data):
    arr = np.array(data, dtype=float)
    arr = np.where(np.isnan(arr), input_fill, arr)
    arr = (arr - scaler_mean) / scaler_scale
    return arr.reshape(1, -1)


if __name__ == "__main__":
    print("Starting server on http://0.0.0.0:5000 - loading model_data.json")
    app.run(host="0.0.0.0", port=5000, debug=True)
