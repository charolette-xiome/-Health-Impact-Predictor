from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # allow your GitHub Pages frontend to call this API

model = joblib.load('knn_mean_model_k1000.pkl')
nbrs = model['nbrs']
X_train = np.array(model['X_train_scaled'])
y_train = np.array(model['y_train'])
k_default = int(model.get('k', 100))

def predict_knn_mean(x_query, k=k_default):
    xq = np.atleast_2d(np.array(x_query, dtype=float))
    distances, indices = nbrs.kneighbors(xq, n_neighbors=min(k, X_train.shape[0]))
    preds = np.mean(y_train[indices], axis=1)
    return float(preds[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data.get('features')
    if features is None:
        return jsonify({"error":"missing features"}), 400
    try:
        pred = predict_knn_mean(features)
        return jsonify({"prediction": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
