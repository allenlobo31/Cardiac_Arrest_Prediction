"""
app.py — Flask API for the HRV cardiac-arrest-warning model.

Accepts raw heartbeat data over HTTP, converts it to the same 90 HRV features
the model was trained on, and returns a Normal / Pre-arrest prediction.

Run locally:
    pip install flask joblib scikit-learn neurokit2 numpy pandas
    python app.py
    # server starts on http://0.0.0.0:5000

Endpoints:
    GET  /health          -> confirms the model artifacts loaded correctly
    POST /predict         -> runs a prediction on raw heartbeat data

Request body for /predict (send ONE of r_peaks or ecg_signal, not both):

    Option A — you already have R-peak sample indices:
    {
        "sampling_rate": 250,
        "r_peaks": [102, 312, 519, 731, ...]
    }

    Option B — you only have the raw ECG waveform:
    {
        "sampling_rate": 250,
        "ecg_signal": [0.01, 0.02, -0.03, ...]
    }

Response:
    {
        "label": "Normal" | "Pre-arrest",
        "probability": 0.24,
        "threshold_used": 0.493,
        "beats_detected": 380
    }

Error response (400):
    {"error": "explanation of what was wrong with the request"}
"""

import json
import os

import numpy as np
from flask import Flask, jsonify, request

from raw_to_hrv_features import ecg_signal_to_features, r_peaks_to_features

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "feature_scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "feature_columns.json")
THRESHOLD_PATH = os.path.join(BASE_DIR, "chosen_threshold.json")

MIN_SAMPLING_RATE = 50    # sanity bound — implausibly low for ECG/PPG
MAX_SAMPLING_RATE = 2000  # sanity bound — implausibly high for a wearable

# Model artifacts are loaded once at startup, not per-request — loading a
# pickle on every call would be slow and pointless since nothing in it changes.
_model = None
_scaler = None
_threshold = None
_threshold_model_name = None


def load_artifacts():
    """Loads the trained model, scaler, and threshold once at process startup.
    Raises immediately (rather than failing silently later) if any file is
    missing, so a misconfigured deployment fails fast and loudly."""
    global _model, _scaler, _threshold, _threshold_model_name
    import joblib

    for path, name in [(MODEL_PATH, "trained_model.pkl"),
                        (SCALER_PATH, "feature_scaler.pkl"),
                        (FEATURE_COLUMNS_PATH, "feature_columns.json"),
                        (THRESHOLD_PATH, "chosen_threshold.json")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required model artifact '{name}' not found at {path}. "
                f"Make sure it's in the same folder as app.py."
            )

    _model = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    with open(THRESHOLD_PATH) as f:
        threshold_info = json.load(f)
    _threshold = threshold_info["threshold"]
    _threshold_model_name = threshold_info.get("model_name", "unknown")


@app.route("/health", methods=["GET"])
def health():
    """Simple readiness check — confirms the model artifacts are loaded and
    the service is actually ready to serve predictions, not just that the
    process is running."""
    ok = _model is not None and _scaler is not None and _threshold is not None
    return jsonify({
        "status": "ok" if ok else "not_ready",
        "model_name": _threshold_model_name,
        "threshold": _threshold,
    }), (200 if ok else 503)


@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"error": "Model not loaded — check server startup logs."}), 503

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    sampling_rate = body.get("sampling_rate")
    r_peaks = body.get("r_peaks")
    ecg_signal = body.get("ecg_signal")

    # --- input validation ---
    if sampling_rate is None:
        return jsonify({"error": "'sampling_rate' is required."}), 400
    try:
        sampling_rate = float(sampling_rate)
    except (TypeError, ValueError):
        return jsonify({"error": "'sampling_rate' must be a number."}), 400
    if not (MIN_SAMPLING_RATE <= sampling_rate <= MAX_SAMPLING_RATE):
        return jsonify({
            "error": f"'sampling_rate' looks implausible ({sampling_rate}). "
                     f"Expected between {MIN_SAMPLING_RATE} and {MAX_SAMPLING_RATE} Hz."
        }), 400

    if (r_peaks is None) == (ecg_signal is None):
        return jsonify({"error": "Provide exactly one of 'r_peaks' or 'ecg_signal', not both/neither."}), 400

    # --- convert raw data -> HRV features -> prediction ---
    try:
        if r_peaks is not None:
            if not isinstance(r_peaks, list) or len(r_peaks) == 0:
                return jsonify({"error": "'r_peaks' must be a non-empty list of sample indices."}), 400
            r_peaks_arr = np.asarray(r_peaks, dtype=float)
            features = r_peaks_to_features(r_peaks_arr, sampling_rate, FEATURE_COLUMNS_PATH)
            beats_detected = len(r_peaks_arr)
        else:
            if not isinstance(ecg_signal, list) or len(ecg_signal) == 0:
                return jsonify({"error": "'ecg_signal' must be a non-empty list of numeric samples."}), 400
            ecg_arr = np.asarray(ecg_signal, dtype=float)
            features = ecg_signal_to_features(ecg_arr, sampling_rate, FEATURE_COLUMNS_PATH)
            beats_detected = None  # peak count only known internally to ecg_signal_to_features

        features_scaled = _scaler.transform(features)
        proba = float(_model.predict_proba(features_scaled)[0, 1])
        label = "Pre-arrest" if proba >= _threshold else "Normal"

        response = {
            "label": label,
            "probability": round(proba, 4),
            "threshold_used": _threshold,
        }
        if beats_detected is not None:
            response["beats_detected"] = beats_detected

        return jsonify(response), 200

    except ValueError as e:
        # raised by raw_to_hrv_features.py for things like "too few beats" —
        # these are the user's input being unusable, not a server bug
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # anything unexpected — log server-side detail, keep the client
        # response generic so internals aren't leaked
        app.logger.exception("Unexpected error during prediction")
        return jsonify({"error": "Internal error while processing the request."}), 500


if __name__ == "__main__":
    load_artifacts()
    print(f"Model loaded: {_threshold_model_name}, threshold={_threshold}")
    app.run(host="0.0.0.0", port=5000, debug=False)
else:
    # also load artifacts when imported by a WSGI server (gunicorn, etc.),
    # not just when run directly with `python app.py`
    load_artifacts()