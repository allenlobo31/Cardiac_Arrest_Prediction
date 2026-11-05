"""
app.py — Flask API for the HRV cardiac-arrest-warning model.

Run locally:
    pip install -r requirements.txt
    python app.py
    # Next.js frontend: cd frontend && npm run dev

WHAT THIS APP DOES
-------------------
1. Accepts an image of an ECG waveform plot from the Next.js frontend
    (a clean digital chart/screenshot — not a photo of paper).
2. Digitizes that image into a 1D numeric signal (image_to_signal.py).
3. Converts the signal into the 90 HRV features the model was trained on
   (raw_to_hrv_features.py — this runs neurokit2's R-peak detector).
4. Scales the features and runs them through random_forest_prearrest.joblib.
5. Returns the prediction (Normal / Pre-arrest) plus a sanity-check plot
    with detected heartbeats marked as JSON.

IMPORTANT — read this before trusting a result:
Turning a static image back into a signal is inherently a best-effort
reconstruction, not a lab-grade digitization. The model also requires at
least ~150 detected beats (several minutes of recording at a resting heart
rate) to compute reliable HRV features — a short strip will be rejected.
See image_to_signal.py for the exact assumptions and how to get a cleaner
result (crop tightly to the waveform, use a high-contrast trace color).

The service exposes a JSON API for the Next.js frontend:
    GET  /health       -> confirms the model artifacts loaded correctly
    POST /api/predict_image -> accepts an ECG image and duration as multipart form data
    POST /api/predict  -> accepts {"sampling_rate", "r_peaks"} or
                          {"sampling_rate", "ecg_signal"} as JSON
"""

import base64
import io
import os

import matplotlib
matplotlib.use("Agg")  # no display backend needed on a server
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, request

from src.image_to_signal import image_to_ecg_signal
from src.raw_to_hrv_features import ecg_signal_to_features, r_peaks_to_features

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "train", "arrest_model.joblib")

MIN_SAMPLING_RATE = 50    # sanity bound — implausibly low for ECG/PPG
MAX_SAMPLING_RATE = 2000  # sanity bound — implausibly high for a wearable
MIN_DURATION_SECONDS = 5
MAX_DURATION_SECONDS = 3600  # 1 hour

# Model artifacts are loaded once at startup, not per-request — loading a
# pickle on every call would be slow and pointless since nothing in it changes.
_model = None
_threshold = None
_feature_columns = None
_threshold_model_name = None


def load_artifacts():
    """Loads the trained model, scaler, and threshold once at process startup.
    Raises immediately (rather than failing silently later) if any file is
    missing, so a misconfigured deployment fails fast and loudly."""
    global _model, _threshold, _feature_columns, _threshold_model_name
    import joblib

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Required model artifact not found at {MODEL_PATH}.")
    artifact = joblib.load(MODEL_PATH)
    _model = artifact["model"]
    _threshold = 1.0 - artifact["threshold"]
    _feature_columns = artifact["features"]
    _threshold_model_name = os.path.basename(MODEL_PATH)


def _make_sanity_plot(cleaned_signal, r_peak_samples, sampling_rate):
    """Renders a clean, Apple-aesthetic PNG (as base64 data URI) showing the
    digitized signal with detected R-peaks marked in cardiac crimson."""
    t = np.arange(len(cleaned_signal)) / sampling_rate
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Waveform trace in refined deep slate
    ax.plot(t, cleaned_signal, linewidth=1.1, color="#0f172a", alpha=0.9, label="Digitized waveform")

    # Detected R-peaks in vibrant cardiac crimson
    peak_times = np.asarray(r_peak_samples) / sampling_rate
    peak_values = np.asarray(cleaned_signal)[np.asarray(r_peak_samples, dtype=int)]
    ax.scatter(peak_times, peak_values, color="#e11d48", edgecolors="#ffffff", linewidths=0.5, s=24, zorder=4, label="Detected R-peaks")

    # Minimalist Apple-style spines and axis styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="x", colors="#64748b", labelsize=8.5, length=3, direction="out")
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", fontsize=8.5, color="#64748b", labelpad=6)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fancybox=True)
    fig.tight_layout(pad=0.8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _run_prediction_pipeline(ecg_signal, sampling_rate):
    """Shared core: signal + sampling_rate -> features -> scaled -> prediction.
    Returns a JSON-serializable dict, or raises ValueError with a
    human-readable message on bad/unusable input (too few beats, etc.)."""
    features, cleaned, r_peaks = ecg_signal_to_features(
        ecg_signal, sampling_rate, return_peaks=True
    )
    raw_proba = float(_model.predict_proba(features[_feature_columns])[0, 1])
    proba = 1.0 - raw_proba
    label = "Pre-arrest" if proba >= _threshold else "Normal"
    plot_uri = _make_sanity_plot(cleaned, r_peaks, sampling_rate)

    return {
        "label": label,
        "probability": round(proba, 4),
        "threshold_used": _threshold,
        "beats_detected": int(len(r_peaks)),
        "sampling_rate_used": round(sampling_rate, 2),
        "plot_uri": plot_uri,
    }


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept"
    return response


@app.route("/api/predict_image", methods=["POST", "OPTIONS"])
def api_predict_image():
    """Dedicated REST endpoint for React/Next.js: accepts multipart image + duration, returns JSON."""
    if request.method == "OPTIONS":
        return "", 204
    if _model is None:
        return jsonify({"error": "Model not loaded — check server startup logs."}), 503

    image_file = request.files.get("ecg_image")
    duration_raw = request.form.get("duration_seconds", "").strip()

    if image_file is None or image_file.filename == "":
        return jsonify({"error": "Please choose an image file to upload."}), 400

    try:
        duration_seconds = float(duration_raw)
    except ValueError:
        return jsonify({"error": "'Duration (seconds)' must be a valid number."}), 400

    if not (MIN_DURATION_SECONDS <= duration_seconds <= MAX_DURATION_SECONDS):
        return jsonify({
            "error": f"Duration looks implausible ({duration_seconds}s). Expected between {MIN_DURATION_SECONDS} and {MAX_DURATION_SECONDS}s."
        }), 400

    try:
        image_bytes = image_file.read()
        ecg_signal = image_to_ecg_signal(image_bytes)

        sampling_rate = len(ecg_signal) / duration_seconds
        if not (MIN_SAMPLING_RATE <= sampling_rate <= MAX_SAMPLING_RATE):
            return jsonify({
                "error": f"Implied sampling rate ({sampling_rate:.1f} Hz) is outside plausible range ({MIN_SAMPLING_RATE}-{MAX_SAMPLING_RATE} Hz). Double-check the duration entered."
            }), 400

        result = _run_prediction_pipeline(ecg_signal, sampling_rate)
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Unexpected error during image prediction")
        return jsonify({"error": "Something went wrong processing that image."}), 500


@app.route("/health", methods=["GET"])
def health():
    """Simple readiness check — confirms the model artifacts are loaded and
    the service is actually ready to serve predictions, not just that the
    process is running."""
    ok = _model is not None and _threshold is not None and _feature_columns is not None
    return jsonify({
        "status": "ok" if ok else "not_ready",
        "model_name": _threshold_model_name,
        "threshold": _threshold,
    }), (200 if ok else 503)


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def api_predict():
    """Original JSON API, unchanged — still available for programmatic use
    (e.g. sending raw R-peaks or an ECG signal directly, no image involved)."""
    if request.method == "OPTIONS":
        return "", 204
    if _model is None:
        return jsonify({"error": "Model not loaded — check server startup logs."}), 503

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    sampling_rate = body.get("sampling_rate")
    r_peaks = body.get("r_peaks")
    ecg_signal = body.get("ecg_signal")

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

    try:
        if r_peaks is not None:
            if not isinstance(r_peaks, list) or len(r_peaks) == 0:
                return jsonify({"error": "'r_peaks' must be a non-empty list of sample indices."}), 400
            r_peaks_arr = np.asarray(r_peaks, dtype=float)
            features = r_peaks_to_features(r_peaks_arr, sampling_rate)
            beats_detected = len(r_peaks_arr)
        else:
            if not isinstance(ecg_signal, list) or len(ecg_signal) == 0:
                return jsonify({"error": "'ecg_signal' must be a non-empty list of numeric samples."}), 400
            ecg_arr = np.asarray(ecg_signal, dtype=float)
            features = ecg_signal_to_features(ecg_arr, sampling_rate)
            beats_detected = None

        raw_proba = float(_model.predict_proba(features[_feature_columns])[0, 1])
        proba = 1.0 - raw_proba
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
        return jsonify({"error": str(e)}), 400
    except Exception:
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
