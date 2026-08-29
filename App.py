"""
app.py — Flask web app for the HRV cardiac-arrest-warning model.

Run locally:
    pip install -r requirements.txt
    python app.py
    # then open http://127.0.0.1:5000 in your browser

WHAT THIS APP DOES
-------------------
1. Shows a web page with a form to upload an image of an ECG waveform plot
   (a clean digital chart/screenshot — not a photo of paper).
2. Digitizes that image into a 1D numeric signal (image_to_signal.py).
3. Converts the signal into the 90 HRV features the model was trained on
   (raw_to_hrv_features.py — this runs neurokit2's R-peak detector).
4. Scales the features and runs them through trained_model.pkl.
5. Shows the prediction (Normal / Pre-arrest) plus a sanity-check plot of
   the digitized signal with detected heartbeats marked, so you can see
   whether the digitization actually worked before trusting the result.

IMPORTANT — read this before trusting a result:
Turning a static image back into a signal is inherently a best-effort
reconstruction, not a lab-grade digitization. The model also requires at
least ~150 detected beats (several minutes of recording at a resting heart
rate) to compute reliable HRV features — a short strip will be rejected.
See image_to_signal.py for the exact assumptions and how to get a cleaner
result (crop tightly to the waveform, use a high-contrast trace color).

This app also still exposes the original JSON API for programmatic use:
    GET  /health       -> confirms the model artifacts loaded correctly
    POST /api/predict  -> same as before: send {"sampling_rate", "r_peaks"}
                           or {"sampling_rate", "ecg_signal"} as JSON
"""

import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")  # no display backend needed on a server
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request

from image_to_signal import image_to_ecg_signal
from raw_to_hrv_features import ecg_signal_to_features, r_peaks_to_features

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "feature_scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "feature_columns.json")
THRESHOLD_PATH = os.path.join(BASE_DIR, "chosen_threshold.json")

MIN_SAMPLING_RATE = 50    # sanity bound — implausibly low for ECG/PPG
MAX_SAMPLING_RATE = 2000  # sanity bound — implausibly high for a wearable
MIN_DURATION_SECONDS = 5
MAX_DURATION_SECONDS = 3600  # 1 hour

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


def _make_sanity_plot(cleaned_signal, r_peak_samples, sampling_rate):
    """Renders a small PNG (as a base64 data URI) showing the digitized
    signal with detected R-peaks marked, so the user can visually confirm
    the digitization worked before trusting the prediction."""
    t = np.arange(len(cleaned_signal)) / sampling_rate
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=110)
    ax.plot(t, cleaned_signal, linewidth=0.8, color="#2563eb")
    peak_times = np.asarray(r_peak_samples) / sampling_rate
    peak_values = np.asarray(cleaned_signal)[np.asarray(r_peak_samples, dtype=int)]
    ax.scatter(peak_times, peak_values, color="#dc2626", s=14, zorder=3, label="detected beats")
    ax.set_xlabel("time (s)")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _run_prediction_pipeline(ecg_signal, sampling_rate):
    """Shared core: signal + sampling_rate -> features -> scaled -> prediction.
    Returns a dict ready to hand to the template, or raises ValueError with a
    human-readable message on bad/unusable input (too few beats, etc.)."""
    features, cleaned, r_peaks = ecg_signal_to_features(
        ecg_signal, sampling_rate, FEATURE_COLUMNS_PATH, return_peaks=True
    )
    features_scaled = _scaler.transform(features)
    proba = float(_model.predict_proba(features_scaled)[0, 1])
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


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict_from_image():
    """Web-form endpoint: image upload + duration -> digitize -> predict."""
    if _model is None:
        return render_template("index.html", result=None,
                                error="Model not loaded — check server startup logs.")

    image_file = request.files.get("ecg_image")
    duration_raw = request.form.get("duration_seconds", "").strip()

    if image_file is None or image_file.filename == "":
        return render_template("index.html", result=None,
                                error="Please choose an image file to upload.")

    try:
        duration_seconds = float(duration_raw)
    except ValueError:
        return render_template("index.html", result=None,
                                error="'Duration (seconds)' must be a number — "
                                      "enter how many seconds of ECG the image spans.")

    if not (MIN_DURATION_SECONDS <= duration_seconds <= MAX_DURATION_SECONDS):
        return render_template("index.html", result=None,
                                error=f"Duration looks implausible ({duration_seconds}s). "
                                      f"Expected between {MIN_DURATION_SECONDS} and "
                                      f"{MAX_DURATION_SECONDS} seconds.")

    try:
        image_bytes = image_file.read()
        ecg_signal = image_to_ecg_signal(image_bytes)

        sampling_rate = len(ecg_signal) / duration_seconds
        if not (MIN_SAMPLING_RATE <= sampling_rate <= MAX_SAMPLING_RATE):
            return render_template(
                "index.html", result=None,
                error=(f"The implied sampling rate ({sampling_rate:.1f} Hz, from "
                       f"{len(ecg_signal)} pixel columns over {duration_seconds}s) is "
                       f"outside a plausible range ({MIN_SAMPLING_RATE}-{MAX_SAMPLING_RATE} Hz). "
                       f"Double-check the duration you entered, or use a wider/narrower image.")
            )

        result = _run_prediction_pipeline(ecg_signal, sampling_rate)
        return render_template("index.html", result=result, error=None)

    except ValueError as e:
        # raised by image_to_signal.py or raw_to_hrv_features.py for things
        # like "no trace detected" or "too few beats" — bad input, not a bug
        return render_template("index.html", result=None, error=str(e))
    except Exception:
        app.logger.exception("Unexpected error during image-based prediction")
        return render_template("index.html", result=None,
                                error="Something went wrong processing that image. "
                                      "Check the server logs for details.")


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


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Original JSON API, unchanged — still available for programmatic use
    (e.g. sending raw R-peaks or an ECG signal directly, no image involved)."""
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
            features = r_peaks_to_features(r_peaks_arr, sampling_rate, FEATURE_COLUMNS_PATH)
            beats_detected = len(r_peaks_arr)
        else:
            if not isinstance(ecg_signal, list) or len(ecg_signal) == 0:
                return jsonify({"error": "'ecg_signal' must be a non-empty list of numeric samples."}), 400
            ecg_arr = np.asarray(ecg_signal, dtype=float)
            features = ecg_signal_to_features(ecg_arr, sampling_rate, FEATURE_COLUMNS_PATH)
            beats_detected = None

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