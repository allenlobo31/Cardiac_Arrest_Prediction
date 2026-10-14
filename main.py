"""Flask app for the NewModel pre-arrest predictor.

This is intentionally separate from app.py.  The new model was trained on
patient-relative window features, so this app keeps a short in-memory history
per patient while preserving the existing image and JSON endpoints.
"""

import base64
import io
import json
import os
from collections import defaultdict, deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from Version2.NewModel.pre_arrest_features import build_features
from src.image_to_signal import image_to_ecg_signal

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Version2", "NewModel", "pre_arrest_predictor.joblib")

BASE_FEATURES = [
    "n_beats", "mean_rr", "mean_hr", "sdnn", "rmssd", "sdsd", "pnn50",
    "cv_rr", "min_rr", "max_rr", "rr_range", "sd1", "sd2", "lf", "hf",
    "lf_hf", "n_ventricular", "pvc_burden", "max_consec_ventricular",
    "non_n_fraction", "premature_fraction",
]
MODEL_FEATURES = [
    "mean_rr_z", "mean_rr_zmean", "mean_rr_zstd", "sdnn_z", "sdnn_zmean",
    "sdnn_zstd", "rmssd_z", "rmssd_zmean", "rmssd_zstd", "pnn50_z",
    "pnn50_zmean", "pnn50_zstd", "min_rr_z", "min_rr_zmean", "min_rr_zstd",
    "max_rr_z", "max_rr_zmean", "max_rr_zstd", "cv_rr_z", "cv_rr_zmean",
    "cv_rr_zstd", "sd2_z", "sd2_zmean", "sd2_zstd", "premature_fraction_z",
    "premature_fraction_zmean", "premature_fraction_zstd",
]

MIN_SAMPLING_RATE = 50
MAX_SAMPLING_RATE = 2000
MIN_DURATION_SECONDS = 5
MAX_DURATION_SECONDS = 3600
DEFAULT_THRESHOLD = 0.5
HISTORY_LENGTH = 500

_model = None
_threshold = DEFAULT_THRESHOLD
_patient_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))


def load_artifacts():
    global _model, _threshold, MODEL_FEATURES
    import joblib

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"New model not found at {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("New model artifact must contain a 'pipeline'.")
    _model = artifact["pipeline"]
    MODEL_FEATURES = artifact.get("feature_columns", MODEL_FEATURES)
    _threshold = float(artifact.get("t_warning", DEFAULT_THRESHOLD))


def _make_sanity_plot(cleaned_signal, r_peak_samples, sampling_rate):
    t = np.arange(len(cleaned_signal)) / sampling_rate
    peaks = np.asarray(r_peak_samples, dtype=int)
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=110)
    ax.plot(t, cleaned_signal, linewidth=0.8, color="#2563eb")
    ax.scatter(peaks / sampling_rate, np.asarray(cleaned_signal)[peaks],
               color="#dc2626", s=14, zorder=3, label="detected beats")
    ax.set_xlabel("time (s)")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _detect_peaks(ecg_signal, sampling_rate):
    import neurokit2 as nk

    cleaned = nk.ecg_clean(np.asarray(ecg_signal), sampling_rate=sampling_rate)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
    return cleaned, np.asarray(info["ECG_R_Peaks"], dtype=int)


def _base_features(r_peak_samples, sampling_rate):
    peaks = np.asarray(r_peak_samples, dtype=float)
    rr = np.diff(peaks) / sampling_rate
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) < 5:
        raise ValueError("Too few valid RR intervals were detected for prediction.")

    differences = np.diff(rr)
    mean_rr = float(np.mean(rr))
    sdnn = float(np.std(rr, ddof=1))
    rmssd = float(np.sqrt(np.mean(differences ** 2)))
    sdsd = float(np.std(differences, ddof=1))
    median_rr = float(np.median(rr))
    premature_fraction = float(np.mean(rr < 0.8 * median_rr))
    sd2 = float(np.sqrt(max(2 * sdnn ** 2 - 0.5 * sdsd ** 2, 0.0)))

    values = {
        "n_beats": len(peaks), "mean_rr": mean_rr, "mean_hr": 60 / mean_rr,
        "sdnn": sdnn, "rmssd": rmssd, "sdsd": sdsd,
        "pnn50": float(np.mean(np.abs(differences) > 0.05)),
        "cv_rr": sdnn / mean_rr if mean_rr else 0.0,
        "min_rr": float(np.min(rr)), "max_rr": float(np.max(rr)),
        "rr_range": float(np.ptp(rr)), "sd1": sdsd / np.sqrt(2), "sd2": sd2,
        "lf": np.nan, "hf": np.nan, "lf_hf": np.nan,
        "n_ventricular": np.nan, "pvc_burden": np.nan,
        "max_consec_ventricular": np.nan, "non_n_fraction": np.nan,
        "premature_fraction": premature_fraction,
    }
    return {name: values[name] for name in BASE_FEATURES}


def _predict_base_row(base_row, patient_id="default", order=None, history=None):
    if history is None:
        history = _patient_history[patient_id]
    rows = list(history) + [base_row]
    if order is None:
        order = len(history)
    frame = pd.DataFrame(rows)
    frame["patient_id"] = patient_id
    frame["orig_row"] = np.arange(len(frame))
    frame.loc[frame.index[-1], "orig_row"] = order
    relative = build_features(frame, BASE_FEATURES, feature_set="relative")
    model_input = relative.loc[[relative.index[-1]], MODEL_FEATURES]
    probability = float(_model.predict_proba(model_input)[0, 1])
    history.append(base_row)
    return probability


def _result(probability, beats_detected, sampling_rate, plot_uri=None):
    result = {
        "label": "Pre-arrest" if probability >= _threshold else "Normal",
        "probability": round(probability, 4),
        "threshold_used": _threshold,
        "beats_detected": int(beats_detected),
        "sampling_rate_used": round(sampling_rate, 2),
    }
    if plot_uri:
        result["plot_uri"] = plot_uri
    return result


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict_from_image():
    if _model is None:
        return render_template("index.html", result=None, error="New model is not loaded.")
    image_file = request.files.get("ecg_image")
    duration_raw = request.form.get("duration_seconds", "").strip()
    patient_id = request.form.get("patient_id", "default").strip() or "default"
    if image_file is None or image_file.filename == "":
        return render_template("index.html", result=None, error="Please choose an image file to upload.")
    try:
        duration_seconds = float(duration_raw)
        if not MIN_DURATION_SECONDS <= duration_seconds <= MAX_DURATION_SECONDS:
            raise ValueError(f"Duration must be between {MIN_DURATION_SECONDS} and {MAX_DURATION_SECONDS} seconds.")
        ecg_signal = image_to_ecg_signal(image_file.read())
        sampling_rate = len(ecg_signal) / duration_seconds
        if not MIN_SAMPLING_RATE <= sampling_rate <= MAX_SAMPLING_RATE:
            raise ValueError(f"The implied sampling rate ({sampling_rate:.1f} Hz) is outside the plausible range.")
        cleaned, peaks = _detect_peaks(ecg_signal, sampling_rate)
        probability = _predict_base_row(_base_features(peaks, sampling_rate), patient_id)
        result = _result(probability, len(peaks), sampling_rate,
                         _make_sanity_plot(cleaned, peaks, sampling_rate))
        return render_template("index.html", result=result, error=None)
    except (TypeError, ValueError) as exc:
        return render_template("index.html", result=None, error=str(exc))
    except Exception:
        app.logger.exception("Unexpected error during image-based prediction")
        return render_template("index.html", result=None, error="Something went wrong processing that image.")


@app.route("/health", methods=["GET"])
def health():
    ready = _model is not None
    return jsonify({"status": "ok" if ready else "not_ready", "model_name": "NewModel"}), (200 if ready else 503)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if _model is None:
        return jsonify({"error": "New model is not loaded."}), 503
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400
    try:
        patient_id = str(body.get("patient_id", "default"))
        history = body.get("history")
        if history is not None:
            if not isinstance(history, list):
                return jsonify({"error": "'history' must be a list of feature rows."}), 400
            history = deque(history[-HISTORY_LENGTH:], maxlen=HISTORY_LENGTH)
        sampling_rate = float(body["sampling_rate"])
        if "r_peaks" in body:
            peaks = np.asarray(body["r_peaks"], dtype=float)
            base_row = _base_features(peaks, sampling_rate)
            beats = len(peaks)
        elif "ecg_signal" in body:
            cleaned, peaks = _detect_peaks(np.asarray(body["ecg_signal"], dtype=float), sampling_rate)
            base_row = _base_features(peaks, sampling_rate)
            beats = len(peaks)
        elif "features" in body:
            base_row = {name: body["features"].get(name, np.nan) for name in BASE_FEATURES}
            beats = base_row["n_beats"]
        else:
            return jsonify({"error": "Provide 'r_peaks', 'ecg_signal', or 'features'."}), 400
        probability = _predict_base_row(base_row, patient_id, history=history)
        return jsonify(_result(probability, beats, sampling_rate)), 200
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Unexpected error during API prediction")
        return jsonify({"error": "Internal error while processing the request."}), 500


if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=False)
else:
    load_artifacts()