"""
raw_to_hrv_features.py

Converts raw heartbeat data into the 7-feature HRV vector that
arrest_model.joblib expects. Two entry points depending on what raw data you have:

    - r_peaks_to_features()  : you already have R-peak locations
    - ecg_signal_to_features(): you only have a raw ECG waveform (peaks get
                                 detected for you)

Both return a single-row pandas DataFrame ready for model.predict_proba().
"""

import os
import numpy as np
import pandas as pd
import neurokit2 as nk

MIN_BEATS_REQUIRED = 150  # matches MIN_BEATS_PER_WINDOW used when the training data was built

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train", "arrest_model.joblib"
)
ARREST_FEATURES = ["mean_rr", "sdnn", "rmssd", "pnn50", "sd1", "sd2", "sd1_sd2"]


def r_peaks_to_features(r_peak_samples, sampling_rate, feature_columns_path=None):
    """
    r_peak_samples : array-like of R-peak locations, in SAMPLES (not seconds).
                      e.g. if your peaks are in seconds, multiply by sampling_rate first.
    sampling_rate  : the sampling rate (Hz) those peak locations were measured at.
    Returns a single-row pandas DataFrame with the 7 HRV feature columns used
    by train/train.py, in the correct order.
    """
    r_peak_samples = np.asarray(r_peak_samples)

    if len(r_peak_samples) < MIN_BEATS_REQUIRED:
        raise ValueError(
            f"Only {len(r_peak_samples)} beats given — need at least {MIN_BEATS_REQUIRED} "
            f"for reliable HRV computation (this matches the minimum used when the "
            f"training data was built). Use a longer recording window."
        )

    rr = np.diff(r_peak_samples) / sampling_rate
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) < MIN_BEATS_REQUIRED - 1:
        raise ValueError("Too few valid RR intervals for reliable HRV computation.")
    rr = rr[np.abs(rr / np.median(rr) - 1) < 0.3]
    differences = np.diff(rr)
    sd1 = np.std(differences) / np.sqrt(2)
    sd2 = np.sqrt(max(2 * np.var(rr) - sd1 ** 2, 0))
    values = [
        rr.mean(), rr.std(), np.sqrt(np.mean(differences ** 2)),
        np.mean(np.abs(differences) > 0.05), sd1, sd2, sd1 / (sd2 + 1e-9),
    ]
    return pd.DataFrame([values], columns=ARREST_FEATURES)


def ecg_signal_to_features(ecg_signal, sampling_rate, feature_columns_path=None,
                            return_peaks=False):
    """
    ecg_signal     : 1D array-like of raw ECG voltage samples.
    sampling_rate  : the sampling rate (Hz) of ecg_signal.
    return_peaks   : if True, also returns (cleaned_signal, r_peak_samples) so callers
                      (e.g. the web UI) can plot the detected beats for a sanity check.

    Detects R-peaks automatically, then returns the same single-row HRV
    feature DataFrame as r_peaks_to_features().
    """
    ecg_signal = np.asarray(ecg_signal)

    # neurokit2's standard ECG cleaning + R-peak detection pipeline
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    _, rpeaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    r_peak_samples = rpeaks_info["ECG_R_Peaks"]

    features = r_peaks_to_features(r_peak_samples, sampling_rate, feature_columns_path)
    if return_peaks:
        return features, ecg_cleaned, r_peak_samples
    return features


def predict_from_raw(r_peak_samples=None, ecg_signal=None, sampling_rate=None,
                      model_path=_DEFAULT_MODEL, scaler_path=None,
                      feature_columns_path=None, threshold_path=None):
    """Full end-to-end prediction using raw R-peaks or an ECG signal."""
    import joblib

    if sampling_rate is None:
        raise ValueError("sampling_rate is required.")
    if (r_peak_samples is None) == (ecg_signal is None):
        raise ValueError("Pass exactly one of r_peak_samples or ecg_signal, not both/neither.")

    if r_peak_samples is not None:
        features = r_peaks_to_features(r_peak_samples, sampling_rate, feature_columns_path)
    else:
        features = ecg_signal_to_features(ecg_signal, sampling_rate, feature_columns_path)

    artifact = joblib.load(model_path)
    model = artifact["model"]
    threshold = 1.0 - artifact["threshold"]
    raw_proba = model.predict_proba(features[artifact["features"]])[0, 1]
    proba = 1.0 - raw_proba
    label = "Pre-arrest" if proba >= threshold else "Normal"

    return {"label": label, "probability": float(proba), "threshold_used": threshold}


if __name__ == "__main__":
    # --- self-test with synthetic R-peaks, just to prove the pipeline runs end to end ---
    # simulates ~5 minutes at 72 bpm with slight natural variability, sampled at 250 Hz
    fs = 250
    rng = np.random.RandomState(0)
    n_beats = 380
    rr_intervals_sec = 0.83 + rng.normal(0, 0.03, n_beats)  # ~72bpm +/- natural variability
    peak_times_sec = np.cumsum(rr_intervals_sec)
    peak_samples = (peak_times_sec * fs).astype(int)

    feats = r_peaks_to_features(peak_samples, sampling_rate=fs)
    print("Extracted feature row shape:", feats.shape)
    print(feats.iloc[0][["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD"]])
