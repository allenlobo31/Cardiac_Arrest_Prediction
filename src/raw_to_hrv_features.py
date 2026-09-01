"""
raw_to_hrv_features.py

Converts raw heartbeat data into the exact 90-feature HRV vector that
trained_model.pkl expects, in the exact column order feature_columns.json
specifies. Two entry points depending on what raw data you have:

    - r_peaks_to_features()  : you already have R-peak locations
    - ecg_signal_to_features(): you only have a raw ECG waveform (peaks get
                                 detected for you)

Both return a single-row pandas DataFrame ready for scaler.transform() and
model.predict_proba().
"""

import json
import numpy as np
import pandas as pd
import neurokit2 as nk

MIN_BEATS_REQUIRED = 150  # matches MIN_BEATS_PER_WINDOW used when the training data was built


def r_peaks_to_features(r_peak_samples, sampling_rate, feature_columns_path="feature_columns.json"):
    """
    r_peak_samples : array-like of R-peak locations, in SAMPLES (not seconds).
                      e.g. if your peaks are in seconds, multiply by sampling_rate first.
    sampling_rate  : the sampling rate (Hz) those peak locations were measured at.
    feature_columns_path : path to the feature_columns.json saved alongside the trained model.

    Returns a single-row pandas DataFrame with the 90 HRV feature columns,
    in the correct order, ready for scaler.transform().
    """
    r_peak_samples = np.asarray(r_peak_samples)

    if len(r_peak_samples) < MIN_BEATS_REQUIRED:
        raise ValueError(
            f"Only {len(r_peak_samples)} beats given — need at least {MIN_BEATS_REQUIRED} "
            f"for reliable HRV computation (this matches the minimum used when the "
            f"training data was built). Use a longer recording window."
        )

    with open(feature_columns_path) as f:
        keep_cols = json.load(f)

    hrv_df = nk.hrv(r_peak_samples, sampling_rate=sampling_rate, show=False)

    missing = [c for c in keep_cols if c not in hrv_df.columns]
    if missing:
        raise ValueError(
            f"neurokit2 did not produce {len(missing)} of the expected features "
            f"(commonly happens with short/irregular windows): {missing}"
        )

    return hrv_df[keep_cols].reset_index(drop=True)


def ecg_signal_to_features(ecg_signal, sampling_rate, feature_columns_path="feature_columns.json",
                            return_peaks=False):
    """
    ecg_signal     : 1D array-like of raw ECG voltage samples.
    sampling_rate  : the sampling rate (Hz) of ecg_signal.
    feature_columns_path : path to the feature_columns.json saved alongside the trained model.
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
                      model_path="trained_model.pkl", scaler_path="feature_scaler.pkl",
                      feature_columns_path="feature_columns.json",
                      threshold_path="chosen_threshold.json"):
    """
    Full end-to-end: raw R-peaks OR raw ECG signal -> HRV features -> scaled ->
    model prediction. Pass exactly one of r_peak_samples / ecg_signal.
    """
    import joblib

    if sampling_rate is None:
        raise ValueError("sampling_rate is required.")
    if (r_peak_samples is None) == (ecg_signal is None):
        raise ValueError("Pass exactly one of r_peak_samples or ecg_signal, not both/neither.")

    if r_peak_samples is not None:
        features = r_peaks_to_features(r_peak_samples, sampling_rate, feature_columns_path)
    else:
        features = ecg_signal_to_features(ecg_signal, sampling_rate, feature_columns_path)

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

    features_scaled = scaler.transform(features)
    proba = model.predict_proba(features_scaled)[0, 1]
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

    feats = r_peaks_to_features(peak_samples, sampling_rate=fs,
                                 feature_columns_path="feature_columns.json")
    print("Extracted feature row shape:", feats.shape)
    print(feats.iloc[0][["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD"]])