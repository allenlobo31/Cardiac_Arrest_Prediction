# pip install wfdb neurokit2 pandas numpy  (run once in your terminal/venv)

import wfdb
import pandas as pd
import neurokit2 as nk
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sddb_records = wfdb.get_record_list('sddb')
print(f"SDDB records ({len(sddb_records)}): {sddb_records}")

NORMAL_BEAT_SYMBOLS = {'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q'}

def get_rpeak_times(record_name, pn_dir):
    """Streams beat annotations for a record directly from PhysioNet and
    returns R-peak sample indices, R-peak times (seconds), and sampling rate."""
    ann = wfdb.rdann(record_name, 'atr', pn_dir=pn_dir)
    fs = ann.fs
    is_beat = np.array([s in NORMAL_BEAT_SYMBOLS for s in ann.symbol])
    peak_samples = ann.sample[is_beat]
    peak_times = peak_samples / fs
    return peak_samples, peak_times, fs


WINDOW_SEC        = 5 * 60   # 5-minute analysis window (standard in HRV literature)
STEP_SEC          = 60       # slide window by 1 minute
LEAD_TIME_SEC     = 30 * 60  # window ending within 30 min of recording end -> pre-arrest
BASELINE_GAP_SEC  = 60 * 60  # window ending >60 min before recording end -> baseline/normal
MIN_BEATS_PER_WINDOW = 150   # skip windows with too few detected beats (unreliable HRV)

def extract_windows(peak_samples, peak_times, fs, record_id, mode):
    """
    mode='sddb'  -> windows near the recording end are labeled pre-arrest (1),
                    windows well before the end are labeled baseline (0),
                    windows in between are discarded (ambiguous buffer zone)
    mode='nsrdb' -> every window is labeled normal (0)
    """
    rows = []
    record_end = peak_times[-1]
    win_start = peak_times[0]

    while win_start + WINDOW_SEC <= peak_times[-1]:
        win_end = win_start + WINDOW_SEC
        mask = (peak_times >= win_start) & (peak_times < win_end)
        win_peaks = peak_samples[mask]

        if len(win_peaks) >= MIN_BEATS_PER_WINDOW:
            if mode == 'sddb':
                time_to_end = record_end - win_end
                if time_to_end <= LEAD_TIME_SEC:
                    label = 1
                elif time_to_end >= BASELINE_GAP_SEC:
                    label = 0
                else:
                    label = None
            else:
                label = 0

            if label is not None:
                try:
                    hrv = nk.hrv(win_peaks, sampling_rate=fs, show=False)
                    hrv['record_id'] = record_id
                    hrv['window_end_sec'] = win_end
                    hrv['label'] = label
                    rows.append(hrv)
                except Exception:
                    pass  # skip windows where HRV computation fails (too irregular/short)

        win_start += STEP_SEC

    return rows

sddb_feature_frames = []
print("Extracting SDDB windows...")
for rec in sddb_records:
    try:
        peak_samples, peak_times, fs = get_rpeak_times(rec, pn_dir='sddb/1.0.0')
        rows = extract_windows(peak_samples, peak_times, fs, record_id=f"sddb_{rec}", mode='sddb')
        sddb_feature_frames.extend(rows)
        print(f"sddb/{rec}: {len(rows)} labeled windows")
    except Exception as e:
        print(f"sddb/{rec}: skipped ({e})")

sddb_df = pd.concat(sddb_feature_frames, ignore_index=True) if sddb_feature_frames else pd.DataFrame()
print("SDDB windows:", sddb_df.shape)


sddb_df = pd.concat(sddb_feature_frames, ignore_index=True) if sddb_feature_frames else pd.DataFrame()
print("SDDB windows:", sddb_df.shape)

sddb_df.to_csv('sddb_features_cache.csv', index=False)