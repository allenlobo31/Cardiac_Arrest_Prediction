"""
scripts/extract_nsrdb.py

Extracts HRV feature windows from Normal Sinus Rhythm Database (NSRDB) on PhysioNet
and caches the results in data/raw/nsrdb_features_cache.csv.
"""

from pathlib import Path
import warnings
import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "nsrdb_features_cache.csv"

NORMAL_BEAT_SYMBOLS = {'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q'}
WINDOW_SEC = 5 * 60          # 5-minute analysis window (standard in HRV literature)
STEP_SEC = 60              # slide window by 1 minute
MIN_BEATS_PER_WINDOW = 150  # skip windows with too few detected beats (unreliable HRV)
NSRDB_SUBSAMPLE_STEP = 30  # keep every 30th window (~30 min apart) so long healthy recordings don't swamp the dataset


def get_rpeak_times(record_name, pn_dir):
    """Streams beat annotations for a record directly from PhysioNet and
    returns R-peak sample indices, R-peak times (seconds), and sampling rate."""
    ann = wfdb.rdann(record_name, 'atr', pn_dir=pn_dir)
    fs = ann.fs
    is_beat = np.array([s in NORMAL_BEAT_SYMBOLS for s in ann.symbol])
    peak_samples = ann.sample[is_beat]
    peak_times = peak_samples / fs
    return peak_samples, peak_times, fs


def extract_windows(peak_samples, peak_times, fs, record_id):
    """Every NSRDB window is labeled normal (0)."""
    rows = []
    win_start = peak_times[0]

    while win_start + WINDOW_SEC <= peak_times[-1]:
        win_end = win_start + WINDOW_SEC
        mask = (peak_times >= win_start) & (peak_times < win_end)
        win_peaks = peak_samples[mask]

        if len(win_peaks) >= MIN_BEATS_PER_WINDOW:
            try:
                hrv = nk.hrv(win_peaks, sampling_rate=fs, show=False)
                hrv['record_id'] = record_id
                hrv['window_end_sec'] = win_end
                hrv['label'] = 0
                rows.append(hrv)
            except Exception:
                pass  # skip windows where HRV computation fails (too irregular/short)

        win_start += STEP_SEC
    return rows


def run_extraction():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    nsrdb_records = wfdb.get_record_list('nsrdb')
    print(f"NSRDB records ({len(nsrdb_records)}): {nsrdb_records}")

    nsrdb_feature_frames = []
    print("Extracting NSRDB windows...")
    for rec in nsrdb_records:
        try:
            peak_samples, peak_times, fs = get_rpeak_times(rec, pn_dir='nsrdb/1.0.0')
            rows = extract_windows(peak_samples, peak_times, fs, record_id=f"nsrdb_{rec}")
            rows = rows[::NSRDB_SUBSAMPLE_STEP]
            nsrdb_feature_frames.extend(rows)
            print(f"nsrdb/{rec}: {len(rows)} labeled windows kept")
        except Exception as e:
            print(f"nsrdb/{rec}: skipped ({e})")

    nsrdb_df = pd.concat(nsrdb_feature_frames, ignore_index=True) if nsrdb_feature_frames else pd.DataFrame()
    print("NSRDB windows:", nsrdb_df.shape)
    nsrdb_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved NSRDB cache to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_extraction()