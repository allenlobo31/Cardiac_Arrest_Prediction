"""
scripts/extract_music.py

Extracts HRV feature windows from the MUSIC database (Sudden Cardiac Death in Chronic
Heart Failure, 24 h Holter) and caches them in data/raw/music.csv, with the same column
layout as nsrdb_features_cache.csv / sddb_features_cache.csv.

Differences from extract_nsrdb.py (and why):
  * MUSIC has NO beat annotations, so R-peaks are detected from the raw ECG with NeuroKit2.
  * The label is per PATIENT (1 = died of sudden cardiac death during follow-up,
    0 = survivor with full follow-up), copied onto every window of that patient.
    This is a long-term risk question, NOT the "next 30 minutes" question that the
    SDDB labels answer. Do not mix MUSIC rows with SDDB/NSRDB rows under one model.
  * Resumable: patients already in music.csv are skipped, so you can stop and re-run.

Expected folder layout (download from PhysioNet, "music-sudden-cardiac-death"):
    <music-dir>/subject-info.csv
    <music-dir>/Holter_ECG/P0001.hea, P0001.dat, P0002.hea, P0002.dat, ...

Usage:
    python scripts/extract_music.py --music-dir /real/path/to/music-sudden-cardiac-death/1.0.1 --limit 5
    python scripts/extract_music.py --music-dir /real/path/to/music-sudden-cardiac-death/1.0.1
"""

import argparse
import io
import re
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "music.csv"
MANIFEST_FILE = RAW_DIR / "music_manifest.csv"
SCHEMA_REFERENCE = RAW_DIR / "sddb_features_cache.csv"   # music.csv columns are locked to this header

WINDOW_SEC = 5 * 60            # same 5-minute window as the NSRDB / SDDB scripts
MIN_BEATS_PER_WINDOW = 150     # same minimum as the other scripts
MAX_WINDOWS_PER_PATIENT = 48   # evenly spaced across the 24 h, so no patient dominates
MIN_VALID_RR_FRACTION = 0.9    # window is dropped if <90% of RR intervals are 0.3-2.0 s (detector failure)
CHUNK_SEC = 30 * 60            # process the 24 h signal in 30-min chunks to bound memory

# Column names / Excel letters in subject-info.csv (see subject-info_codes.csv on PhysioNet).
# If a name isn't found, the Excel letter is used as a fallback.
COLS = {
    "exit":   ("Exit of the study", "D"),   # 0 = alive at end of follow-up
    "cause":  ("Cause of death", "E"),      # 0 = none, 3 = sudden cardiac death
    "rhythm": ("Holter rhythm", "BT"),      # 0 = sinus rhythm
}
SCD_CAUSE = 3


def assign_label(cause, exit_):
    """MUSIC codes (subject-info_codes.csv):
         Exit of the study : 0 survivor, 1 lost to follow-up, 2 cardiac transplantation, 3 death
         Cause of death    : 0 survivor, 1 non-cardiac death, 3 SCD, 6-7 pump-failure death
    Returns 1 (SCD), 0 (survivor) or None (excluded: other deaths, transplant, lost to follow-up).

    In the real CSV, survivors have 'Cause of death' = 0 and a BLANK 'Exit of the study' (verified on the
    downloaded file: 695 blanks = the 695 survivors), so blank exit counts as survivor. Transplants and
    lost-to-follow-up also have cause = 0, which is why exit must be blank/0 (not 1 or 2)."""
    if cause == SCD_CAUSE:
        return 1
    if cause == 0 and (pd.isna(exit_) or exit_ == 0):
        return 0
    if pd.isna(cause) and exit_ == 0:
        return 0
    return None


# ------------------------------------------------------------------ subject-info helpers
def letter_to_idx(letters):
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def find_col(df, name, letter):
    by_name = {re.sub(r"\s+", " ", str(c)).strip().lower(): c for c in df.columns}
    if name.lower() in by_name:
        return by_name[name.lower()]
    idx = letter_to_idx(letter)
    if idx < len(df.columns):
        return df.columns[idx]
    raise ValueError(f"Could not find column '{name}' (or Excel column {letter}) in subject-info.csv")


def make_unique(columns):
    """Strips whitespace and de-duplicates column names ('X', 'X', ' X' -> 'X', 'X.1', 'X.2')."""
    seen, out = {}, []
    for c in columns:
        c = re.sub(r"\s+", " ", str(c)).strip()
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def read_table(path):
    """Reads subject-info.csv robustly. The delimiter is chosen by counting candidates in the
    header line (csv.Sniffer's guess is unreliable on this file), and quoted fields are respected."""
    last_err = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = Path(path).read_text(encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        header = text.splitlines()[0] if text.strip() else ""
        seps = [s for s in sorted([",", ";", "\t", "|"], key=header.count, reverse=True) if header.count(s)]
        for sep in seps:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            except Exception as e:
                last_err = e
                continue
            if df.shape[1] > 5:
                df.columns = make_unique(df.columns)
                return df
        # last resort: skip malformed rows, but say so
        for sep in seps:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="warn")
                if df.shape[1] > 5:
                    print("WARNING: some malformed rows in subject-info.csv were skipped.")
                    df.columns = make_unique(df.columns)
                    return df
            except Exception as e:
                last_err = e
    raise ValueError(f"Could not parse {path} ({last_err}). First 300 characters:\n{text[:300]!r}")


def read_subject_info(music_dir):
    path = music_dir / "subject-info.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - download subject-info.csv from PhysioNet too.")
    return read_table(path)


def find_id_column(df, record_ids):
    """Finds the patient-ID column: the column that maps onto the MOST DISTINCT known record names.

    (Scoring by "fraction of rows that match" is wrong: a column like 'NYHA class' holding only the
    values 2 and 3 maps 100% of its rows onto P0002/P0003 and would beat the real ID column.)"""
    def to_rid(v):
        try:
            return f"P{int(float(v)):04d}"
        except (TypeError, ValueError):
            return str(v).strip().upper()

    record_ids = set(record_ids)
    name_hint = re.compile(r"patient|subject|record|\bid\b|code|number|^n[ºo°]?$", re.I)
    best, best_key = None, (0, 0)
    for i, c in enumerate(df.columns):
        col = df.iloc[:, i]                      # by position: safe even if names repeat
        mapped = col.map(to_rid)
        distinct = mapped[mapped.isin(record_ids)].nunique()
        key = (distinct, 1 if name_hint.search(str(c)) else 0)
        if key > best_key:
            best, best_key = c, key
    needed = 0.5 * min(len(record_ids), len(df))
    if best is None or best_key[0] < needed:
        raise ValueError(
            f"No column in subject-info.csv matches the Holter record names "
            f"(best column matched only {best_key[0]} distinct IDs). First columns: {list(df.columns)[:8]}"
        )
    return best, to_rid


# ------------------------------------------------------------------ signal -> R-peaks
def detect_rpeaks(record_path):
    """Detects R-peaks across a whole 24 h record, chunk by chunk. Returns (peak_samples, fs)."""
    hdr = wfdb.rdheader(str(record_path))
    fs, n_total = float(hdr.fs), int(hdr.sig_len)
    step, pad = int(CHUNK_SEC * fs), int(5 * fs)

    all_peaks = []
    for start in range(0, n_total, step):
        end = min(start + step + pad, n_total)
        sig = wfdb.rdrecord(str(record_path), sampfrom=start, sampto=end, channels=[0]).p_signal[:, 0]
        sig = pd.Series(sig).interpolate(limit_direction="both").fillna(0).to_numpy()
        if np.ptp(sig) == 0:
            continue  # flat-line chunk
        cleaned = nk.ecg_clean(sig, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=int) + start
        # drop the padded overlap so beats aren't counted twice
        all_peaks.append(peaks if end >= n_total else peaks[peaks < start + step])

    if not all_peaks:
        raise ValueError("no usable signal in this record")
    return np.concatenate(all_peaks), fs


# ------------------------------------------------------------------ R-peaks -> HRV windows
def extract_windows(peak_samples, fs, record_id, label):
    """Non-overlapping 5-min windows -> quality check -> evenly subsample -> HRV features."""
    peak_times = peak_samples / fs
    candidates = []
    for win_start in np.arange(peak_times[0], peak_times[-1] - WINDOW_SEC, WINDOW_SEC):
        mask = (peak_times >= win_start) & (peak_times < win_start + WINDOW_SEC)
        win_peaks = peak_samples[mask]
        if len(win_peaks) < MIN_BEATS_PER_WINDOW:
            continue
        rr = np.diff(win_peaks) / fs
        if np.mean((rr >= 0.3) & (rr <= 2.0)) < MIN_VALID_RR_FRACTION:
            continue
        candidates.append((win_start + WINDOW_SEC, win_peaks))

    if not candidates:
        return []

    keep = np.unique(np.linspace(0, len(candidates) - 1,
                                 min(MAX_WINDOWS_PER_PATIENT, len(candidates))).astype(int))
    rows = []
    for i in keep:
        win_end, win_peaks = candidates[i]
        try:
            hrv = nk.hrv(win_peaks, sampling_rate=fs, show=False)
        except Exception:
            continue  # skip windows where HRV computation fails
        hrv["record_id"] = f"music_{record_id}"
        hrv["window_end_sec"] = win_end
        hrv["label"] = int(label)
        rows.append(hrv)
    return rows


# ------------------------------------------------------------------ saving
def append_to_cache(df):
    """Appends rows to music.csv with columns locked to the SDDB cache header, so the files line up."""
    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        cols = list(pd.read_csv(OUTPUT_FILE, nrows=0).columns)
    elif SCHEMA_REFERENCE.exists():
        cols = list(pd.read_csv(SCHEMA_REFERENCE, nrows=0).columns)
    else:
        cols = list(df.columns)
    write_header = not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0
    df.reindex(columns=cols).to_csv(OUTPUT_FILE, mode="a", header=write_header, index=False)


def run_extraction(music_dir, limit=0, include_nonsinus=False):
    music_dir = Path(music_dir).expanduser()
    holter_dir = music_dir / "Holter_ECG"
    if not holter_dir.is_dir():
        raise FileNotFoundError(
            f"{holter_dir} not found. --music-dir must be the folder that CONTAINS "
            f"Holter_ECG/ and subject-info.csv (use a real path, not a placeholder)."
        )

    available = sorted(p.stem for p in holter_dir.glob("P*.hea") if (holter_dir / f"{p.stem}.dat").exists())
    if not available:
        raise FileNotFoundError(f"No complete PXXXX.hea + PXXXX.dat pairs found in {holter_dir}")
    print(f"Found {len(available)} complete Holter records.")

    info = read_subject_info(music_dir)
    id_col, to_rid = find_id_column(info, set(available))
    exit_col = find_col(info, *COLS["exit"])
    cause_col = find_col(info, *COLS["cause"])
    rhythm_col = find_col(info, *COLS["rhythm"])

    info["_rid"] = info[id_col].map(to_rid)
    info = info.drop_duplicates("_rid").set_index("_rid")
    exit_ = pd.to_numeric(info[exit_col], errors="coerce")
    cause = pd.to_numeric(info[cause_col], errors="coerce")
    rhythm = pd.to_numeric(info[rhythm_col], errors="coerce")
    print(f"Columns used -> id: '{id_col}', exit: '{exit_col}', cause: '{cause_col}', rhythm: '{rhythm_col}'")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        done = set(pd.read_csv(OUTPUT_FILE, usecols=["record_id"])["record_id"])

    manifest = []
    todo = available[:limit] if limit else available
    for rid in todo:
        if f"music_{rid}" in done:
            continue
        if rid not in info.index:
            manifest.append((rid, "excluded", "not in subject-info.csv", 0))
            continue

        # Label: sudden cardiac death vs clean survivor. Other deaths, transplants and
        # lost-to-follow-up are excluded rather than guessed.
        label = assign_label(cause[rid], exit_[rid])
        if label is None:
            manifest.append((rid, "excluded", f"cause={cause[rid]} exit={exit_[rid]}", 0))
            print(f"{rid}: excluded (cause={cause[rid]}, exit={exit_[rid]})")
            continue

        if not include_nonsinus and rhythm[rid] != 0:
            manifest.append((rid, "excluded", f"Holter rhythm={rhythm[rid]} (not sinus)", 0))
            print(f"{rid}: excluded (non-sinus rhythm={rhythm[rid]})")
            continue

        try:
            peaks, fs = detect_rpeaks(holter_dir / rid)
            rows = extract_windows(peaks, fs, rid, label)
            if not rows:
                manifest.append((rid, "failed", "no windows passed quality checks", 0))
                print(f"{rid}: no windows passed quality checks")
                continue
            df = pd.concat(rows, ignore_index=True)
            append_to_cache(df)
            manifest.append((rid, "included", f"label={label}", len(df)))
            print(f"{rid}: label={label}, {len(df)} windows saved")
        except Exception as e:
            manifest.append((rid, "failed", f"{type(e).__name__}: {e}", 0))
            print(f"{rid}: FAILED ({e})")

    if manifest:
        m = pd.DataFrame(manifest, columns=["record", "status", "detail", "n_windows"])
        m.to_csv(MANIFEST_FILE, mode="a", header=not MANIFEST_FILE.exists(), index=False)
        print("\nSummary:\n" + m["status"].value_counts().to_string())

    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        out = pd.read_csv(OUTPUT_FILE, usecols=["record_id", "label"])
        per_patient = out.groupby("record_id")["label"].first()
        print(f"\n{OUTPUT_FILE.name}: {len(out)} windows from {len(per_patient)} patients "
              f"({int(per_patient.sum())} SCD, {int((per_patient == 0).sum())} survivors)")
    else:
        print("\nNothing was written - see the exclusion/failure messages above.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--music-dir", required=True,
                    help="folder containing subject-info.csv and Holter_ECG/")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N records (trial run)")
    ap.add_argument("--include-nonsinus", action="store_true",
                    help="also keep patients whose Holter rhythm is not sinus (HRV less comparable)")
    args = ap.parse_args()
    run_extraction(args.music_dir, args.limit, args.include_nonsinus)