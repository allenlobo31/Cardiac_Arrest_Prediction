#!/usr/bin/env python3
"""
preprocess_cardiac_arrest_prediction.py
=======================================
PREDICTION mode: build a dataset that answers
    "will a life-threatening ventricular arrhythmia (VF/VFL/VT) start within
     the next <horizon> minutes?"
from the three long annotated databases only: MITDB, VFDB, SDDB.

STANDALONE: only needs  pip install wfdb numpy scipy pandas

Dataset-specific handling
    SDDB : VF onset taken from header comment 'vfon: HH:MM:SS'; VF assumed to
           last until the end of the record
    VFDB : rhythm labels only -> R-peaks detected from the signal (xqrs);
           (NOISE segments are excluded from both classes)

Per record
    1. rhythm annotations -> per-sample risk class; class 1-3 runs >= 2 s are
       "events" (nearby runs are merged into one episode)
    2. cut long windows (default 60 s, stride 30 s)
    3. label each window by the time between its END and the next event onset:

        time-to-onset   0 .. min_lead        -> EXCLUDED (too close, not useful)
        min_lead .. horizon                  -> label 1  (pre-arrest)
        horizon .. horizon + neg_buffer      -> EXCLUDED (grey zone)
        > horizon + neg_buffer / no event    -> label 0
        window contains VF/VT, or starts within post_buffer after an
        episode (recovery / unstable rhythm) -> EXCLUDED

    4. per window: HRV + ectopy features from the beat annotations, and
       (optionally) the band-passed, z-scored raw ECG at 250 Hz

Outputs (in --out)
    prediction_features.csv   one row per window: metadata, label, features
    prediction_dataset.npz    X (raw ECG, unless --no-signal), y, features,
                              feature_names

Usage
    python preprocess_cardiac_arrest_prediction.py --root ./data --out ./processed_pred
"""
import argparse
import re
import zlib
from fractions import Fraction
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, lombscargle, resample_poly, sosfiltfilt



# ======================= shared helpers =======================
TARGET_FS = 250
BAND = (0.5, 40.0)
SEED = 42

LEAD_PREFS = ["mlii", "ii", "ecg1", "ecg", "i"]

# rhythm token (annotation aux_note without '(') -> risk class
RHYTHM_TO_CLASS = {
    "VF": 1, "VFL": 1,
    "VTTDP": 2,
    "VT": 3, "VTHR": 3,
    "VTLR": 4, "B": 4, "T": 4, "HGEA": 4, "VER": 4, "IVR": 4,
    "AFIB": 5, "AFL": 5, "SVTA": 5, "SBR": 5, "BI": 5, "BII": 5,
    "NOD": 5, "AB": 5, "PREX": 5,
    "NOISE": 7, "ASYS": 7,          # noise / asystole: windows excluded
    "VFIB": 1, "PM": 6,
    "N": 6, "NSR": 6, "NE": 6, "BBB": 6, "P": 6,
}
MITDB_SAME_PATIENT = {"202": "201"}   # records 201/202 are the same patient


# ---------------------------------------------------------------- signal --
def pick_lead(names):
    low = [n.lower().strip() for n in names]
    for pref in LEAD_PREFS:
        if pref in low:
            return low.index(pref)
    return 0


_SOS = {}


def bandpass(x, fs):
    key = round(fs, 3)
    if key not in _SOS:
        hi = min(BAND[1], 0.45 * fs)
        _SOS[key] = butter(4, [BAND[0], hi], btype="band", fs=fs, output="sos")
    sos = _SOS[key]
    if len(x) < 100:
        return x
    return sosfiltfilt(sos, x)


def resample(x, fs_in, fs_out=TARGET_FS):
    if abs(fs_in - fs_out) < 1e-6:
        return x
    f = Fraction(fs_out, int(round(fs_in))).limit_denominator(1000)
    return resample_poly(x, f.numerator, f.denominator)


def clean(x, fs):
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0,
                      posinf=0.0, neginf=0.0)
    return resample(bandpass(x, fs), fs).astype(np.float32)


def zscore(w):
    sd = w.std()
    if not np.isfinite(sd) or sd < 1e-6:
        return None
    return ((w - w.mean()) / sd).astype(np.float32)


# ------------------------------------------------------------ annotations --
def read_annotation(base):
    for ext in ("atr", "ari", "ecg"):
        try:
            return wfdb.rdann(str(base), ext)
        except Exception:
            continue
    return None


def rhythm_class_array(ann, n_orig, n_new):
    """Per-sample risk-class array (int8) at the new sampling rate."""
    cls = np.full(n_new, 6, dtype=np.int8)
    ratio = n_new / float(n_orig)
    sym = np.array(ann.symbol)
    aux = ann.aux_note if ann.aux_note is not None else [""] * len(sym)
    samp = np.asarray(ann.sample)

    rhythm_idx = [i for i, s in enumerate(sym) if s == "+"]
    for k, i in enumerate(rhythm_idx):
        tok = aux[i].replace("\x00", "").strip().lstrip("(").upper()
        c = RHYTHM_TO_CLASS.get(tok)
        if c is None:
            UNKNOWN_TOKENS.add(tok)
            c = 6
        a = int(round(samp[i] * ratio))
        end_o = samp[rhythm_idx[k + 1]] if k + 1 < len(rhythm_idx) else n_orig
        b = int(round(end_o * ratio))
        cls[a:b] = c

    # '[' ... ']' = start/end of ventricular flutter/fibrillation
    start = None
    for i, s in enumerate(sym):
        if s == "[":
            start = int(round(samp[i] * ratio))
        elif s == "]" and start is not None:
            cls[start:int(round(samp[i] * ratio))] = 1
            start = None
    return cls


UNKNOWN_TOKENS = set()


def split_of(group):
    h = zlib.crc32(group.encode()) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def find_records(root):
    return sorted(p.with_suffix("") for p in Path(root).rglob("*.hea"))



BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ")
VENT_SYMBOLS = set("VEFr")          # ventricular ectopic / fusion beats
FEATURE_NAMES = [
    "n_beats", "mean_rr", "mean_hr", "sdnn", "rmssd", "sdsd", "pnn50", "cv_rr",
    "min_rr", "max_rr", "rr_range", "sd1", "sd2", "lf", "hf", "lf_hf",
    "n_ventricular", "pvc_burden", "max_consec_ventricular", "non_n_fraction",
    "premature_fraction",
]


# --------------------------------------------------------------- beats ----
def read_beats(base, fs):
    """Beat times (s) and symbols from the first annotation file that has
    >=100 beat annotations. Returns (times, symbols, ext) or None."""
    for ext in ("atr", "ari", "ecg", "qrs"):
        try:
            ann = wfdb.rdann(str(base), ext)
        except Exception:
            continue
        sym = np.array(ann.symbol)
        m = np.isin(sym, list(BEAT_SYMBOLS))
        if m.sum() >= 100:
            return np.asarray(ann.sample)[m] / fs, sym[m], ext
    return None


def max_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def hrv_features(bt, bs, ws, we, min_beats):
    """HRV / ectopy features for beats inside [ws, we) seconds."""
    i0, i1 = np.searchsorted(bt, ws), np.searchsorted(bt, we)
    t, sym = bt[i0:i1], bs[i0:i1]
    if len(t) < min_beats:
        return None
    rr = np.diff(t)
    ok = (rr > 0.3) & (rr < 2.0)               # drop gaps / missed beats
    rr_ok = rr[ok]
    if len(rr_ok) < min_beats - 1:
        return None
    dr = np.diff(rr)[ok[:-1] & ok[1:]]         # successive differences
    if len(dr) < 5:
        return None

    mean_rr = rr_ok.mean()
    sdnn = rr_ok.std(ddof=1)
    sdsd = dr.std(ddof=1)
    rmssd = np.sqrt(np.mean(dr ** 2))
    sd1 = sdsd / np.sqrt(2)
    sd2 = np.sqrt(max(2 * sdnn ** 2 - 0.5 * sdsd ** 2, 0.0))

    # Lomb-Scargle band powers (only rough on a 60 s window)
    tt = t[1:][ok]
    y = rr_ok - mean_rr
    def band(lo, hi):
        w = 2 * np.pi * np.linspace(lo, hi, 40)
        p = lombscargle(tt, y, w, normalize=False)
        f_hz = w / (2 * np.pi)
        return float(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(f_hz)))
    lf, hf = band(0.04, 0.15), band(0.15, 0.40)

    vent = np.isin(sym, list(VENT_SYMBOLS))
    return dict(
        n_beats=len(t), mean_rr=mean_rr, mean_hr=60.0 / mean_rr, sdnn=sdnn,
        rmssd=rmssd, sdsd=sdsd, pnn50=float(np.mean(np.abs(dr) > 0.05)),
        cv_rr=sdnn / mean_rr, min_rr=rr_ok.min(), max_rr=rr_ok.max(),
        rr_range=rr_ok.max() - rr_ok.min(), sd1=sd1, sd2=sd2, lf=lf, hf=hf,
        lf_hf=lf / hf if hf > 0 else np.nan,
        n_ventricular=int(vent.sum()), pvc_burden=float(vent.mean()),
        max_consec_ventricular=max_run(vent),
        non_n_fraction=float(np.mean(sym != "N")),
        premature_fraction=float(np.mean(rr_ok < 0.6 * np.median(rr_ok))),
    )



def sddb_vf_onset(hdr, duration_sec):
    """SDDB stores the VF onset as clock time in the header comment
    'vfon: HH:MM:SS'. Convert to seconds from the start of the record.
    Returns (onset_sec or None, explanation)."""
    m = None
    for c in hdr.comments or []:
        m = re.search(r"vfon:\s*(\d+):(\d+):(\d+)", c)
        if m:
            break
    bt = getattr(hdr, "base_time", None)
    if not m:
        return None, f"no 'vfon' in header comments {hdr.comments}"
    if bt is None:
        return None, "header has no start time (base_time)"
    vf = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
    start = bt.hour * 3600 + bt.minute * 60 + bt.second
    delta = (vf - start) % 86400
    cands = [delta + k * 86400 for k in range(3)
             if delta + k * 86400 <= duration_sec]
    if not cands:
        return None, (f"vfon {m[0][5:]} is outside the record "
                      f"(start {bt}, length {duration_sec / 60:.0f} min)")
    return max(cands), f"start {bt}, vfon {m[0][5:].strip()}"

# -------------------------------------------------------------- events ----
def find_events(cls, min_event_sec, merge_gap_sec):
    """Sustained class 1-3 runs, merged into episodes: list of [on, off]."""
    sev = (cls <= 3).astype(np.int8)
    d = np.diff(np.concatenate([[0], sev, [0]]))
    on, off = np.where(d == 1)[0], np.where(d == -1)[0]
    runs = [(a, b) for a, b in zip(on, off)
            if (b - a) >= min_event_sec * TARGET_FS]
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < merge_gap_sec * TARGET_FS:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return merged


def label_windows(starts, ends, cls, events, a):
    """Returns (status, time_to_onset_sec, episode_idx).
    status: 1 = pre-arrest, 0 = negative, -1 = excluded."""
    cs = np.concatenate([[0], np.cumsum(cls <= 3)])
    excl = (cs[ends] - cs[starts]) > 0                 # contains VF/VT
    cn = np.concatenate([[0], np.cumsum(cls == 7)])
    excl |= (cn[ends] - cn[starts]) > 0               # contains NOISE
    tn = np.full(len(starts), np.inf)
    ep = np.full(len(starts), -1)
    post = a.post_buffer_min * 60 * TARGET_FS
    if events:
        onsets = np.array([e[0] for e in events])
        for on, off in events:
            excl |= (starts < off + post) & (ends > on)   # during + recovery
        idx = np.searchsorted(onsets, ends, side="left")  # next onset >= end
        has = idx < len(onsets)
        tn[has] = (onsets[idx[has]] - ends[has]) / TARGET_FS
        ep[has] = idx[has]
    lead, hor = a.min_lead_sec, a.horizon_min * 60
    pos = ~excl & (tn >= lead) & (tn <= hor)
    grey = ~excl & ~pos & ((tn < lead) | ((tn > hor) &
                                          (tn <= hor + a.neg_buffer_min * 60)))
    status = np.where(excl | grey, -1, np.where(pos, 1, 0))
    return status, tn, ep


# ------------------------------------------------------------- records ----
def process_record(source, base, rng, a):
    name = base.name
    L = int(round(a.win_sec * TARGET_FS))
    stride = int(round(a.stride_sec * TARGET_FS))

    hdr = wfdb.rdheader(str(base))
    fs = hdr.fs
    ch = pick_lead(hdr.sig_name)
    x = raw = None
    beats = read_beats(base, fs)
    if not a.no_signal or beats is None:   # signal needed for X or beat detection
        raw = wfdb.rdrecord(str(base), channels=[ch]).p_signal[:, 0]
        n_orig = len(raw)
    else:
        n_orig = hdr.sig_len
    if not a.no_signal:
        x = clean(raw, fs)
        n_new = len(x)
    else:
        n_new = int(round(n_orig * TARGET_FS / fs))
    if n_new < L:
        return [], [], "too short"

    ann = read_annotation(base)
    if ann is None:
        return [], [], "no rhythm annotation file"
    if beats is None:                      # e.g. VFDB: rhythm labels only
        from wfdb import processing
        idx_q = processing.xqrs_detect(sig=np.nan_to_num(raw), fs=fs,
                                       verbose=False)
        if len(idx_q) < 100:
            return [], [], "beat detection found <100 beats"
        bt, bs, beat_ext = idx_q / fs, np.array(["N"] * len(idx_q)), "xqrs"
    else:
        bt, bs, beat_ext = beats

    cls = rhythm_class_array(ann, n_orig, n_new)
    note = ""
    if source == "sddb":                   # VF persists from onset to the end
        onset, why = sddb_vf_onset(hdr, n_orig / fs)
        if onset is None:
            return [], [], "skipped: " + why
        cls[int(onset * TARGET_FS):] = 1
        note = (f" | VF onset at {onset / 60:.1f} of {n_orig / fs / 60:.1f} min"
                f" ({why})")
    events = find_events(cls, a.min_event_sec, a.merge_gap_sec)
    starts = np.arange(0, n_new - L + 1, stride)
    ends = starts + L
    status, tn, ep = label_windows(starts, ends, cls, events, a)

    pos_idx = np.where(status == 1)[0]
    neg_idx = np.where(status == 0)[0]
    if len(neg_idx) > a.neg_cap:
        neg_idx = rng.choice(neg_idx, a.neg_cap, replace=False)
    idx = np.sort(np.concatenate([pos_idx, neg_idx]))

    group = f"{source}_{MITDB_SAME_PATIENT.get(name, name)}" \
        if source == "mitdb" else f"{source}_{name}"
    X, rows = [], []
    for i in idx:
        ws, we = starts[i] / TARGET_FS, ends[i] / TARGET_FS
        f = hrv_features(bt, bs, ws, we, a.min_beats)
        if f is None:
            continue
        if x is not None:
            w = zscore(x[starts[i]:ends[i]])
            if w is None:
                continue
            X.append(w)
        rows.append(dict(
            source=source, record=name, group=group, start_sec=round(ws, 2),
            end_sec=round(we, 2), label_prearrest=int(status[i]),
            time_to_onset_sec=round(float(tn[i]), 1) if np.isfinite(tn[i]) else np.nan,
            episode_id=f"{group}_ep{ep[i]}" if status[i] == 1 else "",
            split=split_of(group), orig_fs=fs, beat_src=beat_ext, **f))
    msg = (f"{len(events)} episodes, {int((status == 1).sum())} pos / "
           f"{int((status == 0).sum())} neg candidates, kept {len(rows)}"
           f" (beats from .{beat_ext})" + note)
    return X, rows, msg


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", default="./data")
    ap.add_argument("--out", default="./processed_pred")
    for k in ("mitdb", "vfdb", "sddb"):
        ap.add_argument(f"--{k}", default=None, help=f"override path for {k}")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--win-sec", type=float, default=60.0)
    ap.add_argument("--stride-sec", type=float, default=30.0)
    ap.add_argument("--horizon-min", type=float, default=5.0,
                    help="positive if an event starts within this many minutes")
    ap.add_argument("--min-lead-sec", type=float, default=10.0,
                    help="windows ending closer than this to onset are dropped")
    ap.add_argument("--neg-buffer-min", type=float, default=5.0,
                    help="grey zone after the horizon, dropped from negatives")
    ap.add_argument("--post-buffer-min", type=float, default=5.0,
                    help="windows this soon after an episode are dropped")
    ap.add_argument("--merge-gap-sec", type=float, default=30.0)
    ap.add_argument("--min-event-sec", type=float, default=2.0)
    ap.add_argument("--min-beats", type=int, default=30)
    ap.add_argument("--neg-cap", type=int, default=300,
                    help="max negative windows kept per record")
    ap.add_argument("--no-signal", action="store_true",
                    help="features only (much faster, no raw ECG in the npz)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    X, rows = [], []
    for k in ("vfdb", "mitdb", "sddb"):
        if k in a.skip:
            continue
        p = Path(getattr(a, k) or Path(a.root) / k)
        if not p.exists():
            print(f"[warn] {k}: {p} not found, skipping")
            continue
        print(f"[{k}] {p}")
        for base in find_records(p):
            try:
                x, r, msg = process_record(k, base, rng, a)
            except Exception as e:
                print(f"  [skip] {k}/{base.name}: {e}")
                continue
            print(f"  {k}/{base.name}: {msg}")
            X += x
            rows += r

    if not rows:
        sys.exit("No windows produced - check paths / annotation files.")

    meta = pd.DataFrame(rows)
    meta.to_csv(out / "prediction_features.csv", index=False)
    feats = meta[FEATURE_NAMES].to_numpy(np.float32)
    save = dict(y=meta.label_prearrest.values.astype(np.int8), features=feats,
                feature_names=np.array(FEATURE_NAMES), fs=np.int32(TARGET_FS))
    if X:
        save["X"] = np.stack(X).astype(np.float32)
    np.savez_compressed(out / "prediction_dataset.npz", **save)

    print("\n=========== SUMMARY ===========")
    print(f"windows: {len(meta)}   positives: {int(meta.label_prearrest.sum())}"
          f"   features: {len(FEATURE_NAMES)}"
          + (f"   X shape: {save['X'].shape}" if X else "   (no raw signal)"))
    print("\nlabel by source:\n", pd.crosstab(meta.source, meta.label_prearrest))
    print("\nlabel by split:\n", pd.crosstab(meta.split, meta.label_prearrest))
    n_ep = meta[meta.label_prearrest == 1].episode_id.nunique()
    print(f"\ndistinct pre-arrest episodes: {n_ep}")
    for s in ("train", "val", "test"):
        sub = meta[(meta.split == s) & (meta.label_prearrest == 1)]
        if sub.empty:
            print(f"[warn] split '{s}' has NO positive windows - use "
                  "leave-one-group-out CV instead of the hash split")
    if UNKNOWN_TOKENS:
        print("[warn] unmapped rhythm tokens (treated as class 6):",
              sorted(UNKNOWN_TOKENS))
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()