"""
scripts/build_dataset.py

Builds a combined, REAL-DATA-ONLY HRV dataset from three PhysioNet sources:
  * MUSIC  (Sudden Cardiac Death in Chronic Heart Failure, 24 h Holter)  -> task "long_term_scd"
  * SDDB   (Sudden Cardiac Death Holter Database)                         -> task "imminent"
  * NSRDB  (Normal Sinus Rhythm Database)                                 -> task "imminent"

Nothing here simulates or augments data. Every row comes from a recorded ECG.

IMPORTANT - the two tasks are different questions and must NOT share one `label`:
  imminent       : "is this 5-min window from the 30 min before an arrest?"   (SDDB/NSRDB)
  long_term_scd  : "will this heart-failure patient die of SCD in ~4 years?" (MUSIC; label is
                   per PATIENT, copied to that patient's windows)
Train one model per `task`, always with patient-grouped CV (see train_model.py --task).

Commands
  python scripts/build_dataset.py music --music-dir /path/to/music-sudden-cardiac-death/1.0.1
  python scripts/build_dataset.py sddb-extra          # optional: extra SDDB patients via .ari (needs internet)
  python scripts/build_dataset.py combine [--exclude sddb_36 ...]
  python scripts/build_dataset.py audit               # writes data/processed/dataset_audit.json

Data licence: MUSIC is ODbL v1.0 per its PhysioNet page - read it before redistributing the
combined CSV. Cite MUSIC, SDDB, NSRDB and PhysioNet as each page requests.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / "data" / "raw"
PROCESSED = BASE_DIR / "data" / "processed"
MUSIC_CACHE = RAW / "music.csv"
MUSIC_MANIFEST = RAW / "music_manifest.csv"
SDDB_CACHE = RAW / "sddb_features_cache.csv"
SDDB_EXTRA_CACHE = RAW / "sddb_extra_features_cache.csv"
NSRDB_CACHE = RAW / "nsrdb_features_cache.csv"
COMBINED = PROCESSED / "hrv_features_dataset_v2.csv"
MUSIC_PATIENT = PROCESSED / "music_patient_features.csv"
AUDIT_JSON = PROCESSED / "dataset_audit.json"

# Same window definition as extract_sddb.py / extract_nsrdb.py so features are comparable.
WINDOW_SEC = 5 * 60
MIN_BEATS = 150
META_COLS = ["record_id", "source", "task", "peak_source", "window_end_sec", "label"]

# Column letters/names come from MUSIC's subject-info_codes.csv / _definitions.csv.
MUSIC_COLS = {
    "exit":   (["Exit of the study"], "D"),
    "cause":  (["Cause of death"], "E"),
    "rhythm": (["Holter rhythm"], "BT"),
}
MUSIC_CLINICAL = {  # optional, patient-level file only (the web app has no clinical inputs)
    "clin_nyha":    (["NYHA class"], "K"),
    "clin_lvef":    (["LVEF (%)"], "AZ"),
    "clin_probnp":  (["Pro-BNP (ng/L)"], "AN"),
}


# ----------------------------------------------------------------------------- helpers
class DatasetError(Exception):
    pass


def letter_to_idx(letters):
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def norm(s):
    return str(s).strip().lower()


def find_col(df, names, letter, allowed=None, what=""):
    """Locate a column by name, else by Excel letter. Validate values if `allowed` is given."""
    by_name = {norm(c): c for c in df.columns}
    col, how = None, None
    for n in names:
        if norm(n) in by_name:
            col, how = by_name[norm(n)], "name"
            break
    if col is None and letter_to_idx(letter) < len(df.columns):
        col, how = df.columns[letter_to_idx(letter)], f"letter {letter}"
    if col is None:
        raise DatasetError(f"[{what}] column not found by name {names} or letter {letter}. "
                           f"Columns seen: {list(df.columns)[:15]}...")
    if allowed is not None:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().unique()
        bad = sorted(set(vals) - set(allowed))
        if bad:
            raise DatasetError(f"[{what}] column '{col}' (found by {how}) has unexpected values {bad[:10]}; "
                               f"expected subset of {sorted(allowed)}. Refusing to guess.")
    return col


def read_subject_info(music_dir):
    path = Path(music_dir) / "subject-info.csv"
    if not path.exists():
        raise DatasetError(f"{path} not found")
    last = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:  # noqa: BLE001
            last = e
    raise DatasetError(f"could not parse {path}: {last}")


def find_id_col(df, record_ids):
    """The patient-ID column is the one whose values best match the Holter record names."""
    def to_rid(v):
        try:
            return f"P{int(float(v)):04d}"
        except (TypeError, ValueError):
            return str(v).strip().upper()
    best, best_hit = None, 0.0
    for c in df.columns:
        hit = pd.Series(df[c]).map(to_rid).isin(record_ids).mean()
        if hit > best_hit:
            best, best_hit = c, hit
    if best is None or best_hit < 0.5:
        raise DatasetError(f"no column in subject-info.csv matches Holter record names "
                           f"(best overlap {best_hit:.0%}); cannot link patients to recordings")
    return best, to_rid


# ----------------------------------------------------------------------------- MUSIC
def detect_rpeaks(record_path, channel, chunk_sec=1800):
    """R-peak sample indices for a whole 24 h record, processed in chunks to bound memory."""
    import neurokit2 as nk
    import wfdb

    hdr = wfdb.rdheader(str(record_path))
    fs, n = float(hdr.fs), int(hdr.sig_len)
    if channel >= hdr.n_sig:
        raise DatasetError(f"channel {channel} not in record ({hdr.n_sig} leads)")
    step, pad, peaks = int(chunk_sec * fs), int(5 * fs), []
    for start in range(0, n, step):
        end = min(start + step + pad, n)
        sig = wfdb.rdrecord(str(record_path), sampfrom=start, sampto=end, channels=[channel]).p_signal[:, 0]
        sig = pd.Series(sig).interpolate(limit_direction="both").fillna(0).to_numpy()
        if np.ptp(sig) == 0:
            continue                                         # flat-line chunk
        cleaned = nk.ecg_clean(sig, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
        p = np.asarray(info["ECG_R_Peaks"], dtype=int) + start
        peaks.append(p if end >= n else p[p < start + step])  # drop overlap duplicates
    if not peaks:
        raise DatasetError("no usable signal")
    return np.concatenate(peaks), fs


def music_windows(peaks, fs, record_id, label, max_windows):
    """Non-overlapping 5-min windows -> QC -> evenly sub-sampled -> NeuroKit2 HRV features."""
    import neurokit2 as nk

    t = peaks / fs
    cands = []
    for ws in np.arange(t[0], t[-1] - WINDOW_SEC, WINDOW_SEC):
        wp = peaks[(t >= ws) & (t < ws + WINDOW_SEC)]
        if len(wp) < MIN_BEATS:
            continue
        rr = np.diff(wp) / fs
        if np.mean((rr >= 0.3) & (rr <= 2.0)) < 0.9:         # detector failure / noise, not physiology
            continue
        cands.append((ws + WINDOW_SEC, wp))
    if not cands:
        return []
    keep = np.unique(np.linspace(0, len(cands) - 1, min(max_windows, len(cands))).astype(int))
    rows = []
    for i in keep:
        end, wp = cands[i]
        try:
            hrv = nk.hrv(wp, sampling_rate=fs, show=False)
        except Exception:  # noqa: BLE001
            continue
        hrv["record_id"], hrv["window_end_sec"], hrv["label"] = f"music_{record_id}", end, int(label)
        rows.append(hrv)
    return rows


def append_music(df):
    """Append one patient's windows to data/raw/music.csv with columns locked to a fixed schema:
    the existing music.csv header, else the sddb_features_cache.csv header (so MUSIC matches the
    other caches column-for-column), else this frame's own columns. Prevents silent column shifts."""
    if MUSIC_CACHE.exists():
        cols = list(pd.read_csv(MUSIC_CACHE, nrows=0).columns)
    elif SDDB_CACHE.exists():
        cols = list(pd.read_csv(SDDB_CACHE, nrows=0).columns)
    else:
        cols = list(df.columns)
    dropped = [c for c in df.columns if c not in cols]
    df.reindex(columns=cols).to_csv(MUSIC_CACHE, mode="a", header=not MUSIC_CACHE.exists(), index=False)
    return dropped


def cmd_music(a):
    music_dir = Path(a.music_dir)
    holter = music_dir / "Holter_ECG"
    if not holter.is_dir():
        raise DatasetError(f"{holter} not found - download at least some of Holter_ECG/ plus subject-info*.csv")
    available = sorted(p.stem for p in holter.glob("P*.hea") if (holter / f"{p.stem}.dat").exists())
    if not available:
        raise DatasetError("no complete PXXXX.hea/.dat pairs found in Holter_ECG/")

    info = read_subject_info(music_dir)
    id_col, to_rid = find_id_col(info, set(available))
    c_exit = find_col(info, *MUSIC_COLS["exit"], allowed={0, 1, 2, 3}, what="exit")
    c_cause = find_col(info, *MUSIC_COLS["cause"], allowed={0, 1, 3, 6, 7}, what="cause")
    c_rhythm = find_col(info, *MUSIC_COLS["rhythm"], allowed={0, 1, 2, 3}, what="rhythm")
    info["_rid"] = info[id_col].map(to_rid)
    info = info.drop_duplicates("_rid").set_index("_rid")
    ex = pd.to_numeric(info[c_exit], errors="coerce")
    ca = pd.to_numeric(info[c_cause], errors="coerce")
    rh = pd.to_numeric(info[c_rhythm], errors="coerce")

    RAW.mkdir(parents=True, exist_ok=True)
    done = set()
    if MUSIC_CACHE.exists():
        done = set(pd.read_csv(MUSIC_CACHE, usecols=["record_id"])["record_id"])
    manifest = []
    todo = available[: a.limit] if a.limit else available
    for rid in todo:
        if f"music_{rid}" in done:
            continue
        if rid not in info.index:
            manifest.append((rid, "excluded", "not in subject-info.csv", 0)); continue
        # Label: SCD vs clean survivors only. Other deaths, transplants and lost-to-follow-up are
        # competing outcomes with no valid binary label for this question -> excluded, not guessed.
        if ca[rid] == 3:
            label = 1
        elif ca[rid] == 0 and ex[rid] == 0:
            label = 0
        else:
            manifest.append((rid, "excluded", f"outcome cause={ca[rid]} exit={ex[rid]}", 0)); continue
        if not a.include_nonsinus and rh[rid] != 0:
            manifest.append((rid, "excluded", f"Holter rhythm={rh[rid]} (not sinus; HRV not comparable)", 0)); continue
        try:
            peaks, fs = detect_rpeaks(holter / rid, a.channel)
            rows = music_windows(peaks, fs, rid, label, a.max_windows_per_patient)
            if not rows:
                manifest.append((rid, "failed", "no windows passed QC", 0)); continue
            df = pd.concat(rows, ignore_index=True)
            dropped = append_music(df)
            if dropped:
                print(f"  note: {rid}: dropped features not in schema: {dropped}")
            manifest.append((rid, "included", f"label={label}", len(df)))
            print(f"{rid}: label={label} windows={len(df)}")
        except Exception as e:  # noqa: BLE001
            manifest.append((rid, "failed", f"{type(e).__name__}: {e}", 0))
            print(f"{rid}: FAILED {e}")
    m = pd.DataFrame(manifest, columns=["record", "status", "detail", "n_windows"])
    m.to_csv(MUSIC_MANIFEST, mode="a", header=not MUSIC_MANIFEST.exists(), index=False)
    print("\nManifest summary:\n", m["status"].value_counts().to_string() if len(m) else "(nothing new)")

    # Patient-level table: one row per patient (the statistically honest unit for MUSIC).
    if MUSIC_CACHE.exists():
        w = pd.read_csv(MUSIC_CACHE)
        feats = [c for c in w.columns if c.startswith("HRV_")]
        g = w.groupby("record_id")
        agg = pd.concat([g[feats].median().add_suffix("_med"), g[feats].std().add_suffix("_sd"),
                         g["label"].first(), g.size().rename("n_windows")], axis=1).reset_index()
        rid_plain = agg["record_id"].str.replace("music_", "", regex=False)
        for name, (names, letter) in MUSIC_CLINICAL.items():
            try:
                col = find_col(info.reset_index(), names, letter, what=name)
                agg[name] = pd.to_numeric(info[col].reindex(rid_plain).to_numpy(), errors="coerce")
            except DatasetError as e:
                print(f"note: clinical column {name} skipped ({e})")
        PROCESSED.mkdir(parents=True, exist_ok=True)
        agg.to_csv(MUSIC_PATIENT, index=False)
        print(f"Patient-level MUSIC table: {MUSIC_PATIENT} ({len(agg)} patients, "
              f"{int(agg['label'].sum())} SCD)")


# ----------------------------------------------------------------------------- SDDB extra
def cmd_sddb_extra(_a):
    """Stream SDDB records missing from the existing cache, using the unaudited .ari beat
    annotations (extract_sddb.py only tried the audited 'atr' set). Needs internet. UNTESTED offline."""
    import wfdb
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from extract_sddb import NORMAL_BEAT_SYMBOLS, extract_windows

    have = set(pd.read_csv(SDDB_CACHE, usecols=["record_id"])["record_id"]) if SDDB_CACHE.exists() else set()
    frames = []
    for rec in wfdb.get_record_list("sddb"):
        if f"sddb_{rec}" in have:
            continue
        try:
            ann = wfdb.rdann(rec, "ari", pn_dir="sddb/1.0.0")
            keep = np.array([s in NORMAL_BEAT_SYMBOLS for s in ann.symbol])
            samples = ann.sample[keep]
            rows = extract_windows(samples, samples / ann.fs, ann.fs, record_id=f"sddb_{rec}")
            frames.extend(rows)
            print(f"sddb/{rec} (ari): {len(rows)} labeled windows")
        except Exception as e:  # noqa: BLE001
            print(f"sddb/{rec}: skipped ({e})")
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(SDDB_EXTRA_CACHE, index=False)
        print(f"Saved {SDDB_EXTRA_CACHE}")
    print("REMINDER: .ari files are unaudited, and some SDDB patients have atrial fibrillation or pacing "
          "(HRV not comparable) - use `combine --exclude` for those.")


# ----------------------------------------------------------------------------- combine
def cmd_combine(a):
    parts = []
    spec = [(SDDB_CACHE, "sddb", "imminent", "atr"), (SDDB_EXTRA_CACHE, "sddb", "imminent", "ari"),
            (NSRDB_CACHE, "nsrdb", "imminent", "atr"), (MUSIC_CACHE, "music", "long_term_scd", "detected")]
    for path, source, task, peak_source in spec:
        if path.exists():
            d = pd.read_csv(path)
            d["source"], d["task"], d["peak_source"] = source, task, peak_source
            parts.append(d)
            print(f"{source:6s} {peak_source:8s} {len(d):6d} windows, {d['record_id'].nunique():4d} patients")
        else:
            print(f"{source:6s} {peak_source:8s} (no cache at {path.name}; skipped)")
    if not parts:
        raise DatasetError("no input caches found")
    df = pd.concat(parts, ignore_index=True)
    if a.exclude:
        n0 = len(df)
        df = df[~df["record_id"].isin(a.exclude)]
        print(f"excluded {n0 - len(df)} windows from {a.exclude}")
    hrv = sorted(c for c in df.columns if c.startswith("HRV_"))
    df = df[hrv + META_COLS].drop_duplicates(["record_id", "window_end_sec"]).reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMBINED, index=False)
    print(f"\nSaved {COMBINED}  shape={df.shape}")
    print(df.groupby(["task", "source", "label"]).agg(windows=("label", "size"),
                                                       patients=("record_id", "nunique")).to_string())


# ----------------------------------------------------------------------------- audit
def source_separability(df, a_src, b_src, per_src=1500, seed=0):
    """AUC of a classifier telling two sources apart using ONLY label==0 windows (so disease cannot
    explain it), patient-grouped. High AUC => the sources are distinguishable by recording artefacts."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    hrv = [c for c in df.columns if c.startswith("HRV_")]
    rng = np.random.RandomState(seed)
    d = df[(df["label"] == 0) & df["source"].isin([a_src, b_src])]
    if d.groupby("source")["record_id"].nunique().min() < 4:
        return None
    d = pd.concat([x.sample(min(per_src, len(x)), random_state=seed) for _, x in d.groupby("source")])
    X = d[hrv].replace([np.inf, -np.inf], np.nan)
    keep = X.columns[X.isna().mean() < 0.2]
    X = SimpleImputer(strategy="median").fit_transform(X[keep])
    y = (d["source"] == b_src).astype(int).to_numpy()
    p = np.zeros(len(y))
    for tr, te in GroupKFold(4).split(X, y, d["record_id"]):
        if len(set(y[tr])) < 2:
            return None
        m = RandomForestClassifier(200, min_samples_leaf=3, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p))


def cmd_audit(_a):
    errors, warns, info = [], [], {}
    if not COMBINED.exists():
        raise DatasetError(f"{COMBINED} missing - run `combine` first")
    df = pd.read_csv(COMBINED)
    hrv = [c for c in df.columns if c.startswith("HRV_")]

    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
    if not set(df["label"].dropna().unique()) <= {0, 1}:
        errors.append(f"labels outside {{0,1}}: {sorted(df['label'].dropna().unique())[:5]}")
    if df["label"].isna().any():
        errors.append(f"{int(df['label'].isna().sum())} rows with NaN label")
    n_inf = int(np.isinf(df[hrv].to_numpy(dtype=float)).sum())
    if n_inf:
        warns.append(f"{n_inf} inf values in features (will need cleaning before training)")
    multi = df.groupby("record_id")["source"].nunique()
    if (multi > 1).any():
        errors.append(f"record_id appears in more than one source: {list(multi[multi > 1].index[:5])}")
    if df.duplicated(["record_id", "window_end_sec"]).any():
        errors.append("duplicate (record_id, window_end_sec) rows")
    nan_frac = df[hrv].isna().mean()
    bad = nan_frac[nan_frac > 0.2]
    if len(bad):
        warns.append(f"{len(bad)} features >20% NaN (dropped at train time): {list(bad.index[:6])}...")

    for task, t in df.groupby("task"):
        per_pat = t.groupby("record_id")["label"].agg(["min", "max"])
        if task == "long_term_scd" and (per_pat["min"] != per_pat["max"]).any():
            errors.append(f"[{task}] label varies within a patient: {list(per_pat[per_pat['min'] != per_pat['max']].index[:5])}")
        pat = per_pat["max"]                      # "imminent": a patient is positive if ANY window is pre-arrest
        pos, neg = int((pat == 1).sum()), int((pat == 0).sum())
        info[task] = {"positive_patients": pos, "negative_patients": neg, "windows": int(len(t)),
                      "negatives_by_source": t[t.label == 0].groupby("source")["record_id"].nunique().to_dict(),
                      "positives_by_source": t[t.label == 1].groupby("source")["record_id"].nunique().to_dict()}
        if pos < 30:
            warns.append(f"[{task}] only {pos} positive patients - expect very wide CIs; results are fragile")
        if len(info[task]["positives_by_source"]) == 1 and len(info[task]["negatives_by_source"]) > 1:
            warns.append(f"[{task}] all positives come from '{list(info[task]['positives_by_source'])[0]}' but negatives "
                         f"from {list(info[task]['negatives_by_source'])}: source is partly confounded with label. "
                         f"Report leave-source-out results and compare against same-source negatives only.")
        wpp = t.groupby("record_id").size()
        if wpp.max() / max(wpp.min(), 1) > 20:
            warns.append(f"[{task}] windows per patient range {wpp.min()}-{wpp.max()}; weight or cap per patient")

    sep = {}
    sources = sorted(df["source"].unique())
    for i, s1 in enumerate(sources):
        for s2 in sources[i + 1:]:
            auc = source_separability(df, s1, s2)
            sep[f"{s1}_vs_{s2}"] = auc
            if auc is not None and auc > 0.9:
                warns.append(f"sources {s1} vs {s2} are separable from NORMAL windows alone (AUC={auc:.2f}). "
                             f"A model can learn the source instead of disease; do not pool their negatives for training.")
    info["source_separability_auc_negatives_only"] = sep
    report = {"ok": not errors, "errors": errors, "warnings": warns, "info": info}
    AUDIT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if not errors else 1)


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("music")
    m.add_argument("--music-dir", required=True, help="folder containing subject-info.csv and Holter_ECG/")
    m.add_argument("--limit", type=int, default=0, help="process only the first N recordings (for a trial run)")
    m.add_argument("--channel", type=int, default=0)
    m.add_argument("--max-windows-per-patient", type=int, default=48)
    m.add_argument("--include-nonsinus", action="store_true")
    sub.add_parser("sddb-extra")
    c = sub.add_parser("combine")
    c.add_argument("--exclude", nargs="*", default=[], help="record_ids to drop, e.g. sddb_36")
    sub.add_parser("audit")
    a = ap.parse_args()
    try:
        {"music": cmd_music, "sddb-extra": cmd_sddb_extra, "combine": cmd_combine, "audit": cmd_audit}[a.cmd](a)
    except DatasetError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()