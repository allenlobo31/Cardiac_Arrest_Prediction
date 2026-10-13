from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILES = (
	RAW_DIR / "nsrdb_features_cache.csv",
	RAW_DIR / "sddb_features_cache.csv",
)
OUTPUT_FILE = PROCESSED_DIR / "hrv_features_dataset.csv"


def combine_datasets() -> pd.DataFrame:
	"""Combine the NSRDB and SDDB feature rows into one training dataset."""
	missing_files = [path.name for path in INPUT_FILES if not path.exists()]
	if missing_files:
		raise FileNotFoundError(
			"Missing input file(s): " + ", ".join(missing_files)
		)

	datasets = [pd.read_csv(path) for path in INPUT_FILES]

	first_columns = list(datasets[0].columns)
	for path, dataset in zip(INPUT_FILES[1:], datasets[1:]):
		if list(dataset.columns) != first_columns:
			raise ValueError(
				f"Column mismatch between {INPUT_FILES[0].name} and {path.name}"
			)

	combined = pd.concat(datasets, ignore_index=True)
	combined = combined.drop_duplicates().reset_index(drop=True)
	combined.to_csv(OUTPUT_FILE, index=False)
	return combined


if __name__ == "__main__":
	combined_dataset = combine_datasets()
	print(f"Combined dataset saved to: {OUTPUT_FILE.name}")
	print(f"Rows: {len(combined_dataset)}")
	print(f"Columns: {len(combined_dataset.columns)}")
	if "label" in combined_dataset.columns:
		print("Label counts:")
		print(combined_dataset["label"].value_counts().sort_index().to_string())
# =====================================================================
# PART 1 - DATASET PROCESSING  (Colab: paste as cells, or upload and run)
# Input : the 6 PhysioNet zips in ZIP_DIR
# Output: outputs/train_ready.csv (use for training), outputs/final_dataset.csv (raw), outputs/errors.csv
#
# Local: pip install wfdb numpy scipy pandas tqdm   |  Colab first cell: !pip -q install wfdb
# Step A: set INSPECT_ONLY = True, run, check the printed structure.
# Step B: set INSPECT_ONLY = False, run the full build.
# =====================================================================
import os, glob, re, zipfile, warnings, collections, time, shutil, subprocess
from fractions import Fraction
import numpy as np, pandas as pd
from scipy import signal as sg, stats
import wfdb
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

# ------------------------------ CONFIG -------------------------------
INSPECT_ONLY = False   # True = only print dataset structure and stop
PRINT_INSPECT = True   # print dataset structure at the start of every run
TEST_FRAC = 0.2        # share of patients held out as split == 'test'
# Local paths (VS Code / Linux). On Colab set BASE = "/content" instead.
BASE = os.path.dirname(os.path.abspath(__file__))
ZIP_DIR, DATA_DIR, OUT_DIR = (os.path.join(BASE, d) for d in ("zips", "data", "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)

# key -> substring expected in the zip filename (taken from your URLs)
ZIPS = {"sddb": "sudden-cardiac-death", "fragment": "ecg-fragment", "vfdb": "malignant-ventricular",
        "staff3": "staff-iii", "ptb": "ptb-diagnostic", "mitdb": "mit-bih-arrhythmia"}

# Download straight from PhysioNet into DATA_DIR/<key>/ (no zip upload needed).
# Zips in ZIP_DIR are only used as a fallback if a download fails / is turned off.
DOWNLOAD = False      # you already synced with aws s3; wget only runs for datasets with no records
SLUGS = {"sddb": "sddb", "fragment": "ecg-fragment-high-risk-label", "vfdb": "vfdb",
         "staff3": "staffiii", "ptb": "ptbdb", "mitdb": "mitdb"}   # from your page URLs
VERSION = "1.0.0"
DOWNLOAD_ORDER = ["mitdb", "vfdb", "sddb", "ptb", "staff3", "fragment"]   # small -> large
# add "nsrdb": ("nsrdb", "<version from its PhysioNet page>") style entries yourself if you use it

FS = 250                        # common sampling rate after resampling
WIN_SEC, STRIDE_SEC, MIN_SEC = 60, 30, 8
PRE_SEC        = 30*60          # window ends <=30 min before VF/VFL onset -> label 1
NEG_GAP_SEC    = 120*60         # window ends >2 h before onset (or no event) -> label 0
POST_GUARD_SEC = 10*60          # drop windows right after an episode
EVENT_RHYTHMS  = {"(VF", "(VFL"}    # add "(VT" if VT should count as an arrest-type event
MAX_NEG_PER_REC = 150
PTB_MI_AS_RISK, STAFF3_AS_RISK = False, True
SEED = 42
PREFER = ["MLII", "II", "ii", "ECG1", "ECG", "i", "V5"]   # lead preference

rng = np.random.default_rng(SEED)
W, S = WIN_SEC*FS, STRIDE_SEC*FS


# ------------------------------ SETUP --------------------------------
def has_records(key):
    return len(glob.glob(f"{DATA_DIR}/{key}/**/*.hea", recursive=True)) > 0

def download_all():
    """wget -c resumes, -N skips files already complete, so re-running is safe."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if shutil.which("wget") is None:
        print("[WARN] wget not found; skipping download (zip fallback will be used)"); return
    for key in DOWNLOAD_ORDER:
        if key not in SLUGS: continue
        if has_records(key): print(f"[download] {key}: already present, skipping"); continue
        url = f"https://physionet.org/files/{SLUGS[key]}/{VERSION}/"
        out = f"{DATA_DIR}/{key}"; os.makedirs(out, exist_ok=True)
        t = time.time(); print(f"[download] {key}: {url}")
        r = subprocess.run(["wget", "-r", "-N", "-c", "-np", "-nv", "-nH", "--cut-dirs=3",
                            "-R", "index.html*", "-P", out, url], capture_output=True, text=True)
        n_hea = len(glob.glob(f"{out}/**/*.hea", recursive=True))
        n_files = sum(len(f) for _, _, f in os.walk(out))
        status = "OK" if (r.returncode == 0 and n_files > 0) else f"PROBLEM (wget exit {r.returncode})"
        print(f"[download] {key}: {status} | {n_files} files, {n_hea} .hea | {time.time()-t:.0f}s")
        if status != "OK": print("  last wget output:", (r.stderr or r.stdout)[-300:])

def unzip_all():
    os.makedirs(ZIP_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)
    for key, sub in ZIPS.items():
        if has_records(key): print(f"[OK] {key}: already present (downloaded)"); continue
        hits = [z for z in glob.glob(f"{ZIP_DIR}/*.zip") if sub in os.path.basename(z).lower()]
        out = f"{DATA_DIR}/{key}"
        if not hits: print(f"[MISSING] zip for {key}"); continue
        if not os.path.isdir(out):
            with zipfile.ZipFile(hits[0]) as z: z.extractall(out)
        print(f"[OK] {key}: {os.path.basename(hits[0])}")

def record_stems(key):
    return sorted(p[:-4] for p in glob.glob(f"{DATA_DIR}/{key}/**/*.hea", recursive=True))

def inspect_all():
    for key in ZIPS:
        stems = record_stems(key)
        files = [f for f in glob.glob(f"{DATA_DIR}/{key}/**/*", recursive=True) if os.path.isfile(f)]
        exts = pd.Series([os.path.splitext(f)[1] for f in files]).value_counts().to_dict()
        print(f"\n=== {key}: {len(stems)} wfdb records | extensions: {exts}")
        if stems:
            h = wfdb.rdheader(stems[0])
            print("first record:", os.path.basename(stems[0]), "| fs", h.fs, "| len", h.sig_len, "| leads", h.sig_name)
            print("header comments:", h.comments[:8])
            print("sibling files:", sorted(os.path.basename(f) for f in glob.glob(stems[0] + ".*")))
        else:
            print("sample files:", [os.path.relpath(f, DATA_DIR) for f in files[:8]])


# ------------------------- SIGNAL + FEATURES -------------------------
def load_signal(stem):
    hdr = wfdb.rdheader(stem)
    names = [n.strip() for n in hdr.sig_name]
    ch = next((names.index(p) for p in PREFER if p in names), 0)
    rec = wfdb.rdrecord(stem, channels=[ch])
    x = np.nan_to_num(rec.p_signal[:, 0].astype(np.float64))
    fs = float(rec.fs)
    if abs(fs - FS) > 1e-6:
        fr = Fraction(FS/fs).limit_denominator(1000)
        x = sg.resample_poly(x, fr.numerator, fr.denominator)
    sos = sg.butter(4, [0.5, 40], btype="band", fs=FS, output="sos")
    return sg.sosfiltfilt(sos, x).astype(np.float32), fs, names[ch], hdr.comments

def detect_rpeaks(x):   # simple fast detector; swap in neurokit2/xqrs for higher fidelity
    y = sg.sosfiltfilt(sg.butter(2, [5, 15], btype="band", fs=FS, output="sos"), x)
    k = int(.12*FS)
    env = np.convolve(np.diff(y, prepend=y[0])**2, np.ones(k)/k, mode="same")
    pk, _ = sg.find_peaks(env, height=0.3*np.percentile(env, 99), distance=int(.25*FS))
    w = int(.05*FS)
    r = [max(0, p-w) + np.argmax(np.abs(x[max(0, p-w):p+w])) for p in pk]
    return np.unique(r).astype(int)

def sampen(x, m=2, rf=0.2):
    x = np.asarray(x); N = len(x); r = rf*np.std(x)
    if N < m+3 or r == 0: return np.nan
    def cnt(mm):
        t = np.array([x[i:i+mm] for i in range(N-m)])
        d = np.max(np.abs(t[:, None, :] - t[None, :, :]), axis=2)
        return np.sum(d <= r) - len(t)
    B, A = cnt(m), cnt(m+1)
    return -np.log(A/B) if A > 0 and B > 0 else np.nan

def hrv_features(r):
    f = {"n_beats": len(r)}
    rr = np.diff(r)/FS
    rr = rr[(rr > .25) & (rr < 2.5)]
    if len(rr) < 10: return f
    d = np.diff(rr); med = np.median(rr); sd = rr.std(ddof=1)
    f.update(mean_rr=rr.mean(), median_rr=med, mean_hr=60/rr.mean(), sdnn=sd,
             rmssd=np.sqrt(np.mean(d**2)), sdsd=d.std(ddof=1),
             pnn50=np.mean(np.abs(d) > .05)*100, pnn20=np.mean(np.abs(d) > .02)*100,
             cv_rr=sd/rr.mean(), min_rr=rr.min(), max_rr=rr.max(), range_rr=np.ptp(rr),
             skew_rr=stats.skew(rr), kurt_rr=stats.kurtosis(rr),
             prem_ratio=np.mean(rr < .8*med), pause_ratio=np.mean(rr > 1.3*med),
             hr_slope=np.polyfit(np.arange(len(rr)), 60/rr, 1)[0], sampen=sampen(rr))
    sd1 = d.std(ddof=1)/np.sqrt(2); sd2 = np.sqrt(max(2*sd**2 - sd1**2, 1e-12))
    f.update(sd1=sd1, sd2=sd2, sd1_sd2=sd1/sd2)
    t = np.cumsum(rr); fr = np.linspace(.04, .4, 100)
    p = sg.lombscargle(t, rr - rr.mean(), 2*np.pi*fr)     # LF/HF are weak on 60 s windows
    lf, hf = p[(fr >= .04) & (fr < .15)].sum(), p[fr >= .15].sum()
    f.update(lf=lf, hf=hf, lf_hf=lf/hf if hf > 0 else np.nan, lf_nu=lf/(lf+hf) if lf+hf > 0 else np.nan)
    return f

def morph_features(x, r):   # amplitude features are ratios to R-amplitude (avoids gain/lead differences)
    a, b = int(.08*FS), int(.04*FS)
    ok = r[(r > int(.11*FS)) & (r < len(x) - int(.36*FS))]
    if len(ok) < 5: return {}
    st, ta, ra, qw = [], [], [], []
    for p in ok:
        base = np.median(x[p-a:p-b])
        ra.append(abs(x[p] - base))
        st.append(np.median(x[p+int(.08*FS):p+int(.12*FS)]) - base)
        ta.append(np.max(x[p+int(.15*FS):p+int(.35*FS)]) - base)
        seg = np.abs(x[p-int(.1*FS):p+int(.1*FS)] - base)
        qw.append(np.sum(seg > .5*seg.max())/FS*1000)
    st, ta, ra, qw = map(np.array, (st, ta, ra, qw)); R = ra.mean() + 1e-9
    return dict(st_ratio=st.mean()/R, st_ratio_std=st.std()/R, t_ratio=ta.mean()/R,
                t_ratio_std=ta.std()/R, r_amp_cv=ra.std()/R, qrs_width_ms=qw.mean(), qrs_width_std=qw.std())

def spectral_features(x):
    fq, P = sg.welch(x, fs=FS, nperseg=min(len(x), 4*FS))
    m = (fq >= .5) & (fq <= 40); fq, P = fq[m], P[m]; Pn = P/P.sum()
    band = lambda lo, hi: Pn[(fq >= lo) & (fq < hi)].sum()
    d1 = np.diff(x); d2 = np.diff(d1)
    mob = np.sqrt(d1.var()/x.var()); cmp_ = np.sqrt(d2.var()/d1.var())/mob
    return dict(dom_freq=fq[np.argmax(P)], spec_entropy=-(Pn*np.log(Pn + 1e-12)).sum()/np.log(len(Pn)),
                spec_centroid=(fq*Pn).sum(), bp_0p5_3=band(.5, 3), bp_3_7=band(3, 7),
                bp_7_15=band(7, 15), bp_15_40=band(15, 40), sig_skew=stats.skew(x),
                sig_kurt=stats.kurtosis(x), zcr=np.mean(np.diff(np.sign(x)) != 0),
                hjorth_mob=mob, hjorth_cmp=cmp_)

def window_features(xw):
    xw = xw.astype(np.float64) - np.median(xw)
    s = xw.std()
    if s < 1e-6 or len(xw) < MIN_SEC*FS: return None     # flat line / too short
    xn = xw/s; r = detect_rpeaks(xn)
    return {**hrv_features(r), **morph_features(xn, r), **spectral_features(xn)}

def extract_rows(ds, rec, patient, x, segs, lead):
    rows = []
    for a, b, lab, mte in segs:
        f = window_features(x[a:b])
        if f is None: continue
        f.update(dataset=ds, record=rec, patient=patient, lead=lead, start_s=a/FS, label=lab, min_to_event=mte)
        rows.append(f)
    return rows


# ---------------------------- LABELING -------------------------------
# label: 0 = normal/low-risk, 1 = pre-arrest / high-risk, 2 = arrest (VF/VFL in progress)
def read_rhythm(stem):
    for ext in ("atr", "ari", "ecg", "alm"):
        if os.path.exists(f"{stem}.{ext}"):
            try: ann = wfdb.rdann(stem, ext)
            except Exception: continue
            ev = [(int(s), a.strip("\x00 \n")) for s, a in zip(ann.sample, ann.aux_note)
                  if a and a.strip("\x00 \n").startswith("(")]
            if ev: return ev
    return []

def episodes_from_rhythm(ev, scale, n):
    eps = []
    for i, (s, r) in enumerate(ev):
        e = ev[i+1][0] if i+1 < len(ev) else n/scale
        if r in EVENT_RHYTHMS:
            a, b = int(s*scale), int(e*scale)
            if eps and a <= eps[-1][1]: eps[-1] = (eps[-1][0], b)
            else: eps.append((a, b))
    return eps

def segs_from_episodes(n, eps):
    on = np.array([e[0] for e in eps]); off = np.array([e[1] for e in eps]); segs = []
    for a in range(0, n - W + 1, S):
        b, mid = a + W, a + W//2
        if len(eps):
            if np.any((on <= mid) & (mid < off)): segs.append((a, b, 2, 0.0)); continue
            if np.any((on < b) & (off > a)): continue          # partial overlap -> ambiguous
            fut, past = on[on >= b], off[off <= a]
            t_to = (fut.min() - b)/FS if len(fut) else np.inf
            t_since = (a - past.max())/FS if len(past) else np.inf
        else:
            t_to = t_since = np.inf
        if t_to <= PRE_SEC: lab = 1
        elif t_since < POST_GUARD_SEC: continue
        elif t_to > NEG_GAP_SEC: lab = 0
        else: continue                                         # gray zone
        segs.append((a, b, lab, t_to/60 if np.isfinite(t_to) else np.nan))
    return segs

def subsample(segs):
    pos = [s for s in segs if s[2] > 0]; neg = [s for s in segs if s[2] == 0]
    if len(neg) > MAX_NEG_PER_REC:
        neg = [neg[i] for i in sorted(rng.choice(len(neg), MAX_NEG_PER_REC, replace=False))]
    return pos + neg

def tile_range(a, b, lab):
    if b - a < 20*FS: return []
    if b - a < W: return [(a, b, lab, np.nan)]
    return [(a+s, a+s+W, lab, np.nan) for s in range(0, b-a-W+1, S)]

def err(errs, ds, stem, msg): errs.append(dict(dataset=ds, record=os.path.basename(stem), error=msg))


# ---------------------------- LOADERS --------------------------------
def run_rhythm_dataset(key, require_event, rows, errs):      # sddb, vfdb, mitdb (and nsrdb if added)
    for stem in tqdm(record_stems(key), desc=key):
        try:
            x, fs0, lead, _ = load_signal(stem)
            eps = episodes_from_rhythm(read_rhythm(stem), FS/fs0, len(x))
            if require_event and not eps:
                err(errs, key, stem, "no VF/VFL episode found in annotations - skipped"); continue
            rid = os.path.basename(stem)
            rows.extend(extract_rows(key, rid, rid, x, subsample(segs_from_episodes(len(x), eps)), lead))
        except Exception as e:
            err(errs, key, stem, repr(e))

def run_ptb(rows, errs):
    dxs = collections.Counter()
    for stem in tqdm(record_stems("ptb"), desc="ptb"):
        try:
            x, _, lead, com = load_signal(stem)
            dx = next((c.split(":", 1)[1].strip().lower() for c in com
                       if c.lower().startswith("reason for admission")), None)
            dxs[dx] += 1
            lab = 0 if dx == "healthy control" else (1 if (dx == "myocardial infarction" and PTB_MI_AS_RISK) else None)
            if lab is None: continue
            pat = os.path.basename(os.path.dirname(stem))
            rows.extend(extract_rows("ptb", os.path.basename(stem), pat, x, tile_range(0, len(x), lab), lead))
        except Exception as e:
            err(errs, "ptb", stem, repr(e))
    print("PTB diagnosis counts (verify these match the dataset):", dict(dxs))

# ---- HOOKS: fill after reading the INSPECT output (send it to me and I'll write them) ----
def staff3_intervals(stem, n):
    """Return [(label, start_sample, end_sample), ...] at FS rate, e.g.
       [(0, 0, infl_start), (1, infl_start, infl_end)], or None to skip."""
    return None

def fragment_label(stem, comments):
    """Return 0/1 (1 = high-risk) for this fragment, or None to skip."""
    return None

def run_staff3(rows, errs):
    for stem in tqdm(record_stems("staff3"), desc="staff3"):
        try:
            x, _, lead, _ = load_signal(stem)
            iv = staff3_intervals(stem, len(x))
            if iv is None: err(errs, "staff3", stem, "no interval hook implemented - skipped"); continue
            segs = [s for lab, a, b in iv for s in tile_range(a, b, lab if lab == 0 or STAFF3_AS_RISK else -1) if s[2] >= 0]
            rid = os.path.basename(stem)
            rows.extend(extract_rows("staff3", rid, rid[:3], x, segs, lead))   # patient id guess: adjust!
        except Exception as e:
            err(errs, "staff3", stem, repr(e))

def run_fragment(rows, errs):
    for stem in tqdm(record_stems("fragment"), desc="fragment"):
        try:
            x, _, lead, com = load_signal(stem)
            lab = fragment_label(stem, com)
            if lab is None: err(errs, "fragment", stem, "no label hook implemented - skipped"); continue
            rid = os.path.basename(stem)
            rows.extend(extract_rows("fragment", rid, rid, x, [(0, len(x), lab, np.nan)], lead))
        except Exception as e:
            err(errs, "fragment", stem, repr(e))


# ------------------- TRAINING-READY DATASET --------------------------
META_COLS = ["dataset", "record", "patient", "lead", "start_s", "label", "min_to_event"]

def patient_split(df):
    """Patient-wise train/test split, stratified by whether a patient has any risk window."""
    g = (df.dataset + "_" + df.patient.astype(str))
    has_risk = df.assign(g=g).groupby("g").label.max().gt(0)
    test = set()
    for flag in (True, False):
        ids = np.array(sorted(has_risk[has_risk == flag].index))
        if len(ids) == 0: continue
        rng.shuffle(ids)
        k = max(1, int(round(TEST_FRAC*len(ids)))) if len(ids) > 1 else 0
        test |= set(ids[:k])
    return np.where(g.isin(test), "test", "train")

def make_train_ready(df):
    """Clean numeric features, add group + split columns. Missing values are filled with TRAIN-split
    medians only (no test leakage); hrv_missing flags windows where no usable RR series was found."""
    df = df.reset_index(drop=True)
    split = patient_split(df)
    feat = [c for c in df.columns if c not in META_COLS]
    X = df[feat].replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    hrv_missing = X["mean_rr"].isna().astype(int) if "mean_rr" in X else pd.Series(0, index=X.index)
    X = X.fillna(X[split == "train"].median())
    X = X.fillna(0.0)
    out = pd.concat([df[["dataset", "record", "patient", "label", "min_to_event"]], X], axis=1)
    out.insert(3, "group", out.dataset + "_" + out.patient.astype(str))
    out.insert(4, "split", split)
    out["hrv_missing"] = hrv_missing.values
    return out


# ------------------------------ MAIN ---------------------------------
def main():
    if DOWNLOAD: download_all()
    unzip_all()    # fallback only: fills any dataset that still has no records
    if PRINT_INSPECT or INSPECT_ONLY: inspect_all()
    if INSPECT_ONLY:
        print("\nINSPECT_ONLY=True -> stopping. Set False for the full build."); return
    rows, errs, t0 = [], [], time.time()
    steps = [("sddb", lambda: run_rhythm_dataset("sddb", True, rows, errs)),
             ("vfdb", lambda: run_rhythm_dataset("vfdb", False, rows, errs)),
             ("mitdb", lambda: run_rhythm_dataset("mitdb", False, rows, errs)),
             ("ptb", lambda: run_ptb(rows, errs)),
             ("staff3", lambda: run_staff3(rows, errs)),
             ("fragment", lambda: run_fragment(rows, errs))]
    for name, fn in steps:
        t = time.time(); fn(); print(f"[time] {name}: {time.time()-t:.0f}s, rows so far {len(rows)}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows produced - check errors above / errors.csv"); 
        pd.DataFrame(errs, columns=["dataset", "record", "error"]).to_csv(f"{OUT_DIR}/errors.csv", index=False); return
    df.to_csv(f"{OUT_DIR}/final_dataset.csv", index=False)        # raw features (NaNs kept)
    ready = make_train_ready(df)
    ready.to_csv(f"{OUT_DIR}/train_ready.csv", index=False)       # <- use this for training
    edf = pd.DataFrame(errs, columns=["dataset", "record", "error"]); edf.to_csv(f"{OUT_DIR}/errors.csv", index=False)
    print(f"\nTOTAL time: {(time.time()-t0)/60:.1f} min | raw shape: {df.shape} | train_ready shape: {ready.shape}")
    print(pd.crosstab(df.dataset, df.label))
    print("\nsplit x label (patient-wise):\n", pd.crosstab(ready.split, ready.label))
    print("\nmissing-value rate (top 10):\n", df.isna().mean().sort_values(ascending=False).head(10))
    print(f"\nskipped/failed records: {len(edf)} -> {OUT_DIR}/errors.csv")
    if len(edf): print(edf.groupby(["dataset", "error"]).size().to_string())

if __name__ == "__main__":
    main()