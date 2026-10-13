#!/usr/bin/env python3
"""
Clean_dataset.py  (fixed v2)
============================
Input : prediction_features.csv  (output of preprocess_cardiac_arrest_prediction.py)
Output: train_ready.csv + cleaning_report.txt

WHAT THIS VERSION DOES
  * KEEPS every feature listed in FEATURES that exists in the CSV.
    Constant / stratum-constant / redundant (correlated) features are NOT
    dropped any more. They are only reported as [info] so you can decide.
    (Old behaviour is available through opt-in flags, see below.)
  * REMOVES unwanted COLUMNS: answer-leaking / bookkeeping metadata
        start_sec, end_sec, time_to_onset_sec, episode_id,
        source, record, group, beat_src, orig_fs
    and any other column that is neither a feature nor a kept meta column.
  * REPORTS MISSING FEATURES: any name in FEATURES that is not in the CSV is
    listed clearly (console + cleaning_report.txt). Use --require-all to make
    the script stop with an error instead of only warning.
  * REMOVES unusable ROWS: NaN/inf feature values, implausible mean HR,
    exact duplicates.
  * FIXES A DATA ARTIFACT: in VFDB the beats are detected from the signal
    (beat_src = xqrs / qrs), so every beat is labelled "N" and the four
    beat-TYPE features (n_ventricular, pvc_burden, max_consec_ventricular,
    non_n_fraction) are always ~0 there. That is "not measured", not "no
    ectopy", and it identifies the database (VFDB is ~39% positive vs ~3.5%
    elsewhere). Those values are set to NaN for such rows (rows are KEPT).
    Use --keep-untyped-ectopy to disable.
  * stratum = source (vfdb / mitdb / sddb) when the CSV has no 'stratum'
    column. It is for evaluation only, never a model input.

COLUMNS kept (not features, never feed them to the model)
  orig_row   row number in the original CSV / npz
  patient_id use for grouped cross-validation
  stratum    used by balance_dataset.py and per-stratum evaluation
  split      train / val / test
  label_prearrest   the target

Optional (OFF by default) feature dropping:
  --drop-constant            drop features with zero variance on the train split
  --drop-stratum-constant    drop features constant inside one stratum
  --corr-threshold 0.95      drop features with |Spearman r| above this value
                             (default 1.01 = disabled)
  --winsorize                clip features to the 0.5-99.5 percentile of train

Usage
    python clean_dataset.py --csv processed_pred/prediction_features.csv \
                            --out processed_pred/clean
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "n_beats", "mean_rr", "mean_hr", "sdnn", "rmssd", "sdsd", "pnn50", "cv_rr",
    "min_rr", "max_rr", "rr_range", "sd1", "sd2", "lf", "hf", "lf_hf",
    "n_ventricular", "pvc_burden", "max_consec_ventricular", "non_n_fraction",
    "premature_fraction",
]
# features computed from beat TYPE symbols (only valid when beats were annotated)
ECTOPY_FEATURES = ["n_ventricular", "pvc_burden", "max_consec_ventricular",
                   "non_n_fraction"]
# beat sources that carry no beat types (R-peaks detected from the signal / .qrs)
UNTYPED_BEAT_SRC = {"xqrs", "qrs"}
LABEL = "label_prearrest"
KEEP_META = ["orig_row", "patient_id", "stratum", "split", LABEL]
DROP_META = ["start_sec", "end_sec", "time_to_onset_sec", "episode_id",
             "source", "record", "group", "beat_src", "orig_fs"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="./clean")
    ap.add_argument("--hr-min", type=float, default=30.0)
    ap.add_argument("--hr-max", type=float, default=220.0)
    ap.add_argument("--require-all", action="store_true",
                    help="exit with an error if any feature in FEATURES is missing")
    ap.add_argument("--keep-untyped-ectopy", action="store_true",
                    help="do NOT set ectopy features to NaN for rows whose beats were "
                         "detected from the signal (xqrs/qrs)")
    ap.add_argument("--drop-constant", action="store_true",
                    help="drop features with zero variance (off by default)")
    ap.add_argument("--drop-stratum-constant", action="store_true",
                    help="drop features constant inside one stratum (off by default)")
    ap.add_argument("--corr-threshold", type=float, default=1.01,
                    help="drop a feature if |Spearman r| with a kept one is above this "
                         "(default 1.01 = disabled, all features are kept)")
    ap.add_argument("--winsorize", action="store_true",
                    help="clip features to the 0.5-99.5 percentile of the train split")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg=""):
        print(msg)
        log.append(msg)

    df = pd.read_csv(a.csv)
    df.insert(0, "orig_row", np.arange(len(df)))

    # ---- required columns ----
    if LABEL not in df.columns:
        sys.exit(f"ERROR: target column '{LABEL}' not found in {a.csv}")

    # ---- feature availability check ----
    feats = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    say(f"loaded {len(df)} rows, {int(df[LABEL].sum())} positives")
    say(f"features found: {len(feats)} of {len(FEATURES)}")
    if missing:
        say(f"\n[WARNING] MISSING FEATURES ({len(missing)}): {missing}")
        say("          These are not in the input CSV. Re-run the feature-extraction "
            "script if you need them.")
        if a.require_all:
            (out / "cleaning_report.txt").write_text("\n".join(log))
            sys.exit("ERROR: missing features and --require-all was set.")
    else:
        say("all expected features are present in the input CSV")

    # ---- columns: drop metadata / unwanted columns ----
    if "split" not in df.columns:
        df["split"] = "train"
    if "stratum" not in df.columns:
        df["stratum"] = df["source"].astype(str) if "source" in df.columns else "A"
    if "patient_id" not in df.columns:
        df["patient_id"] = df["group"] if "group" in df.columns else "P000"

    untyped = pd.Series(False, index=df.index)
    if "beat_src" in df.columns:
        untyped = df["beat_src"].astype(str).str.lower().isin(UNTYPED_BEAT_SRC)
    ect = [f for f in ECTOPY_FEATURES if f in feats]

    known = set(FEATURES) | set(KEEP_META) | set(DROP_META)
    dropped_meta = [c for c in DROP_META if c in df.columns]
    dropped_other = [c for c in df.columns if c not in known]
    say(f"\ncolumns dropped (metadata / not features): {dropped_meta}")
    if dropped_other:
        say(f"columns dropped (unknown, not a feature): {dropped_other}")
    df = df[KEEP_META + feats].copy()

    # ---- ectopy features are meaningless where beat types were not annotated ----
    allowed_nan = pd.DataFrame(False, index=df.index, columns=feats)
    if ect and untyped.any():
        allowed_nan.loc[untyped, ect] = True       # NaN is expected (not measured) there
        if a.keep_untyped_ectopy:
            say(f"\n[info] {int(untyped.sum())} rows have detector-derived beats "
                "(no beat types); ectopy features KEPT as-is (--keep-untyped-ectopy)")
        else:
            df[ect] = df[ect].astype(float)
            df.loc[untyped, ect] = np.nan
            by_src = df.loc[untyped, "stratum"].value_counts().to_dict()
            say(f"\nectopy features {ect}\n  set to NaN for {int(untyped.sum())} rows "
                f"whose beats were detected from the signal (beat_src in "
                f"{sorted(UNTYPED_BEAT_SRC)}): {by_src}\n  (not measured there, "
                "rows are KEPT; the model must not treat them as 'zero ectopy')")
    elif ect and "beat_src" not in df.columns:
        say("\n[info] no beat_src column: cannot tell which rows lack beat types")

    # ---- rows ----
    n0 = len(df)
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    bad = (df[feats].isna() & ~allowed_nan).any(axis=1)
    say(f"\nrows dropped - NaN/inf in a feature: {int(bad.sum())}")
    df = df[~bad]

    if "mean_hr" in feats:
        bad = ~df.mean_hr.between(a.hr_min, a.hr_max)
        say(f"rows dropped - mean HR outside {a.hr_min:g}-{a.hr_max:g} bpm: "
            f"{int(bad.sum())}")
        df = df[~bad]

    bad = df.duplicated(subset=feats + [LABEL, "patient_id"], keep="first")
    say(f"rows dropped - exact duplicates: {int(bad.sum())}")
    df = df[~bad]
    say(f"rows: {n0} -> {len(df)}")

    fit = df[df.split == "train"]
    if fit[LABEL].nunique() < 2:
        say("[warn] train split has no positives/negatives; using ALL rows "
            "for the feature checks")
        fit = df

    # ---- features: report only, drop only when explicitly asked ----
    keep = list(feats)

    const = [f for f in keep if fit[f].std() < 1e-12]
    if const:
        if a.drop_constant:
            say(f"\nfeatures dropped - constant (--drop-constant): {const}")
            keep = [f for f in keep if f not in const]
        else:
            say(f"\n[info] constant features (KEPT, use --drop-constant to remove): {const}")

    if df.stratum.nunique() > 1:
        sc = []
        for f in keep:
            for st, g in df.groupby("stratum"):
                if len(g) >= 30 and g[f].std() < 1e-12 and df[f].std() > 1e-12:
                    sc.append((f, st))
                    break
        if sc:
            names = [f for f, _ in sc]
            if a.drop_stratum_constant:
                say("\nfeatures dropped - constant inside one stratum "
                    "(--drop-stratum-constant):")
                for f, st in sc:
                    say(f"    {f}  (constant in stratum {st})")
                keep = [f for f in keep if f not in names]
            else:
                say("\n[info] features constant inside one stratum "
                    f"(KEPT, use --drop-stratum-constant to remove): {names}")

    if len(keep) > 1:
        thr = a.corr_threshold if a.corr_threshold <= 1.0 else 0.95
        corr = fit[keep].corr(method="spearman").abs()
        kept, redundant = [], []
        for f in keep:
            hit = next((k for k in kept if corr.loc[f, k] > thr), None)
            if hit is None:
                kept.append(f)
            else:
                redundant.append(f"{f} (~{hit}, r={corr.loc[f, hit]:.3f})")
        if redundant:
            if a.corr_threshold <= 1.0:
                say(f"\nfeatures dropped - redundant (|r| > {thr}):")
                for r in redundant:
                    say(f"    {r}")
                keep = kept
            else:
                say(f"\n[info] highly correlated features, |r| > {thr} "
                    "(ALL KEPT; use --corr-threshold 0.95 to remove):")
                for r in redundant:
                    say(f"    {r}")

    df = df[KEEP_META + keep].copy()

    if a.winsorize:
        lo = fit[keep].quantile(0.005)
        hi = fit[keep].quantile(0.995)
        df[keep] = df[keep].clip(lo, hi, axis=1)
        say("\nfeatures winsorized to train 0.5-99.5 percentile")

    # ---- report ----
    say(f"\nFINAL: {len(df)} rows, {len(keep)} features: {keep}")
    if missing:
        say(f"MISSING FROM INPUT (not in output): {missing}")
    say("\nlabel by split:")
    say(pd.crosstab(df.split, df[LABEL], margins=True).to_string())
    say("\nlabel by stratum:")
    say(pd.crosstab(df.stratum, df[LABEL], margins=True).to_string())
    say(f"\npatients: {df.patient_id.nunique()}   "
        f"patients with a positive window: "
        f"{df[df[LABEL] == 1].patient_id.nunique()}")
    for sp in ("train", "val", "test"):
        sub = df[df.split == sp]
        if len(sub) and sub[LABEL].sum() == 0:
            say(f"[warn] split '{sp}' has no positive windows after cleaning")

    df.to_csv(out / "train_ready.csv", index=False)
    (out / "cleaning_report.txt").write_text("\n".join(log))
    say(f"\nSaved train_ready.csv and cleaning_report.txt to {out.resolve()}")


if __name__ == "__main__":
    main()