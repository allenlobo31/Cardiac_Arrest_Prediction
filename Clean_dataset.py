#!/usr/bin/env python3
"""
clean_dataset.py
================
Input : prediction_features.csv  (output of preprocess_cardiac_arrest_prediction.py)
Output: train_ready.csv + cleaning_report.txt

Removes what a model should NOT see or does not need.

COLUMNS removed
  * start_sec, end_sec, time_to_onset_sec, episode_id
        position / answer-leaking metadata, not features
  * features that are constant (no information)
  * features that are constant inside any one stratum but not overall
        (e.g. ectopy counts that are always 0 for one source: the model could
         learn the source instead of the physiology)
  * redundant features (|Spearman r| > --corr-threshold with an earlier one)

ROWS removed
  * NaN / inf feature values (e.g. LF/HF when HF power is 0)
  * physiologically implausible windows (mean HR outside --hr-min..--hr-max)
  * exact duplicate feature rows

COLUMNS kept (not features, never feed them to the model)
  orig_row   row number in the original CSV / npz   (to fetch the raw ECG)
  patient_id use for grouped cross-validation
  stratum    used by balance_dataset.py and per-stratum evaluation
  split      train / val / test
  label_prearrest   the target

Statistics (constant / correlation / winsorizing) are fitted on the TRAIN
split only, so nothing leaks from validation or test.

Usage
    python clean_dataset.py --csv processed_pred/prediction_features.csv \
                            --out processed_pred/clean
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "n_beats", "mean_rr", "mean_hr", "sdnn", "rmssd", "sdsd", "pnn50", "cv_rr",
    "min_rr", "max_rr", "rr_range", "sd1", "sd2", "lf", "hf", "lf_hf",
    "n_ventricular", "pvc_burden", "max_consec_ventricular", "non_n_fraction",
    "premature_fraction",
]
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
    ap.add_argument("--corr-threshold", type=float, default=0.95,
                    help="drop a feature if |Spearman r| with a kept one is above this "
                         "(use 1.01 to disable)")
    ap.add_argument("--keep-stratum-constant", action="store_true",
                    help="do NOT drop features that are constant within a stratum")
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
    feats = [f for f in FEATURES if f in df.columns]
    say(f"loaded {len(df)} rows, {len(feats)} features, "
        f"{int(df[LABEL].sum())} positives")

    # ---- columns: drop metadata that must not be used as features ----
    dropped_meta = [c for c in DROP_META if c in df.columns]
    if "split" not in df.columns:
        df["split"] = "train"
    if "stratum" not in df.columns:
        df["stratum"] = "A"
    if "patient_id" not in df.columns:
        df["patient_id"] = df["group"] if "group" in df.columns else "P000"
    say(f"\ncolumns dropped (metadata): {dropped_meta}")
    df = df[KEEP_META + feats]

    # ---- rows ----
    n0 = len(df)
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    bad = df[feats].isna().any(axis=1)
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
            "for fitting the column rules")
        fit = df

    # ---- columns: feature selection ----
    keep = list(feats)

    const = [f for f in keep if fit[f].std() < 1e-12]
    if const:
        say(f"\nfeatures dropped - constant: {const}")
        keep = [f for f in keep if f not in const]

    if not a.keep_stratum_constant and df.stratum.nunique() > 1:
        sc = []
        for f in keep:
            for st, g in df.groupby("stratum"):
                if len(g) >= 30 and g[f].std() < 1e-12 and df[f].std() > 1e-12:
                    sc.append((f, st))
                    break
        if sc:
            say("\nfeatures dropped - constant inside one stratum "
                "(model could learn the source instead):")
            for f, st in sc:
                say(f"    {f}  (constant in stratum {st})")
            keep = [f for f in keep if f not in {f for f, _ in sc}]

    if a.corr_threshold <= 1.0 and len(keep) > 1:
        corr = fit[keep].corr(method="spearman").abs()
        kept, redundant = [], []
        for f in keep:
            hit = next((k for k in kept if corr.loc[f, k] > a.corr_threshold), None)
            if hit is None:
                kept.append(f)
            else:
                redundant.append(f"{f} (~{hit}, r={corr.loc[f, hit]:.3f})")
        if redundant:
            say(f"\nfeatures dropped - redundant (|r| > {a.corr_threshold}):")
            for r in redundant:
                say(f"    {r}")
        keep = kept

    df = df[KEEP_META + keep].copy()

    if a.winsorize:
        lo = fit[keep].quantile(0.005)
        hi = fit[keep].quantile(0.995)
        df[keep] = df[keep].clip(lo, hi, axis=1)
        say("\nfeatures winsorized to train 0.5-99.5 percentile")

    # ---- report ----
    say(f"\nFINAL: {len(df)} rows, {len(keep)} features: {keep}")
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