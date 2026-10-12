"""
scripts/music_extract.py

Extracts HRV feature windows from the MUSIC database (24 h Holter, local files) and caches them in
data/raw/music.csv - same folder and same 98-column layout as nsrdb_features_cache.csv and
sddb_features_cache.csv. MUSIC only; no SDDB/NSRDB rows.

Needs scripts/build_dataset.py alongside (it holds the shared extraction/QC/labelling code).

    python scripts/music_extract.py --music-dir /path/to/music-sudden-cardiac-death/1.0.1 --limit 5   # trial
    python scripts/music_extract.py --music-dir /path/to/music-sudden-cardiac-death/1.0.1             # full

Label (per PATIENT, copied to that patient's windows): 1 = sudden cardiac death, 0 = survivor with
full follow-up. Other deaths, transplants, lost-to-follow-up and non-sinus Holters are excluded;
data/raw/music_manifest.csv records every exclusion and why.

Resumable: patients already in data/raw/music.csv are skipped. Delete music.csv (and the manifest)
before re-running with a different --channel or --max-windows-per-patient.
"""
import argparse
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "music.csv"   # <- MUSIC output lives here, always
bd.MUSIC_CACHE = OUTPUT_FILE                             # force the shared code to use this exact path


def run_extraction(args):
    if not hasattr(bd, "append_music"):
        raise bd.DatasetError("scripts/build_dataset.py is an older version (no schema-locked append). "
                              "Replace it with the latest build_dataset.py, then re-run.")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    bd.cmd_music(Namespace(music_dir=args.music_dir, limit=args.limit, channel=args.channel,
                           max_windows_per_patient=args.max_windows_per_patient,
                           include_nonsinus=args.include_nonsinus))
    if not OUTPUT_FILE.exists():
        raise bd.DatasetError("no MUSIC rows were produced - see data/raw/music_manifest.csv for why")

    df = pd.read_csv(OUTPUT_FILE)
    problems = []
    per = df.groupby("record_id")["label"].nunique()
    if (per > 1).any():
        problems.append(f"label varies within patient(s): {list(per[per > 1].index[:5])}")
    if df.duplicated(["record_id", "window_end_sec"]).any():
        problems.append("duplicate (record_id, window_end_sec) rows")
    if bd.SDDB_CACHE.exists():
        ref = list(pd.read_csv(bd.SDDB_CACHE, nrows=0).columns)
        schema_ok = list(df.columns) == ref
        print(f"schema identical to {bd.SDDB_CACHE.name}: {schema_ok}")
        if not schema_ok:
            problems.append("music.csv columns differ from sddb_features_cache.csv")
    if problems:
        raise bd.DatasetError("; ".join(problems))

    pat = df.groupby("record_id")["label"].first()
    print(f"\nSaved {OUTPUT_FILE}  shape={df.shape}")
    print(f"patients: {len(pat)}  SCD={int(pat.sum())}  survivors={int((pat == 0).sum())}")
    if pat.nunique() < 2:
        print("WARNING: only one class present - you need both SCD and survivors to train.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--music-dir", required=True, help="folder containing subject-info.csv and Holter_ECG/")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N recordings")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--max-windows-per-patient", type=int, default=48)
    ap.add_argument("--include-nonsinus", action="store_true")
    args = ap.parse_args()
    try:
        run_extraction(args)
    except bd.DatasetError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()