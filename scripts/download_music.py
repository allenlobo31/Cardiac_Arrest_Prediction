"""
scripts/download_music.py

Downloads a SMALL, USEFUL SAMPLE of the MUSIC database from PhysioNet into a folder that
scripts/extract_music.py can read directly (--music-dir).

Each Holter .dat file is ~100 MB and the full dataset is ~90 GB, so this picks N sudden-cardiac-death
(SCD) patients and N survivors (sinus rhythm only) using subject-info.csv, and downloads only those.
Only patients that actually have a Holter recording (listed in the dataset's RECORDS file) are chosen.

    python scripts/download_music.py --out ~/Datasets/music --per-class 5
    python scripts/extract_music.py --music-dir ~/Datasets/music

Disk use is roughly (2 x per-class x 100 MB): --per-class 5 is about 1 GB.
Already-downloaded files are skipped, so you can re-run later with a larger --per-class.
"""

import argparse
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from extract_music import (COLS, assign_label, find_col, find_id_column,  # same folder
                           read_table)

BASE_URL = "https://physionet.org/files/music-sudden-cardiac-death/1.0.1"


def download(url, dest, retries=3, quiet=False):
    """Downloads url -> dest via a .part file, so an interrupted download is never mistaken for a finished one."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (music-downloader)"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got, t0 = 0, time.time()
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total and not quiet and total > 5_000_000:
                        print(f"\r     {got / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
                if total and got != total:
                    raise IOError(f"incomplete download ({got} of {total} bytes)")
            part.replace(dest)
            if total and total > 5_000_000 and not quiet:
                print()
            return
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(2 * attempt)
    if part.exists():
        part.unlink()
    raise RuntimeError(f"{url}: {last}")


def get_available_holter_ids(out_dir, base_url):
    """Patient IDs that really have a Holter recording (936 of the 992 patients)."""
    path = out_dir / "RECORDS"
    if not path.exists():
        download(f"{base_url}/RECORDS", path, quiet=True)
    text = path.read_text(errors="ignore")
    ids = set(re.findall(r"Holter_ECG/(P\d{4})", text))
    if not ids:  # fallback if RECORDS uses a different layout
        ids = {m for m in re.findall(r"\b(P\d{4})\b", text)}
    return ids


def choose_patients(subject_info_path, per_class, seed, available_ids=None):
    df = read_table(subject_info_path)

    guess_ids = available_ids if available_ids else {f"P{i:04d}" for i in range(1, 2000)}
    id_col, to_rid = find_id_column(df, guess_ids)
    exit_col = find_col(df, *COLS["exit"])
    cause_col = find_col(df, *COLS["cause"])
    rhythm_col = find_col(df, *COLS["rhythm"])
    print(f"Columns used -> id: '{id_col}', exit: '{exit_col}', cause: '{cause_col}', rhythm: '{rhythm_col}'")

    d = pd.DataFrame({
        "rid": df.loc[:, id_col].map(to_rid),
        "exit": pd.to_numeric(df.loc[:, exit_col], errors="coerce"),
        "cause": pd.to_numeric(df.loc[:, cause_col], errors="coerce"),
        "rhythm": pd.to_numeric(df.loc[:, rhythm_col], errors="coerce"),
    }).drop_duplicates("rid")

    # Show the raw code counts so a wrong column is obvious immediately.
    print("  exit  values:", d["exit"].value_counts(dropna=False).sort_index().to_dict())
    print("  cause values:", d["cause"].value_counts(dropna=False).sort_index().to_dict())
    print("  rhythm values:", d["rhythm"].value_counts(dropna=False).sort_index().to_dict())

    d["label"] = [assign_label(c, e) for c, e in zip(d["cause"], d["exit"])]
    if available_ids is not None:
        d = d[d["rid"].isin(available_ids)]
    sinus = d[(d["rhythm"] == 0) & d["label"].notna()]
    scd = sinus[sinus["label"] == 1]
    surv = sinus[sinus["label"] == 0]
    print(f"Eligible patients with a Holter recording: {len(scd)} SCD, {len(surv)} survivors (sinus rhythm only)")
    if len(scd) == 0 or len(surv) == 0:
        raise SystemExit("One class has no eligible patients - check the code counts printed above.")

    scd_pick = scd.sample(n=min(per_class, len(scd)), random_state=seed)["rid"].tolist()
    surv_pick = surv.sample(n=min(per_class, len(surv)), random_state=seed)["rid"].tolist()
    return sorted(scd_pick), sorted(surv_pick)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="folder to create, e.g. ~/Datasets/music")
    ap.add_argument("--per-class", type=int, default=5, help="patients to download per class (SCD / survivor)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser()
    (out_dir / "Holter_ECG").mkdir(parents=True, exist_ok=True)

    print("Downloading metadata ...")
    for name in ("subject-info.csv", "subject-info_codes.csv"):
        if not (out_dir / name).exists():
            try:
                download(f"{args.base_url}/{name}", out_dir / name, quiet=True)
            except Exception as e:
                if name == "subject-info.csv":
                    raise
                print(f"  (optional file {name} skipped: {e})")

    available = get_available_holter_ids(out_dir, args.base_url)
    print(f"{len(available)} patients have a Holter recording.")

    scd, surv = choose_patients(out_dir / "subject-info.csv", args.per_class, args.seed, available)
    chosen = scd + surv
    print(f"\nSelected {len(scd)} SCD: {scd}\nSelected {len(surv)} survivors: {surv}")
    print(f"Downloading {len(chosen)} records (~{len(chosen) * 100} MB) ...")

    ok = 0
    for i, rid in enumerate(chosen, 1):
        dest_dir = out_dir / "Holter_ECG"
        if (dest_dir / f"{rid}.hea").exists() and (dest_dir / f"{rid}.dat").exists():
            print(f"[{i}/{len(chosen)}] {rid}: already downloaded")
            ok += 1
            continue
        print(f"[{i}/{len(chosen)}] {rid} ...")
        try:
            download(f"{args.base_url}/Holter_ECG/{rid}.hea", dest_dir / f"{rid}.hea")
            download(f"{args.base_url}/Holter_ECG/{rid}.dat", dest_dir / f"{rid}.dat")
            ok += 1
        except Exception as e:
            print(f"   FAILED {rid}: {e}")

    print(f"\nDone: {ok}/{len(chosen)} records available in {out_dir / 'Holter_ECG'}")
    if ok:
        print(f"Now run:\n  python scripts/extract_music.py --music-dir {out_dir}")


if __name__ == "__main__":
    main()