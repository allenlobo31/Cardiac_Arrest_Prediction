#!/usr/bin/env python3
"""
Show what annotation data each record really contains.

    python inspect_annotations.py ./data/sddb 3
    python inspect_annotations.py ./data/vfdb 3

Arguments: folder, and how many records to inspect (default 3).
Paste the output back so the labelling can be fixed.
"""
import sys
from collections import Counter
from pathlib import Path

import wfdb

folder = Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3
bases = sorted(p.with_suffix("") for p in folder.rglob("*.hea"))
print(f"{folder}: {len(bases)} records found\n")

for base in bases[:limit]:
    print("=" * 60)
    hdr = wfdb.rdheader(str(base))
    print(f"record {base.name}: fs={hdr.fs}, length={hdr.sig_len / hdr.fs / 60:.1f} min, "
          f"channels={hdr.sig_name}")
    if hdr.comments:
        print("header comments:", hdr.comments)
    exts = sorted({p.suffix for p in base.parent.glob(base.name + ".*")})
    print("files with this name:", exts)
    for ext in ("atr", "ari", "ecg", "qrs", "pu", "pu0", "pu1"):
        try:
            ann = wfdb.rdann(str(base), ext)
        except Exception:
            continue
        sym = Counter(ann.symbol)
        aux = Counter(a.replace("\x00", "").strip() for a in (ann.aux_note or []) if a.strip())
        print(f"\n  .{ext}: {len(ann.sample)} annotations")
        print("   symbols:", dict(sym.most_common(15)))
        print("   aux notes:", dict(aux.most_common(20)) or "none")
        rhythm = [(s / hdr.fs, a.replace('\x00', '').strip())
                  for s, sy, a in zip(ann.sample, ann.symbol, ann.aux_note or [])
                  if sy == "+"]
        if rhythm:
            print("   first rhythm changes (sec, token):", [(round(t), a) for t, a in rhythm[:8]])
            print("   last rhythm changes (sec, token):", [(round(t), a) for t, a in rhythm[-8:]])
print()