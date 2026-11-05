import wfdb, glob, os
for f in sorted(glob.glob("data/sddb/*.atr") + glob.glob("data/sddb/*.ari")):
    base, ext = os.path.splitext(f)
    a = wfdb.rdann(base, ext[1:])
    vf = [(n, s) for n, s in zip(a.aux_note, a.sample) if n.startswith("(")]
    print(os.path.basename(f), "| rhythm notes:", sorted({n for n, _ in vf}), "| VF:", sum(n.startswith("(VF") for n, _ in vf))