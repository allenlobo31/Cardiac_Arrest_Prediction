"""Feature + risk-state helpers. Shipped next to the saved model so training and inference use identical code."""
import numpy as np, pandas as pd


def _hist_stats(d, f, group_col, mode, min_history):
    g = d.groupby(group_col, sort=False)[f]
    if mode == "expanding":     # baseline = only windows BEFORE the current one (causal)
        mu = g.transform(lambda x: x.expanding(min_periods=min_history).mean().shift(1))
        sd = g.transform(lambda x: x.expanding(min_periods=min_history).std().shift(1))
    else:                       # baseline = whole patient (non-causal, diagnostic only)
        mu = g.transform("mean")
        sd = g.transform("std")
    return mu, sd


def build_features(df, base_feats, group_col="patient_id", order_col="orig_row", feature_set="relative",
                   baseline_mode="expanding", min_history=5, short_win=10):
    """Returns a feature frame aligned to df.index.
       absolute: the raw HRV values
       relative: per patient -> <f>_z     z-score of the current window vs. the patient's own baseline
                                <f>_zmean rolling mean of that z-score over the last `short_win` windows
                                <f>_zstd  rolling std of that z-score (instability)"""
    d = df.reset_index(drop=True)
    d = d.sort_values([group_col, order_col], kind="mergesort")
    out = pd.DataFrame(index=d.index)
    if feature_set in ("absolute", "both"):
        for f in base_feats:
            out[f] = d[f].astype(float)
    if feature_set in ("relative", "both"):
        grp = d[group_col]
        for f in base_feats:
            x = d[f].astype(float)
            mu, sd = _hist_stats(d.assign(**{f: x}), f, group_col, baseline_mode, min_history)
            sd_eff = np.maximum(sd.fillna(0.0), 0.05 * x.std() + 1e-12)      # floor: no blow-up when history is flat
            z = ((x - mu) / sd_eff).clip(-5, 5).fillna(0.0)                  # no baseline yet -> neutral 0
            out[f + "_z"] = z
            out[f + "_zmean"] = z.groupby(grp).transform(lambda s: s.rolling(short_win, min_periods=1).mean())
            out[f + "_zstd"] = z.groupby(grp).transform(lambda s: s.rolling(short_win, min_periods=3).std()).fillna(0.0)
    res = out.reindex(range(len(d)))
    res.index = df.index
    return res


def smooth_by_patient(p, patient, order, k):
    """Causal rolling median of the probability over the last k windows of the same patient."""
    d = pd.DataFrame({"p": np.asarray(p, dtype=float), "g": np.asarray(patient), "o": np.asarray(order)})
    d["pos"] = np.arange(len(d))
    d = d.sort_values(["g", "o", "pos"], kind="mergesort")
    d["ps"] = d.groupby("g", sort=False)["p"].transform(lambda s: s.rolling(max(int(k), 1), min_periods=1).median())
    return d.sort_values("pos")["ps"].values


def risk_state(p, t_watch, t_warning):
    p = np.asarray(p)
    return np.where(p >= t_warning, "PRE-ARREST WARNING", np.where(p >= t_watch, "WATCH", "STABLE"))
