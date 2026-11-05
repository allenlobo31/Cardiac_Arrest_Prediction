# pip install wfdb scikit-learn numpy matplotlib joblib
import os
import numpy as np, wfdb, joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, recall_score

WIN = 300                           # 5-minute HRV window
OFFSETS = np.arange(60, 1801, 120)  # window ends 1..30 min before VF onset
BEATS = set("NLRBAaJSVFejnE/fQ")    # beat symbols only
NAMES = ["mean_rr", "sdnn", "rmssd", "pnn50", "sd1", "sd2", "sd1_sd2"]

# VF onset (elapsed time) per SDDB record, from the dataset's index page.
# Records 40, 42, 49 have no VF and are left out.
ONSET = {"30": "07:54:33", "31": "13:42:24", "32": "16:45:18", "33": "04:46:19", "34": "06:35:44",
         "35": "24:34:56", "36": "18:59:01", "37": "01:31:13", "38": "08:01:54", "39": "04:37:51",
         "41": "02:59:24", "43": "15:37:11", "44": "19:38:45", "45": "18:09:17", "46": "03:41:47",
         "47": "06:13:01", "48": "02:29:40", "50": "11:45:43", "51": "22:58:23", "52": "02:32:40"}


def hms(s):
    h, m, x = map(int, s.split(":"))
    return h * 3600 + m * 60 + x


# ---------- 1. Download (skipped if already downloaded) ----------
for db in ["sddb", "nsrdb"]:
    d = f"data/{db}"
    if not (os.path.isdir(d) and os.listdir(d)):
        wfdb.dl_database(db, d)


def load(rec, db):
    h = wfdb.rdheader(f"data/{db}/{rec}")
    a = wfdb.rdann(f"data/{db}/{rec}", "ari" if db == "sddb" else "atr")
    t = np.array([s for s, c in zip(a.sample, a.symbol) if c in BEATS]) / h.fs
    return t, a, h.fs


def feats(t, start, end):
    rr = np.diff(t[(t > start) & (t <= end)])
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) < 150:
        return None
    rr = rr[np.abs(rr / np.median(rr) - 1) < 0.3]   # drop ectopic-like beats
    d = np.diff(rr)
    sd1 = np.std(d) / np.sqrt(2)
    sd2 = np.sqrt(max(2 * np.var(rr) - sd1 ** 2, 0))
    return [rr.mean(), rr.std(), np.sqrt(np.mean(d ** 2)),
            np.mean(np.abs(d) > 0.05), sd1, sd2, sd1 / (sd2 + 1e-9)]


# ---------- 2. Preprocess ----------
X, y, g, onsets = [], [], [], {}

for rec in wfdb.get_record_list("sddb"):           # cardiac arrest (pre-VF) windows
    if rec not in ONSET:
        continue                                    # 40, 42, 49 have no VF
    t, a, fs = load(rec, "sddb")
    onsets[rec] = hms(ONSET[rec])
    for o in OFFSETS:
        end = onsets[rec] - o
        f = feats(t, end - WIN, end) if end - WIN > 0 else None
        if f:
            X.append(f); y.append(1); g.append("sddb" + rec)

for rec in wfdb.get_record_list("nsrdb"):          # normal windows from the 2-hour mark
    t, a, fs = load(rec, "nsrdb")
    for end in 7200 + OFFSETS - OFFSETS[0]:
        f = feats(t, end - WIN, end)
        if f:
            X.append(f); y.append(0); g.append("nsrdb" + rec)

X, y, g = np.array(X), np.array(y), np.array(g)
print("windows:", len(y), "| arrest windows:", y.sum(),
      "| SDDB records used:", len(set(g[y == 1])), "| NSRDB records used:", len(set(g[y == 0])))


# ---------- 3. Plot ECG waveforms: normal vs pre-arrest vs arrest ----------
def ecg(db, rec, t0, t1):
    h = wfdb.rdheader(f"data/{db}/{rec}")
    t1 = min(t1, h.sig_len / h.fs)
    r = wfdb.rdrecord(f"data/{db}/{rec}", sampfrom=int(t0 * h.fs), sampto=int(t1 * h.fs), channels=[0])
    return np.arange(len(r.p_signal)) / h.fs + t0, r.p_signal[:, 0]


srec = next(iter(onsets)); on = onsets[srec]; nrec = wfdb.get_record_list("nsrdb")[0]
fig, ax = plt.subplots(3, 1, figsize=(12, 8))
x, s = ecg("nsrdb", nrec, 7200, 7210)
ax[0].plot(x, s, color="green"); ax[0].set_title(f"NORMAL sinus rhythm (NSRDB {nrec})")
x, s = ecg("sddb", srec, on - 300, on - 290)
ax[1].plot(x, s, color="darkorange"); ax[1].set_title(f"PRE-ARREST: 5 min before VF onset (SDDB {srec})")
x, s = ecg("sddb", srec, on - 30, on + 30)
ax[2].plot(x, s, color="black", lw=0.7)
ax[2].axvspan(on - 30, on, color="orange", alpha=0.25, label="pre-arrest")
ax[2].axvspan(on, x[-1], color="red", alpha=0.25, label="VF / cardiac arrest")
ax[2].axvline(on, color="red"); ax[2].legend(loc="upper left")
ax[2].set_title(f"Around VF onset (SDDB {srec})"); ax[2].set_xlabel("time (s)")
plt.tight_layout(); plt.savefig("ecg_waveforms.png", dpi=150)
try:
    plt.show()
except Exception:
    pass

# ---------- 4. Train with record-grouped CV, find best threshold ----------
clf = RandomForestClassifier(300, class_weight="balanced", random_state=0)
p = cross_val_predict(clf, X, y, groups=g, cv=GroupKFold(5), method="predict_proba")[:, 1]
fpr, tpr, thr = roc_curve(y, p)
best = float(thr[np.argmax(tpr - fpr)])             # Youden's J: max(sensitivity + specificity - 1)
pred = (p >= best).astype(int)
print(f"AUROC {roc_auc_score(y, p):.3f} | best threshold {best:.3f}")
print(f"at best threshold -> accuracy {accuracy_score(y, pred):.3f}, "
      f"sensitivity {recall_score(y, pred):.3f}, specificity {recall_score(y, pred, pos_label=0):.3f}")

# ---------- 5. Final model on all data, save ----------
clf.fit(X, y)
joblib.dump({"model": clf, "threshold": best, "features": NAMES}, "arrest_model.joblib")
print("saved arrest_model.joblib and ecg_waveforms.png")