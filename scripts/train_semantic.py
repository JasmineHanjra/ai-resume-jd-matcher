# ─────────────────────────────────────────────────────────────────────────────
# Project: AI Resume ↔ JD Matcher
# File: scripts/train_semantic.py
# Author: Jasmine Kaur Hanjra
# Created: Aug 2025
# Purpose:
#   Train logistic classifier on features (TF-IDF + SBERT + overlap/length).
#   Evaluate with GroupKFold by resume and by JD to prevent leakage.
# Artifacts:
#   - data/matcher_model.joblib
#   - data/threshold.json
#   - data/model_metrics.json
# Future work:
#   - Add calibration curve & temperature scaling if needed.
#   - Save confusion matrices and per-category metrics.
# ─────────────────────────────────────────────────────────────────────────────

import json, pathlib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from featurize_shared import build_features, clean_text

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

pairs = pd.read_csv(DATA / "pairs.csv")
pairs["resume_text"] = pairs["resume_text"].astype(str).map(clean_text)
pairs["jd_text"]     = pairs["jd_text"].astype(str).map(clean_text)
y = pairs["label"].astype(int).values

# groups for leak-proof CV
g_resume = pairs["resume_text"].factorize()[0]
g_jd     = pairs["jd_text"].factorize()[0]

def make_X(df):
    rows = []
    for _, r in df.iterrows():
        X, *_ = build_features(r["resume_text"], r["jd_text"])
        rows.append(X.iloc[0].to_dict())
    return pd.DataFrame(rows)

X = make_X(pairs)

pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),   # features are non-negative; with_mean=False for sparse-compat
    ("clf", LogisticRegression(C=5.0, max_iter=500))
])

def eval_grouped(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    probs, ys = [], []
    aucs, accs, f1s, thrs = [], [], [], []
    for tr, va in gkf.split(X, y, groups):
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[va])[:,1]
        probs.append(p); ys.append(y[va])
        # choose threshold that maximizes F1 on the fold
        ts = np.linspace(0.05, 0.95, 19)
        fbest, tbest = -1, 0.5
        for t in ts:
            pred = (p >= t).astype(int)
            f = f1_score(y[va], pred)
            if f > fbest: fbest, tbest = f, t
        thrs.append(tbest)
        aucs.append(roc_auc_score(y[va], p))
        accs.append(accuracy_score(y[va], (p >= tbest).astype(int)))
        f1s.append(fbest)
    return {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "acc_mean": float(np.mean(accs)), "f1_mean": float(np.mean(f1s)),
        "threshold_mean": float(np.mean(thrs))
    }

m_r = eval_grouped(X, y, g_resume, n_splits=5)
m_j = eval_grouped(X, y, g_jd,     n_splits=5)

# final fit on all data
pipe.fit(X, y)

# choose final threshold on all data (max F1)
p_all = pipe.predict_proba(X)[:,1]
ts = np.linspace(0.05, 0.95, 19)
fbest, tbest = -1, 0.5
for t in ts:
    f = f1_score(y, (p_all >= t).astype(int))
    if f > fbest: fbest, tbest = f, t

import joblib, json
joblib.dump(pipe, DATA / "matcher_model.joblib")
(DATA / "threshold.json").write_text(json.dumps({"threshold": float(tbest)}))
(DATA / "model_metrics.json").write_text(json.dumps({
    "grouped_by_resume": m_r, "grouped_by_jd": m_j, "final_threshold": float(tbest)
}, indent=2))

print("[OK] Saved matcher_model.joblib, threshold.json")
print("Resume groups:", m_r)
print("JD groups:", m_j)
print("Final threshold:", tbest)
