"""Baseline line classifier: TF-IDF of each line plus its neighbours -> logistic regression.
Trained on dev episodes, evaluated on held-out with the same interval scoring as the LLM runs."""
import json, sys
import numpy as np
import adtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack


def data(split):
    X, y, eps = [], [], []
    for ep, m in adtest.EPISODES.items():
        if m.get("split", "dev") != split:
            continue
        dur, gt = adtest.load_gt(ep)
        lines = adtest.load_lines("moonshine", ep)
        rows = []
        for i, l in enumerate(lines):
            mid = (l["t"] + l["e"]) / 2
            lab = any(a <= mid <= b for a, b, t in gt if t == "ad")
            neutral = any(a <= mid <= b for a, b, t in gt if t != "ad")
            rows.append((l["text"], lab, neutral))
        eps.append((ep, lines, rows, dur, gt))
    return eps


def featurize(vec, vec_ctx, eps, fit=False):
    cur, ctx, y, idx = [], [], [], []
    for ep, lines, rows, dur, gt in eps:
        for i, (t, lab, neu) in enumerate(rows):
            cur.append(t)
            ctx.append(" ".join(r[0] for r in rows[max(0, i - 2):i + 3]))
            y.append(lab)
            idx.append((ep, i, neu))
    if fit:
        vec.fit(cur)
        vec_ctx.fit(ctx)
    return hstack([vec.transform(cur), vec_ctx.transform(ctx)]).tocsr(), np.array(y), idx


train, test = data("dev"), data("test")
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
vec_ctx = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr, ytr, itr = featurize(vec, vec_ctx, train, fit=True)
mask = np.array([not n for _, _, n in itr])
clf = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(Xtr[mask], ytr[mask])
Xte, yte, ite = featurize(vec, vec_ctx, test)
p = clf.predict_proba(Xte)[:, 1]
for thr in (0.15, 0.25, 0.35):
    scores, k = [], 0
    for ep, lines, rows, dur, gt in test:
        n = len(rows)
        probs = p[k:k + n]
        k += n
        sm = np.convolve(probs, np.ones(3) / 3, mode="same")
        pred = adtest.to_intervals(lines, list(sm > thr), dur)
        scores.append(adtest.score(pred, gt))
    print(f"threshold {thr}: held-out", json.dumps(adtest.summarize(scores)))
names = np.array(list(vec.get_feature_names_out()) + ["ctx:" + f for f in vec_ctx.get_feature_names_out()])
top = np.argsort(clf.coef_[0])
print("most ad-like features:", ", ".join(names[top[-25:]][::-1]))
print("most content-like features:", ", ".join(names[top[:15]]))

import os
out = f"{adtest.ROOT}/results/clf_tfidf_test"
os.makedirs(out, exist_ok=True)
k = 0
for ep, lines, rows, dur, gt in test:
    n = len(rows)
    sm = np.convolve(p[k:k + n], np.ones(3) / 3, mode="same")
    k += n
    pred = adtest.to_intervals(lines, list(sm > 0.15), dur)
    json.dump({"episode": ep, "pred": [[adtest.fmt(s), adtest.fmt(e)] for s, e in pred],
               "stats": {"calls": 0, "in_tokens": 0, "out_tokens": 0}, "audio_s": dur},
              open(f"{out}/{ep}.json", "w"))
