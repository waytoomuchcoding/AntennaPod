"""Leave-one-episode-out classifier scores per transcript line (smoothed), saved as clfp/<ep>.json {line start: prob}."""
import json, os
import numpy as np
import adtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

eps = {}
for ep in adtest.EPISODES:
    dur, gt = adtest.load_gt(ep)
    lines = adtest.load_lines("moonshine", ep)
    lab = [any(a <= (l["t"] + l["e"]) / 2 <= b for a, b, t in gt if t == "ad") for l in lines]
    neu = [any(a <= (l["t"] + l["e"]) / 2 <= b for a, b, t in gt if t != "ad") for l in lines]
    eps[ep] = (lines, lab, neu)
os.makedirs("clfp", exist_ok=True)
for held in eps:
    cur, ctx, y, m = [], [], [], []
    for ep, (lines, lab, neu) in eps.items():
        if ep == held:
            continue
        for i, l in enumerate(lines):
            cur.append(l["text"]); ctx.append(" ".join(x["text"] for x in lines[max(0, i - 2):i + 3]))
            y.append(lab[i]); m.append(not neu[i])
    y, m = np.array(y), np.array(m)
    v1 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(cur)
    v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(ctx)
    clf = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(
        hstack([v1.transform(cur), v2.transform(ctx)]).tocsr()[m], y[m])
    lines = eps[held][0]
    c2 = [l["text"] for l in lines]
    x2 = [" ".join(x["text"] for x in lines[max(0, i - 2):i + 3]) for i in range(len(lines))]
    p = clf.predict_proba(hstack([v1.transform(c2), v2.transform(x2)]).tocsr())[:, 1]
    sm = np.convolve(p, np.ones(3) / 3, mode="same")
    json.dump({str(l["t"]): round(float(s), 4) for l, s in zip(lines, sm)}, open(f"clfp/{held}.json", "w"))
    print(held, "done")
