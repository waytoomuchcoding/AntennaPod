"""Leave-one-episode-out over all 13 episodes: train the TF-IDF classifier on the other 12, predict the held-out one.
Then use it as a veto on the LLM's copy+verify predictions (dev and test runs)."""
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
    eps[ep] = (lines, lab, neu, dur, gt)


def feats(names):
    cur, ctx, y, m = [], [], [], []
    for ep in names:
        lines, lab, neu, _, _ = eps[ep]
        for i, l in enumerate(lines):
            cur.append(l["text"])
            ctx.append(" ".join(x["text"] for x in lines[max(0, i - 2):i + 3]))
            y.append(lab[i])
            m.append(not neu[i])
    return cur, ctx, np.array(y), np.array(m)


clf_pred = {}
for held in eps:
    train = [e for e in eps if e != held]
    cur, ctx, y, m = feats(train)
    v1 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(cur)
    v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(ctx)
    X = hstack([v1.transform(cur), v2.transform(ctx)]).tocsr()
    clf = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(X[m], y[m])
    c2, x2, _, _ = feats([held])
    p = clf.predict_proba(hstack([v1.transform(c2), v2.transform(x2)]).tocsr())[:, 1]
    sm = np.convolve(p, np.ones(3) / 3, mode="same")
    lines, _, _, dur, _ = eps[held]
    clf_pred[held] = adtest.to_intervals(lines, list(sm > 0.15), dur)

for split, run in (("dev", "gemma-4-E2B-it.litertlm__moonshine__quotes_copy6_verified"),
                   ("test", "gemma-4-E2B-it.litertlm__moonshine__quotes_copy6_test_verified"),
                   ("dev", "gemma-4-E4B-it.litertlm__moonshine__quotes_copy6_verified"),
                   ("test", "gemma-4-E4B-it.litertlm__moonshine__quotes_copy6_test_verified")):
    base, veto, alone = [], [], []
    for f in sorted(os.listdir(f"results/{run}")):
        ep = f[:-5]
        _, gt = adtest.load_gt(ep)
        a = [(adtest.mmss(s), adtest.mmss(e)) for s, e in json.load(open(f"results/{run}/{f}"))["pred"]]
        b = clf_pred[ep]
        base.append(adtest.score(a, gt))
        veto.append(adtest.score([x for x in a if any(adtest.overlap(x, q) > 0 for q in b)], gt))
        alone.append(adtest.score(b, gt))
    print(f"{run.split('__')[0]:24s} {split:4s} LLM {adtest.summarize(base)}\n{'':29s} +veto {adtest.summarize(veto)}\n{'':29s} clf only {adtest.summarize(alone)}")
