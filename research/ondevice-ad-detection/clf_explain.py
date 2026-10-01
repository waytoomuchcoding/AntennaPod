import json, pickle, numpy as np, adtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
held = "daily2"
cur, ctx, y, m = [], [], [], []
for ep in adtest.EPISODES:
    if ep == held: continue
    dur, gt = adtest.load_gt(ep); lines = adtest.load_lines("moonshine", ep)
    for i, l in enumerate(lines):
        mid = (l["t"] + l["e"]) / 2
        cur.append(l["text"]); ctx.append(" ".join(x["text"] for x in lines[max(0, i-2):i+3]))
        y.append(any(a <= mid <= b for a, b, t in gt if t == "ad")); m.append(not any(a <= mid <= b for a, b, t in gt if t != "ad"))
y, m = np.array(y), np.array(m)
v1 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(cur)
v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(ctx)
X = hstack([v1.transform(cur), v2.transform(ctx)]).tocsr()
clf = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(X[m], y[m])
print("training lines", int(m.sum()), "ad lines", int(y[m].sum()), "features", X.shape[1])
blob = pickle.dumps((v1, v2, clf)); print("pickled size KB", len(blob) // 1024)
names = np.array(["line:" + f for f in v1.get_feature_names_out()] + ["context:" + f for f in v2.get_feature_names_out()])
order = np.argsort(clf.coef_[0])
print("TOP AD", [(names[i], round(clf.coef_[0][i], 2)) for i in order[-12:][::-1]])
print("TOP CONTENT", [(names[i], round(clf.coef_[0][i], 2)) for i in order[:8]])
lines = adtest.load_lines("moonshine", held); dur, gt = adtest.load_gt(held)
c2 = [l["text"] for l in lines]; x2 = [" ".join(x["text"] for x in lines[max(0, i-2):i+3]) for i in range(len(lines))]
p = clf.predict_proba(hstack([v1.transform(c2), v2.transform(x2)]).tocsr())[:, 1]
sm = np.convolve(p, np.ones(3) / 3, mode="same")
for lo, hi in ((780, 935), (120, 160)):
    print("----")
    for i, l in enumerate(lines):
        if lo <= l["t"] <= hi:
            lab = "AD " if any(a <= (l["t"]+l["e"])/2 <= b for a, b, t in gt if t == "ad") else "   "
            print(f"{adtest.fmt(l['t'])} {lab} p={p[i]:.2f} smooth={sm[i]:.2f} {'>' if sm[i] > 0.15 else ' '} {l['text'][:75]}")
