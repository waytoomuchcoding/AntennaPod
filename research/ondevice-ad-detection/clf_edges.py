"""Leave-one-episode-out TF-IDF line scores for the current labels ($GT_DIR) and their use on a saved LLM run:
veto (drop an interval whose best line score is below `veto`) and edge trim (drop leading/trailing lines of an
interval while their score is below `trim`). Thresholds are picked on the dev split and reported on test."""
import itertools, json, os, sys
import numpy as np
import adtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

GT = os.environ.get("GT_DIR", "gt")
OUT = f"{adtest.ROOT}/clfp_{GT}"


def loo_scores():
    eps = {}
    for ep in adtest.EPISODES:
        dur, gt = adtest.load_gt(ep)
        lines = adtest.load_lines("moonshine", ep)
        lab = [any(a <= (l["t"] + l["e"]) / 2 <= b for a, b, t in gt if t == "ad") for l in lines]
        neu = [any(a <= (l["t"] + l["e"]) / 2 <= b for a, b, t in gt if t != "ad") for l in lines]
        eps[ep] = (lines, lab, neu)
    os.makedirs(OUT, exist_ok=True)
    ctx = lambda lines, i: " ".join(x["text"] for x in lines[max(0, i - 2):i + 3])
    for held in eps:
        cur, cx, y, m = [], [], [], []
        for ep, (lines, lab, neu) in eps.items():
            if ep == held:
                continue
            for i, l in enumerate(lines):
                cur.append(l["text"]); cx.append(ctx(lines, i)); y.append(lab[i]); m.append(not neu[i])
        y, m = np.array(y), np.array(m)
        v1 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(cur)
        v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(cx)
        clf = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(
            hstack([v1.transform(cur), v2.transform(cx)]).tocsr()[m], y[m])
        lines = eps[held][0]
        p = clf.predict_proba(hstack([v1.transform([l["text"] for l in lines]),
                                      v2.transform([ctx(lines, i) for i in range(len(lines))])]).tocsr())[:, 1]
        sm = np.convolve(p, np.ones(3) / 3, mode="same")
        json.dump([round(float(s), 4) for s in sm], open(f"{OUT}/{held}.json", "w"))


def apply(ep, pred, veto, trim, grow=1.1):
    lines = adtest.load_lines("moonshine", ep)
    probs = json.load(open(f"{OUT}/{ep}.json"))
    dur, _ = adtest.load_gt(ep)
    out = []
    for s, e in pred:
        idx = [i for i, l in enumerate(lines) if l["t"] >= s - 0.5 and l["t"] < e]
        if not idx:
            continue
        if max(probs[i] for i in idx) < veto:
            continue
        while idx and probs[idx[0]] < trim:
            idx.pop(0)
        while idx and probs[idx[-1]] < trim:
            idx.pop()
        while idx and idx[0] > 0 and probs[idx[0] - 1] >= grow:
            idx.insert(0, idx[0] - 1)
        while idx and idx[-1] + 1 < len(lines) and probs[idx[-1] + 1] >= grow:
            idx.append(idx[-1] + 1)
        if not idx:
            continue
        a = lines[idx[0]]["t"]
        b = lines[idx[-1] + 1]["t"] if idx[-1] + 1 < len(lines) else dur
        if b - a >= 10:
            out.append((a, b))
    return out


def evaluate(run, eps, veto, trim, grow=1.1):
    scores = []
    for ep in eps:
        _, gt = adtest.load_gt(ep)
        r = json.load(open(f"{adtest.ROOT}/results/{run}/{ep}.json"))
        pred = [(adtest.mmss(s), adtest.mmss(e)) for s, e in r["pred"]]
        scores.append(adtest.score(apply(ep, pred, veto, trim, grow), gt))
    return adtest.summarize(scores)


if __name__ == "__main__":
    if not os.path.exists(f"{OUT}/{list(adtest.EPISODES)[-1]}.json") or "--refit" in sys.argv:
        loo_scores()
    run = sys.argv[1]
    have = [f[:-5] for f in os.listdir(f"{adtest.ROOT}/results/{run}")]
    dev = [e for e in have if adtest.EPISODES[e].get("split", "dev") == "dev"]
    test = [e for e in have if adtest.EPISODES[e].get("split") == "test"]
    grid = list(itertools.product([0, 0.1, 0.15, 0.2, 0.3], [0, 0.02, 0.05, 0.1, 0.15, 0.2],
                                  [1.1, 0.9, 0.8, 0.7, 0.6, 0.5]))
    best = max(grid, key=lambda g: evaluate(run, dev, *g)["f1"])
    print("base   dev", evaluate(run, dev, 0, 0), "\n       test", evaluate(run, test, 0, 0),
          "\n       all ", evaluate(run, have, 0, 0))
    print(f"best on dev: veto={best[0]} trim={best[1]} grow={best[2]}")
    print("tuned  dev", evaluate(run, dev, *best), "\n       test", evaluate(run, test, *best),
          "\n       all ", evaluate(run, have, *best))
