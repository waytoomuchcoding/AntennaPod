"""One evaluation for any per-line feature set or per-line score set (README 15):
  1. leave-one-show-out on the 13 human-labelled episodes: HMM knobs picked on dev, dev/test F1
  2. worst cases: labelled breaks with >=20 s of ad still playing (all 13 episodes)
  3. recall on the 29 unseen shows with automatic labels (gt_auto/), model trained on all 13 episodes
usage: GT_DIR=gt_v2 GROUP=show ./venv/bin/python eval_all.py feats <name>    (emb/<name>/<ep>.npy, logistic regression)
       GT_DIR=gt_v2 GROUP=show ./venv/bin/python eval_all.py scores <dir>    (<dir>/<ep>.json per-line probabilities,
                                                                              for all 13 + the 29 unseen episodes)
"""
import json, os, sys
import numpy as np
import adtest, seq_smooth
from emb_loo import eps, group_of
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BAD = {"x_casefiletruecrim", "x_revisionisthisto", "x_hiddenbrain", "x_lore", "x_acquired"}
UNSEEN = [f[:-5] for f in sorted(os.listdir(f"{adtest.ROOT}/gt_auto")) if f[:-5] not in BAD]
LINES = {e: adtest.load_lines("moonshine", e) for e in UNSEEN}


def auto_gt(e):
    g = json.load(open(f"{adtest.ROOT}/gt_auto/{e}.json"))
    return g["duration"], [(adtest.mmss(s["s"]), adtest.mmss(s["e"]), "ad") for s in g["segments"]]


def fit(X, y, m):
    sc = StandardScaler().fit(X[m])
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc.transform(X[m]), y[m])
    return lambda Z: clf.predict_proba(sc.transform(Z))[:, 1]


def worst(P, kw):
    n = 0
    for e in P:
        iv = adtest.to_intervals(eps[e]["lines"], list(seq_smooth.viterbi(P[e], **kw)), eps[e]["dur"])
        for a, b, t in eps[e]["gt"]:
            if t == "ad" and (b - a) - sum(adtest.overlap(p, (a, b)) for p in iv) >= 20:
                n += 1
    return n


def unseen_recall(Pu, kw):
    tp = gt = found = br = 0
    for e in UNSEEN:
        dur, g = auto_gt(e)
        s = adtest.score(adtest.to_intervals(LINES[e], list(seq_smooth.viterbi(Pu[e], **kw)), dur), g)
        tp += s["tp"]; gt += s["gt"]; found += s["found"]; br += s["breaks"]
    return tp / gt, f"{found}/{br}"


def run(name, P, Pu):
    fn, grid = seq_smooth.GRIDS["hmm"]
    kw = max(grid, key=lambda k: (seq_smooth.evaluate(P, eps, fn, "dev", **k)["f1"],
                                  -seq_smooth.evaluate(P, eps, fn, "dev", **k)["false_alarms"]))
    d, t = seq_smooth.evaluate(P, eps, fn, "dev", **kw), seq_smooth.evaluate(P, eps, fn, "test", **kw)
    r, f = unseen_recall(Pu, kw)
    print(f"{name:34s} dev F1 {d['f1']:.3f} ({d['false_alarms']} FA) | test F1 {t['f1']:.3f} ({t['breaks_found']}, "
          f"{t['false_alarms']} FA) | breaks with >=20 s ad left {worst(P, kw):2d}/54 | unseen recall {r:.3f} ({f})  {kw}",
          flush=True)


if __name__ == "__main__":
    kind, name = sys.argv[1], sys.argv[2]
    if kind == "feats":
        E = {e: np.load(f"emb/{name}/{e}.npy") for e in list(eps) + UNSEEN}
        P = {}
        for held in eps:
            tr = [e for e in eps if group_of(e) != group_of(held)]
            f = fit(np.vstack([E[e] for e in tr]), np.concatenate([eps[e]["y"] for e in tr]),
                    ~np.concatenate([eps[e]["neu"] for e in tr]))
            P[held] = f(E[held])
        f = fit(np.vstack([E[e] for e in eps]), np.concatenate([eps[e]["y"] for e in eps]),
                ~np.concatenate([eps[e]["neu"] for e in eps]))
        Pu = {e: f(E[e]) for e in UNSEEN}
    else:
        P = {e: np.array(json.load(open(f"{name}/{e}.json"))) for e in eps}
        Pu = {e: np.array(json.load(open(f"{name}/{e}.json"))) for e in UNSEEN}
    run(name, P, Pu)
