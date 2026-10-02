"""Refine an LLM run's interval edges with a line classifier: at each edge, extend outwards while the next line's
score is above `ext`, or trim inwards while the edge line's score is below `trim` (at most K lines either way).
Knobs picked on dev; reports dev/test F1, ad seconds left per break, and boundary errors.
usage: GT_DIR=gt_v2 GROUP=show ./venv/bin/python edge_refine.py RUN SCORES [--write]
  SCORES = a name under embp_<GT>_<GROUP>/ (held-out per-line probabilities from emb_loo.py)
  --write saves the refined run as results/<RUN>__edges_<SCORES>/ (for boundary_err.py)"""
import itertools, json, os, sys
import numpy as np
import adtest, emb_loo
from emb_loo import eps

RUN, NAME = sys.argv[1], sys.argv[2]
P = {e: np.convolve(json.load(open(f"embp_{emb_loo.GT}_{emb_loo.GROUP}/{NAME}/{e}.json")), np.ones(3) / 3, "same")
     for e in eps}
PRED = {e: [(adtest.mmss(s), adtest.mmss(t)) for s, t in json.load(open(f"results/{RUN}/{e}.json"))["pred"]] for e in eps}


def refine(e, ext, trim, K):
    L, p, out = eps[e]["lines"], P[e], []
    mid = np.array([(l["t"] + l["e"]) / 2 for l in L])
    for s, t in PRED[e]:
        idx = np.where((mid >= s) & (mid <= t))[0]
        if not len(idx):
            out.append((s, t)); continue
        a, b = idx[0], idx[-1]
        for _ in range(K):                                   # start
            if a > 0 and p[a - 1] > ext: a -= 1
            elif a < b and p[a] < trim: a += 1
            else: break
        for _ in range(K):                                   # end
            if b + 1 < len(L) and p[b + 1] > ext: b += 1
            elif b > a and p[b] < trim: b -= 1
            else: break
        # keep the LLM's own edge when it lies inside the edge line (sub-line precision)
        ns = s if a == idx[0] else L[a]["t"]
        nt = t if b == idx[-1] else (L[b + 1]["t"] if b + 1 < len(L) else eps[e]["dur"])
        out.append((ns, nt))
    return out


def ev(split, **kw):
    sc = [adtest.score(refine(e, **kw), eps[e]["gt"]) for e in eps if eps[e]["split"] == split]
    s = adtest.summarize(sc)
    s["ad_s_left_per_break"] = round(sum(x["gt"] - x["tp"] for x in sc) / sum(x["breaks"] for x in sc), 1)
    return s


grid = [dict(ext=x, trim=t, K=k) for x, t, k in itertools.product((0.5, 0.7, 0.9, 1.1), (-1, 0.05, 0.1, 0.2), (1, 2, 3))]
base = dict(ext=1.1, trim=-1, K=1)                           # no change
best = max(grid, key=lambda kw: ev("dev", **kw)["f1"])
print(f"== {RUN[:50]} edges by {NAME}")
print(f"  unchanged  dev {ev('dev', **base)}\n             test {ev('test', **base)}")
print(f"  refined {best}\n             dev {ev('dev', **best)}\n             test {ev('test', **best)}")
if "--write" in sys.argv:
    out = f"results/{RUN}__edges_{NAME}"
    os.makedirs(out, exist_ok=True)
    for e in eps:
        iv = refine(e, **best)
        json.dump({"episode": e, "pred": [[adtest.fmt(s), adtest.fmt(t)] for s, t in iv],
                   "score": adtest.score(iv, eps[e]["gt"])}, open(f"{out}/{e}.json", "w"), indent=1)
