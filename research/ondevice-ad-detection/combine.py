"""Combine saved predictions of two runs offline (union / quotes-anchored) and score."""
import json, os, sys
import adtest


def preds(run, ep):
    r = json.load(open(f"{adtest.ROOT}/results/{run}/{ep}.json"))
    return [(adtest.mmss(s), adtest.mmss(e)) for s, e in r["pred"]]


def union(a, b, gap):
    return [tuple(x) for x in adtest.merge(a + b, gap)]


def anchored(blocks, quotes, gap):
    """Block intervals give recall; where a quote interval overlaps a block interval, snap the merged interval's
    outer edges to the quote's edges when the quote extends past it or lies within 30 s of the block edge."""
    out = []
    for s, e in adtest.merge(blocks + quotes, gap):
        qs = [q for q in quotes if adtest.overlap(q, (s, e)) > 0]
        bs = [b for b in blocks if adtest.overlap(b, (s, e)) > 0]
        if qs and bs:
            qmin, qmax = min(q[0] for q in qs), max(q[1] for q in qs)
            bmin, bmax = min(b[0] for b in bs), max(b[1] for b in bs)
            s = qmin if qmin <= bmin + 30 else bmin
            e = qmax if qmax >= bmax - 30 else bmax
        out.append((s, e))
    return out


def confirmed(blocks, quotes, gap, min_len=60):
    keep = [b for b in blocks if b[1] - b[0] >= min_len or any(adtest.overlap(b, q) > 0 for q in quotes)]
    return anchored(keep, quotes, gap)


if __name__ == "__main__":
    ra, rb = sys.argv[1], sys.argv[2]
    eps = sorted(f[:-5] for f in os.listdir(f"{adtest.ROOT}/results/{ra}") if os.path.exists(f"{adtest.ROOT}/results/{rb}/{f}"))
    for name, fn in [("A", lambda a, b: a), ("B", lambda a, b: b), ("union", lambda a, b: union(a, b, 20)),
                     ("union45", lambda a, b: union(a, b, 45)), ("anchored", lambda a, b: anchored(a, b, 20)),
                     ("confirm45", lambda a, b: confirmed(a, b, 20, 45)), ("confirm60", lambda a, b: confirmed(a, b, 20, 60)),
                     ("confirm90", lambda a, b: confirmed(a, b, 20, 90))]:
        scores = []
        for ep in eps:
            dur, gt = adtest.load_gt(ep)
            scores.append(adtest.score(fn(preds(ra, ep), preds(rb, ep)), gt))
        print(name, json.dumps(adtest.summarize(scores)))
