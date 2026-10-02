"""Turn per-line ad probabilities into ad intervals: 3-line average + threshold (baseline), hysteresis, or a
two-state HMM decoded with Viterbi. All knobs are picked on the dev episodes and applied to the held-out ones.

usage: GT_DIR=gt_v2 GROUP=show ./venv/bin/python seq_smooth.py <name> [--veto RUN]
reads embp_<GT_DIR>_<GROUP>/<name>/<ep>.json written by emb_loo.py; report() is also used by distill.py.
"""
import itertools, json, os, sys
import numpy as np
import adtest


def ma(p, th):
    return np.convolve(p, np.ones(3) / 3, mode="same") > th


def hyst(p, hi, lo):
    sm = np.convolve(p, np.ones(3) / 3, mode="same")
    out, on = [], False
    for x in sm:
        on = x > hi or (on and x > lo)
        out.append(on)
    # also extend backwards: a run that starts above hi may have begun at a line above lo
    out = np.array(out)
    for i in range(len(out) - 2, -1, -1):
        if out[i + 1] and not out[i] and sm[i] > lo:
            out[i] = True
    return out


def viterbi(p, alpha, switch, bias=0.0):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    llr = alpha * np.log(p / (1 - p)) + bias          # evidence for "ad" per line
    ls = np.log(switch); stay = np.log(1 - switch)
    score = np.array([0.0, llr[0]]); back = []
    for x in llr[1:]:
        c = np.array([max(score[0] + stay, score[1] + ls), max(score[0] + ls, score[1] + stay) + x])
        back.append([int(score[1] + ls > score[0] + stay), int(score[1] + stay >= score[0] + ls)])
        score = c
    s = int(score[1] > score[0]); path = [s]
    for b in reversed(back):
        s = b[s]; path.append(s)
    return np.array(path[::-1], dtype=bool)


GRIDS = {
    "ma3": (ma, [dict(th=t) for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]),
    "hysteresis": (hyst, [dict(hi=h, lo=l) for h, l in itertools.product((0.3, 0.5, 0.7, 0.85), (0.05, 0.1, 0.2, 0.3))
                          if l < h]),
    "hmm": (viterbi, [dict(alpha=a, switch=s, bias=b) for a, s, b in itertools.product(
        (0.5, 1.0, 2.0, 4.0), (1e-3, 1e-2, 0.05, 0.2), (-1.0, 0.0, 1.0, 2.0, 3.0))]),
}


def evaluate(P, eps, fn, split, **kw):
    return adtest.summarize([adtest.score(adtest.to_intervals(eps[e]["lines"], list(fn(P[e], **kw)), eps[e]["dur"]),
                                          eps[e]["gt"]) for e in P if eps[e]["split"] == split])


def veto(P, eps, run, fn, split, **kw):
    out = []
    for e in P:
        if eps[e]["split"] != split or not os.path.exists(f"results/{run}/{e}.json"):
            continue
        a = [(adtest.mmss(s), adtest.mmss(t)) for s, t in json.load(open(f"results/{run}/{e}.json"))["pred"]]
        b = adtest.to_intervals(eps[e]["lines"], list(fn(P[e], **kw)), eps[e]["dur"])
        out.append(adtest.score([x for x in a if any(adtest.overlap(x, q) > 0 for q in b)], eps[e]["gt"]))
    return adtest.summarize(out)


def report(P, eps, run=None):
    for method, (fn, grid) in GRIDS.items():
        best = max(grid, key=lambda kw: (evaluate(P, eps, fn, "dev", **kw)["f1"],
                                         -evaluate(P, eps, fn, "dev", **kw)["false_alarms"]))
        line = (f"  {method:10s} {best}\n     dev  {evaluate(P, eps, fn, 'dev', **best)}"
                f"\n     test {evaluate(P, eps, fn, 'test', **best)}")
        if run:
            line += f"\n     veto test {veto(P, eps, run, fn, 'test', **best)}  veto dev {veto(P, eps, run, fn, 'dev', **best)}"
        print(line, flush=True)


if __name__ == "__main__":
    import emb_loo
    name = sys.argv[1]
    run = sys.argv[sys.argv.index("--veto") + 1] if "--veto" in sys.argv else None
    P = {e: np.array(json.load(open(f"embp_{emb_loo.GT}_{emb_loo.GROUP}/{name}/{e}.json"))) for e in emb_loo.eps}
    print(f"== {name} ({emb_loo.GROUP} held out)")
    report(P, emb_loo.eps, run)
