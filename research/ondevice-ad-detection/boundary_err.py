"""Boundary errors per labelled ad break: signed start and end error (seconds, prediction minus label) of the
predicted interval(s) covering each break. Negative start = starts early (skips content), positive start = starts
late (ad plays). Prints median / mean absolute and the share of breaks with >5 s error, per run.
usage: GT_DIR=gt_v2 ./venv/bin/python boundary_err.py RUN [RUN ...]"""
import json, sys
import numpy as np
import adtest


def errors(run):
    st, en = [], []
    for e, m in adtest.EPISODES.items():
        if m.get("split") not in ("dev", "test"):
            continue
        pred = [(adtest.mmss(s), adtest.mmss(t)) for s, t in json.load(open(f"results/{run}/{e}.json"))["pred"]]
        for a, b, typ in adtest.load_gt(e)[1]:
            if typ != "ad":
                continue
            ov = [p for p in pred if adtest.overlap(p, (a, b)) > 0]
            if not ov:
                continue
            st.append(min(p[0] for p in ov) - a); en.append(max(p[1] for p in ov) - b)
    return np.array(st), np.array(en)


if __name__ == "__main__":
    for run in sys.argv[1:]:
        s, e = errors(run)
        f = lambda x: (f"median {np.median(x):+5.1f}s  mean|err| {np.abs(x).mean():4.1f}s  late>5s {np.mean(x > 5):.0%}  "
                       f"early>5s {np.mean(x < -5):.0%}")
        print(f"{run[:60]}\n   start: {f(s)}\n   end:   {f(e)}")
