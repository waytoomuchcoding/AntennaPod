"""Write a classifier's intervals as a results/ run, so verify.py can check them with an LLM
("classifier proposes, LLM verifies"). Smoothing knobs as picked on dev by seq_smooth.py.
usage: GT_DIR=gt_v2 GROUP=show ./venv/bin/python clf_run.py <name> hmm alpha=1 switch=0.01 bias=-1"""
import json, os, sys
import numpy as np
import adtest, emb_loo, seq_smooth

name, method = sys.argv[1], sys.argv[2]
kw = {k: float(v) for k, v in (a.split("=") for a in sys.argv[3:])}
fn = seq_smooth.GRIDS[method][0]
run = f"clf_{name}__moonshine__{method}"
os.makedirs(f"results/{run}", exist_ok=True)
scores = []
for e, ep in emb_loo.eps.items():
    p = np.array(json.load(open(f"embp_{emb_loo.GT}_{emb_loo.GROUP}/{name}/{e}.json")))
    iv = adtest.to_intervals(ep["lines"], list(fn(p, **kw)), ep["dur"])
    sc = adtest.score(iv, ep["gt"]); scores.append(sc)
    json.dump({"episode": e, "pred": [[adtest.fmt(s), adtest.fmt(t)] for s, t in iv], "score": sc,
               "stats": {"calls": 0, "in_tokens": 0, "out_tokens": 0, "seconds": 0.0}, "audio_s": ep["dur"]},
              open(f"results/{run}/{e}.json", "w"), indent=1)
print(run, adtest.summarize(scores))
