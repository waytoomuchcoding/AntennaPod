"""Average the per-line held-out probabilities of several classifiers into a new score set.
usage: GT_DIR=gt_v2 GROUP=show ./venv/bin/python ensemble.py egemma bge-small tfidf  -> embp_.../ens_egemma+bge-small+tfidf/"""
import json, os, sys
import numpy as np
import emb_loo

d = f"embp_{emb_loo.GT}_{emb_loo.GROUP}"
out = f"{d}/ens_" + "+".join(sys.argv[1:])
os.makedirs(out, exist_ok=True)
for e in emb_loo.eps:
    p = np.mean([json.load(open(f"{d}/{n}/{e}.json")) for n in sys.argv[1:]], axis=0)
    json.dump([round(float(x), 4) for x in p], open(f"{out}/{e}.json", "w"))
print(os.path.basename(out))
