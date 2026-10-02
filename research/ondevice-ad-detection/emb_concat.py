"""Concatenate cached feature sets (e.g. text + audio) into emb/<a+b>/ for emb_loo.py.
usage: GT_DIR=gt_v2 ./venv/bin/python emb_concat.py egemma whisper"""
import os, sys
import numpy as np
from emb_loo import eps

names = sys.argv[1:]
out = "emb/" + "+".join(names)
os.makedirs(out, exist_ok=True)
for e in eps:
    np.save(f"{out}/{e}.npy", np.hstack([np.load(f"emb/{n}/{e}.npy") for n in names]))
print(os.path.basename(out))
