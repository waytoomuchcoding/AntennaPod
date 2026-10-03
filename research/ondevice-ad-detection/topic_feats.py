"""Idea 1 (README 15.1): "is this line off-topic for this episode?" features from the cached EmbeddingGemma
vectors (emb/egemma/<ep>.npy = [line 768, 5-line window 768]). Writes new feature sets for eval_all.py:
  egemma_ctr   [original, window vector minus the episode's mean window vector]   (episode-centred)
  egemma_nov   [original, novelty scalars]: cosine of the line/window to the episode mean, to the previous and next
               ~2 minutes (20 lines), and between previous and next 2 minutes
  egemma_both  [original, centred, novelty]
"""
import os
import numpy as np
import adtest
from emb_loo import eps
import eval_all

D = 768


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def novelty(X, n=20):
    L, W = unit(X[:, :D]), unit(X[:, D:])
    mean_w = unit(W.mean(0))
    out = []
    for i in range(len(W)):
        prev = unit(W[max(0, i - n):i].mean(0)) if i else mean_w
        nxt = unit(W[i + 1:i + 1 + n].mean(0)) if i + 1 < len(W) else mean_w
        out.append([L[i] @ mean_w, W[i] @ mean_w, W[i] @ prev, W[i] @ nxt, prev @ nxt,
                    min(W[i] @ prev, W[i] @ nxt)])
    return np.array(out, dtype=np.float32)


for e in list(eps) + eval_all.UNSEEN:
    X = np.load(f"emb/egemma/{e}.npy")
    ctr = X[:, D:] - X[:, D:].mean(0)
    nov = novelty(X)
    for name, F in (("egemma_ctr", np.hstack([X, ctr])), ("egemma_nov", np.hstack([X, nov * 10])),
                    ("egemma_both", np.hstack([X, ctr, nov * 10]))):
        os.makedirs(f"emb/{name}", exist_ok=True)
        np.save(f"emb/{name}/{e}.npy", F.astype(np.float32))
print("ok")
