"""README 16.3: does more (Gemini-labelled) training data help the cheap classifier?
Features: EmbeddingGemma Q8_0 on the GPU (llama-server, port 8093; see egemma_gguf.py) for [line, 5-line window],
plus the episode-centred window vector (idea 1). Cached in emb/egemma_q8_gpu/<ep>.npy (raw) for all episodes.
Training sets: human (13 hand-labelled episodes, leave-one-show-out), human + bulk, bulk only. Bulk labels come
from gt_gemini/ (self-promo lines left out of training). Scored with eval_all.run (dev/test F1 on the 13, worst
cases, recall on the 29 unseen auto-labelled shows).
usage: GT_DIR=gt_v2 GROUP=show ./venv/bin/python train_bulk.py [port]
"""
import json, os, sys, time, urllib.request
import numpy as np
import adtest, eval_all
from emb_loo import eps, group_of

PORT = sys.argv[1] if len(sys.argv) > 1 else "8093"
PROMPT = "task: classification | query: "
D = 768


def texts_of(lines):
    L = [l["text"] for l in lines]
    return L, [" ".join(L[max(0, i - 2):i + 3]) for i in range(len(L))]


def embed(xs):
    out = []
    for k in range(0, len(xs), 32):
        body = json.dumps({"input": [PROMPT + x for x in xs[k:k + 32]]}).encode()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"http://localhost:{PORT}/v1/embeddings", body, {"Content-Type": "application/json"}), timeout=600).read())
        out += [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
    return np.array(out, dtype=np.float32)


def feats(ep, lines):
    path = f"emb/egemma_q8_gpu/{ep}.npy"
    if not os.path.exists(path):
        cur, ctx = texts_of(lines)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.hstack([embed(cur), embed(ctx)]))
    X = np.load(path)
    return np.hstack([X, X[:, D:] - X[:, D:].mean(0)])           # + episode-centred window vector


def bulk_labels(ep, lines):
    g = json.load(open(f"gt_gemini/{ep}.json"))["segments"]
    seg = [(adtest.mmss(s["s"]), adtest.mmss(s["e"]), s["type"]) for s in g]
    mid = [(l["t"] + l["e"]) / 2 for l in lines]
    if os.environ.get("LABELS") == "union" and os.path.exists(f"gt_auto_bulk/{ep}.json"):
        # add the inserted ads found by audio alignment (Gemini misses ~18% of them in bulk episodes)
        seg += [(adtest.mmss(s["s"]), adtest.mmss(s["e"]), "ad") for s in json.load(open(f"gt_auto_bulk/{ep}.json"))["segments"]]
    y = np.array([any(a <= m <= b for a, b, t in seg if t == "ad") for m in mid])
    neu = np.array([any(a <= m <= b for a, b, t in seg if t != "ad") for m in mid])
    return y, neu


t0 = time.time()
F = {e: feats(e, eps[e]["lines"]) for e in eps}
F.update({e: feats(e, eval_all.LINES[e]) for e in eval_all.UNSEEN})
bulk = sorted(f[:-5] for f in os.listdir("gt_gemini") if f.startswith("b_"))
# BULK_MAX caps the bulk set (prefers episodes whose features are already cached), so a run can finish while
# labelling continues in the background.
if os.environ.get("BULK_MAX"):
    bulk = sorted(bulk, key=lambda e: not os.path.exists(f"emb/egemma_q8_gpu/{e}.npy"))[:int(os.environ["BULK_MAX"])]
B = {}
for e in bulk:
    lines = adtest.load_lines("moonshine", e)
    F[e] = feats(e, lines)
    B[e] = bulk_labels(e, lines)
hours = sum(adtest.load_lines("moonshine", e)[-1]["e"] for e in bulk) / 3600
print(f"features ready in {time.time() - t0:.0f}s; {len(bulk)} Gemini-labelled bulk episodes ({hours:.0f} h), "
      f"{sum(B[e][0].sum() for e in bulk)} ad lines of {sum(len(B[e][0]) for e in bulk)}", flush=True)


def data(names):
    X = np.vstack([F[e] for e in names])
    y = np.concatenate([eps[e]["y"] if e in eps else B[e][0] for e in names])
    m = ~np.concatenate([eps[e]["neu"] if e in eps else B[e][1] for e in names])
    return X, y, m


for mode in ("human", "human+bulk", "bulk"):
    P = {}
    for held in eps:
        others = [e for e in eps if group_of(e) != group_of(held)]
        names = {"human": others, "human+bulk": others + bulk, "bulk": bulk}[mode]
        P[held] = eval_all.fit(*data(names))(F[held])
    names = {"human": list(eps), "human+bulk": list(eps) + bulk, "bulk": bulk}[mode]
    f = eval_all.fit(*data(names))
    Pu = {e: f(F[e]) for e in eval_all.UNSEEN}
    eval_all.run(f"{mode} ({len(bulk)} bulk eps)" if mode != "human" else "human only (q8, centred)", P, Pu)
