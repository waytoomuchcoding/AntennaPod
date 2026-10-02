"""EmbeddingGemma through llama.cpp (GGUF, e.g. Q8_0) instead of PyTorch: checks speed on CPU or GPU and whether
the quantised model keeps the classifier's accuracy. Writes emb/<name>/<ep>.npy in emb_loo.py's layout.

  llama.cpp/build/bin/llama-server -m models/embeddinggemma-300M-Q8_0.gguf --embeddings -ngl 99 \
      -c 8192 -b 8192 -ub 8192 -np 1 --port 8093
  GT_DIR=gt_v2 ./venv/bin/python egemma_gguf.py egemma_q8_gpu 8093 [line]
"""
import json, os, sys, time, urllib.request
import numpy as np
import adtest
from emb_loo import eps, texts

name, port = sys.argv[1], sys.argv[2]
LINE_ONLY = len(sys.argv) > 3 and sys.argv[3] == "line"
PROMPT = "task: classification | query: "


def embed(xs):
    out = []
    for k in range(0, len(xs), 32):
        body = json.dumps({"input": [PROMPT + x for x in xs[k:k + 32]]}).encode()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"http://localhost:{port}/v1/embeddings", body, {"Content-Type": "application/json"}), timeout=600).read())
        out += [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
    return np.array(out, dtype=np.float32)


total_t = total_lines = 0
for ep in eps:
    path = f"emb/{name}/{ep}.npy"
    if os.path.exists(path):
        continue
    cur, ctx = texts(ep)
    t = time.time()
    X = embed(cur) if LINE_ONLY else np.hstack([embed(cur), embed(ctx)])
    dt = time.time() - t
    total_t += dt; total_lines += len(cur)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, X)
    print(f"  {name} {ep}: {len(cur)} lines in {dt:.1f}s", flush=True)
hours = sum(eps[e]["dur"] for e in eps) / 3600
print(f"done {name}: {total_lines} lines in {total_t:.0f}s = {total_t / hours:.0f}s per audio hour")
