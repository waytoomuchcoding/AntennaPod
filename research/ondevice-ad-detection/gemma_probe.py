"""Gemma hidden states as line features ("linear probe"), through llama-server's per-token embeddings.

Start a server first, e.g.
  llama.cpp/build/bin/llama-server -m models/gemma-4-E2B-it-Q4_K_M.gguf --embeddings --pooling none -ngl 99 \
      -c 2048 -b 2048 -ub 2048 --port 8092 [--override-kv gemma4.block_count=int:N]
(--override-kv truncates the network, so the "final" hidden state is the output of layer N.)

The transcript is fed as windows of `ctx` context lines + `step` target lines (no prompt, no generation). Each
target line gets the mean and the last of its token states. Saved to emb/<name>/<ep>.npy for emb_loo.py.
usage: ./venv/bin/python gemma_probe.py <name> [port] [ctx] [step]
"""
import json, os, sys, time, urllib.request
import numpy as np
import adtest

name = sys.argv[1]
url = f"http://localhost:{sys.argv[2] if len(sys.argv) > 2 else 8092}"
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 20
STEP = int(sys.argv[4]) if len(sys.argv) > 4 else 40
MAX_TOK = 2000


def post(path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


bos = post("/tokenize", {"content": "", "add_special": True})["tokens"]
GT = os.environ.get("GT_DIR", "gt")
total_tok, t_start = 0, time.time()
for ep in adtest.EPISODES:
    path = f"emb/{name}/{ep}.npy"
    if os.path.exists(path) or not os.path.exists(f"{adtest.ROOT}/{GT}/{ep}.json"):
        continue
    lines = adtest.load_lines("moonshine", ep)
    toks = [post("/tokenize", {"content": "\n" + l["text"], "add_special": False})["tokens"] for l in lines]
    feats = [None] * len(lines)
    t0 = time.time()
    for start in range(0, len(lines), STEP):
        lo = max(0, start - CTX)
        idx = list(range(lo, min(len(lines), start + STEP)))
        while sum(len(toks[i]) for i in idx) + len(bos) > MAX_TOK:   # drop context first, then targets
            idx = idx[1:] if idx[0] < start else idx[:-1]
        seq, spans = list(bos), {}
        for i in idx:
            spans[i] = (len(seq), len(seq) + len(toks[i]))
            seq += toks[i]
        H = np.array(post("/embeddings", {"content": seq})[0]["embedding"], dtype=np.float32)
        total_tok += len(seq)
        for i in idx:
            if i >= start and feats[i] is None:
                a, b = spans[i]
                feats[i] = np.concatenate([H[a:b].mean(0), H[b - 1]])
    missing = [i for i, f in enumerate(feats) if f is None]
    assert not missing, f"{ep}: lines without features {missing[:5]}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.vstack(feats))
    print(f"  {name} {ep}: {len(lines)} lines in {time.time() - t0:.1f}s", flush=True)
print(f"done {name}: {total_tok} tokens in {time.time() - t_start:.0f}s")
