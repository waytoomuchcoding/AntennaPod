"""Fine-tune a small encoder (default bge-small-en-v1.5, 33M params) end to end as a line classifier on the
E4B teacher's labels for the unlabelled extra episodes, then score every line of the 13 human-labelled episodes
(none of them, nor their shows, is in the training data). Input is a text pair: (line, 5-line window).
Writes per-line probabilities to embp_distill/<out>/<ep>.json and prints the seq_smooth report.

usage: GT_DIR=gt_v2 ./venv/bin/python finetune.py [out_name] [model] [epochs]
"""
import json, os, random, sys, time
import numpy as np
import torch
import adtest
from emb_loo import eps, texts
import seq_smooth

OUT = sys.argv[1] if len(sys.argv) > 1 else "ft_bge_small_teacher"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "BAAI/bge-small-en-v1.5"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
TEACHER = os.environ.get("TEACHER", "gemma-4-E4B-it.litertlm__moonshine__quotes_teacher")
torch.manual_seed(1); random.seed(1)
torch.set_num_threads(os.cpu_count())

from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
enc = AutoModel.from_pretrained(MODEL)
head = torch.nn.Linear(enc.config.hidden_size, 1)

train = []
for e, meta in adtest.EPISODES.items():
    f = f"results/{TEACHER}/{e}.json"
    if meta.get("split") != "extra" or not os.path.exists(f):
        continue
    lines = adtest.load_lines("moonshine", e)
    ivs = [(adtest.mmss(s), adtest.mmss(t)) for s, t in json.load(open(f))["pred"]]
    L = [l["text"] for l in lines]
    for i, l in enumerate(lines):
        y = any(a <= (l["t"] + l["e"]) / 2 <= b for a, b in ivs)
        train.append((L[i], " ".join(L[max(0, i - 2):i + 3]), float(y)))
pos = sum(t[2] for t in train)
print(f"{len(train)} training lines, {int(pos)} ad", flush=True)
pos_weight = torch.tensor((len(train) - pos) / max(pos, 1))


def logits(a, b):
    x = tok(a, b, truncation="only_second", max_length=192, padding=True, return_tensors="pt")
    return head(enc(**x).last_hidden_state[:, 0]).squeeze(-1)


params = list(enc.parameters()) + list(head.parameters())
opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.01)
BS = 32
steps = EPOCHS * (len(train) // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-5, total_steps=steps, pct_start=0.1)
lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
enc.train(); t0 = time.time(); step = 0
for ep in range(EPOCHS):
    random.shuffle(train)
    for k in range(0, len(train) - BS + 1, BS):
        a, b, y = zip(*train[k:k + BS])
        loss = lossf(logits(list(a), list(b)), torch.tensor(y))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        step += 1
        if step % 50 == 0:
            print(f"epoch {ep} step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)

enc.eval(); P = {}
os.makedirs(f"embp_distill/{OUT}", exist_ok=True)
with torch.no_grad():
    for e in eps:
        if eps[e]["split"] not in ("dev", "test"):
            continue
        cur, ctx = texts(e)
        p = np.concatenate([torch.sigmoid(logits(cur[k:k + 64], ctx[k:k + 64])).numpy() for k in range(0, len(cur), 64)])
        P[e] = p
        json.dump([round(float(x), 4) for x in p], open(f"embp_distill/{OUT}/{e}.json", "w"))
torch.save({"enc": enc.state_dict(), "head": head.state_dict()}, f"logs/{OUT}.pt")
from sklearn.metrics import average_precision_score
for split in ("dev", "test"):
    y = np.concatenate([eps[e]["y"][~eps[e]["neu"]] for e in P if eps[e]["split"] == split])
    p = np.concatenate([P[e][~eps[e]["neu"]] for e in P if eps[e]["split"] == split])
    print(f"== {OUT} {split} line AP {average_precision_score(y, p):.3f}")
seq_smooth.report(P, eps, "gemma-4-E2B-it.litertlm__moonshine__quotes_copy6_v2_verified")
