"""Fine-tune a small pretrained encoder (default bge-small-en-v1.5, 33M, BERT architecture) as a line classifier on
the HUMAN labels. Input: (line, 5-line window) text pair. Clean evaluation without 9 leave-one-show-out runs:
3 folds by show group (each labelled episode is scored by a model that never saw its show), plus one model on all
13 episodes for the 29 unseen auto-labelled shows. Writes embp_ft/<out>/<ep>.json for eval_all.py scores.
usage: GT_DIR=gt_v2 ./venv/bin/python ft_human.py [model] [epochs]
"""
import json, os, random, sys, time
import numpy as np
import torch
import adtest
from emb_loo import eps, texts
import eval_all

MODEL = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-en-v1.5"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = f"embp_ft/{MODEL.split('/')[-1]}_human_e{EPOCHS}"
FOLDS = [{"daily", "dateline"}, {"sysk", "crimejunkie", "freak", "planetmoney"}, {"conan", "morbid", "huberman"}]
torch.set_num_threads(6)
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)


def show(e):
    return e.rstrip("0123456789")


def data(names):
    rows = []
    for e in names:
        cur, ctx = texts(e)
        for i in range(len(cur)):
            if not eps[e]["neu"][i]:
                rows.append((cur[i], ctx[i], float(eps[e]["y"][i])))
    return rows


def train(names, seed=1):
    torch.manual_seed(seed); random.seed(seed)
    enc = AutoModel.from_pretrained(MODEL); head = torch.nn.Linear(enc.config.hidden_size, 1)
    rows = data(names); pos = sum(r[2] for r in rows)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(rows) - pos) / pos))
    params = list(enc.parameters()) + list(head.parameters())
    BS = 32; steps = EPOCHS * (len(rows) // BS)
    opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-5, total_steps=steps, pct_start=0.1)
    logits = lambda a, b: head(enc(**tok(a, b, truncation="longest_first", max_length=192, padding=True,
                                         return_tensors="pt")).last_hidden_state[:, 0]).squeeze(-1)
    enc.train(); t0 = time.time()
    for ep in range(EPOCHS):
        random.shuffle(rows)
        for k in range(0, len(rows) - BS + 1, BS):
            a, b, y = zip(*rows[k:k + BS])
            loss = lossf(logits(list(a), list(b)), torch.tensor(y))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    print(f"  trained on {len(names)} episodes, {len(rows)} lines, {EPOCHS} epochs in {time.time() - t0:.0f}s", flush=True)
    enc.eval()

    def predict(e):
        cur, ctx = texts(e) if e in eps else _texts_unseen(e)
        t = time.time()
        with torch.no_grad():
            p = np.concatenate([torch.sigmoid(logits(cur[k:k + 64], ctx[k:k + 64])).numpy() for k in range(0, len(cur), 64)])
        return p, time.time() - t
    return predict


def _texts_unseen(e):
    L = [l["text"] for l in eval_all.LINES[e]]
    return L, [" ".join(L[max(0, i - 2):i + 3]) for i in range(len(L))]


os.makedirs(OUT, exist_ok=True)
for fold in FOLDS:
    held = [e for e in eps if show(e) in fold]
    pred = train([e for e in eps if show(e) not in fold])
    for e in held:
        p, _ = pred(e); json.dump([round(float(v), 4) for v in p], open(f"{OUT}/{e}.json", "w"))
pred = train(list(eps))
secs = hours = 0
for e in eval_all.UNSEEN:
    p, dt = pred(e); secs += dt; hours += eval_all.auto_gt(e)[0] / 3600
    json.dump([round(float(v), 4) for v in p], open(f"{OUT}/{e}.json", "w"))
print(f"inference: {secs:.0f}s for {hours:.1f} h of unseen audio = {secs / hours:.0f}s per audio hour (CPU, fp32)")
print(OUT)
