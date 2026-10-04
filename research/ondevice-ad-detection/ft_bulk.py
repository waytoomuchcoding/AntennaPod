"""README 16.5: fine-tune a small BERT-style encoder (bge-small-en-v1.5, 33M) as a line classifier on the
Gemini-labelled bulk episodes ONLY (union with the alignment labels, self-promo left out), then score the 13
human-labelled episodes and the 29 unseen auto-labelled shows (none of their shows is in training), in fp32 and
after dynamic int8 quantisation (the size a phone would run). Writes embp_ft/<out>[_int8]/<ep>.json for
`eval_all.py scores`. Input per line: (line, 5-line window) text pair, like ft_human.py.
usage: BULK=80 GT_DIR=gt_v2 ./venv/bin/python ft_bulk.py [model] [epochs]
"""
import json, os, random, sys, time
import numpy as np
import torch
import adtest, eval_all
from emb_loo import eps, texts

MODEL = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-en-v1.5"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
NB = int(os.environ.get("BULK", 80))
OUT = f"embp_ft/{MODEL.split('/')[-1]}_bulk{NB}_e{EPOCHS}"
torch.manual_seed(1); random.seed(1); torch.set_num_threads(6)
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)


def pair_texts(lines):
    L = [l["text"] for l in lines]
    return L, [" ".join(L[max(0, i - 2):i + 3]) for i in range(len(L))]


rows, bulk = [], sorted(f[:-5] for f in os.listdir("gt_gemini") if f.startswith("b_"))[:NB]
for e in bulk:
    lines = adtest.load_lines("moonshine", e)
    seg = [(adtest.mmss(s["s"]), adtest.mmss(s["e"]), s["type"]) for s in json.load(open(f"gt_gemini/{e}.json"))["segments"]]
    if os.path.exists(f"gt_auto_bulk/{e}.json"):
        seg += [(adtest.mmss(s["s"]), adtest.mmss(s["e"]), "ad") for s in json.load(open(f"gt_auto_bulk/{e}.json"))["segments"]]
    cur, ctx = pair_texts(lines)
    for i, l in enumerate(lines):
        m = (l["t"] + l["e"]) / 2
        if any(a <= m <= b for a, b, t in seg if t != "ad"):
            continue                                            # self-promo: neutral, not trained on
        rows.append((cur[i], ctx[i], float(any(a <= m <= b for a, b, t in seg if t == "ad"))))
pos = sum(r[2] for r in rows)
print(f"{len(bulk)} bulk episodes, {len(rows)} training lines, {int(pos)} ad", flush=True)

enc = AutoModel.from_pretrained(MODEL); head = torch.nn.Linear(enc.config.hidden_size, 1)
logits = lambda a, b: head(enc(**tok(a, b, truncation="longest_first", max_length=192, padding=True,
                                     return_tensors="pt")).last_hidden_state[:, 0]).squeeze(-1)
lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(rows) - pos) / pos))
params = list(enc.parameters()) + list(head.parameters())
BS = 32; steps = EPOCHS * (len(rows) // BS)
opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-5, total_steps=steps, pct_start=0.1)
enc.train(); t0 = time.time(); step = 0
for ep in range(EPOCHS):
    random.shuffle(rows)
    for k in range(0, len(rows) - BS + 1, BS):
        a, b, y = zip(*rows[k:k + BS])
        loss = lossf(logits(list(a), list(b)), torch.tensor(y))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
print(f"trained in {time.time() - t0:.0f}s", flush=True)
enc.eval()
os.makedirs("logs/models", exist_ok=True)
torch.save({"enc": enc.state_dict(), "head": head.state_dict()}, f"logs/models/{os.path.basename(OUT)}.pt")

test = {e: pair_texts(eps[e]["lines"]) for e in eps}
test.update({e: pair_texts(eval_all.LINES[e]) for e in eval_all.UNSEEN})


def score_all(out):
    os.makedirs(out, exist_ok=True); t = time.time(); n = 0
    with torch.no_grad():
        for e, (cur, ctx) in test.items():
            p = np.concatenate([torch.sigmoid(logits(cur[k:k + 64], ctx[k:k + 64])).numpy()
                                for k in range(0, len(cur), 64)])
            json.dump([round(float(v), 4) for v in p], open(f"{out}/{e}.json", "w")); n += len(cur)
    print(f"{out}: {n} lines scored in {time.time() - t:.0f}s ({(time.time() - t) / n * 1000:.1f} ms/line)", flush=True)


score_all(OUT)
enc = torch.ao.quantization.quantize_dynamic(enc, {torch.nn.Linear}, dtype=torch.qint8)
score_all(OUT + "_int8")
print("DONE")
