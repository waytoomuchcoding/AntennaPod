"""Idea 2 (README 15.2): a small bidirectional GRU over the per-line feature sequence of a whole episode, instead of
the per-line logistic regression. Learns break structure (e.g. the strong signal of a break's second ad can pull
its disguised first ad in). Trained on human labels only: leave-one-show-out for the 13 labelled episodes, and one
model on all 13 for the 29 unseen shows. Writes per-line probabilities to embp_seq/<out>/ for eval_all.py scores.
usage: GT_DIR=gt_v2 GROUP=show ./venv/bin/python seq_model.py egemma_ctr [hidden] [epochs]
"""
import json, os, sys
import numpy as np
import torch
from emb_loo import eps, group_of
import eval_all

FEAT = sys.argv[1]
H = int(sys.argv[2]) if len(sys.argv) > 2 else 64
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
OUT = f"embp_seq/{FEAT}_gru{H}_e{EPOCHS}"
torch.manual_seed(1); torch.set_num_threads(3)
E = {e: np.load(f"emb/{FEAT}/{e}.npy") for e in list(eps) + eval_all.UNSEEN}
# BULK=N adds N Gemini-labelled bulk episodes (labels as in train_bulk.py, union with alignment) to every
# training set; FEAT must then have their features (egemma_q8_ctr does).
BULK = []
if os.environ.get("BULK"):
    import adtest
    for e in sorted(f[:-5] for f in os.listdir("gt_gemini") if f.startswith("b_")):
        if len(BULK) >= int(os.environ["BULK"]) or not os.path.exists(f"emb/{FEAT}/{e}.npy"):
            continue
        lines = adtest.load_lines("moonshine", e)
        seg = [(adtest.mmss(s["s"]), adtest.mmss(s["e"]), s["type"]) for s in json.load(open(f"gt_gemini/{e}.json"))["segments"]]
        if os.path.exists(f"gt_auto_bulk/{e}.json"):
            seg += [(adtest.mmss(s["s"]), adtest.mmss(s["e"]), "ad") for s in json.load(open(f"gt_auto_bulk/{e}.json"))["segments"]]
        mid = [(l["t"] + l["e"]) / 2 for l in lines]
        eps[e] = {"y": np.array([any(a <= m <= b for a, b, t in seg if t == "ad") for m in mid]),
                  "neu": np.array([any(a <= m <= b for a, b, t in seg if t != "ad") for m in mid])}
        E[e] = np.load(f"emb/{FEAT}/{e}.npy"); BULK.append(e)
    OUT += f"_bulk{len(BULK)}"
LABELLED = [e for e in eps if e not in BULK]


class Net(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(d, 128), torch.nn.GELU())
        self.gru = torch.nn.GRU(128, H, batch_first=True, bidirectional=True)
        self.out = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(2 * H, 1))

    def forward(self, x):
        return self.out(self.gru(self.proj(x))[0]).squeeze(-1)


def train(names):
    mu = np.concatenate([E[e] for e in names]).mean(0); sd = np.concatenate([E[e] for e in names]).std(0) + 1e-6
    norm = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32)[None]
    net = Net(E[names[0]].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    y_all = np.concatenate([eps[e]["y"] for e in names])
    pw = torch.tensor((1 - y_all.mean()) / y_all.mean())
    data = [(norm(E[e]), torch.tensor(eps[e]["y"], dtype=torch.float32), torch.tensor(~eps[e]["neu"])) for e in names]
    for ep in range(EPOCHS):
        net.train()
        for k in np.random.RandomState(ep).permutation(len(data)):
            x, y, m = data[k]
            # random crops of 60-200 lines so the model also sees episode edges in the middle of sequences
            n = x.shape[1]; L = min(n, np.random.randint(60, 201)); s = np.random.randint(0, n - L + 1)
            logit = net(x[:, s:s + L])[0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit[m[s:s + L]], y[s:s + L][m[s:s + L]],
                                                                        pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return lambda X: torch.sigmoid(net(norm(X))[0]).detach().numpy()


os.makedirs(OUT, exist_ok=True)
for held in LABELLED:
    f = train([e for e in LABELLED if group_of(e) != group_of(held)] + BULK)
    json.dump([round(float(v), 4) for v in f(E[held])], open(f"{OUT}/{held}.json", "w"))
f = train(LABELLED + BULK)
for e in eval_all.UNSEEN:
    json.dump([round(float(v), 4) for v in f(E[e])], open(f"{OUT}/{e}.json", "w"))
print(OUT)
