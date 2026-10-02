"""Embedding line classifiers vs TF-IDF, leave-one-episode-out on all 13 episodes.

Each line gets two vectors: the line itself and a 5-line window around it (both embedded as text). Logistic
regression on [line, window]. Scores are smoothed (3-line average) and turned into intervals like clf_loo.py.
The threshold is picked on the 7 dev episodes and applied unchanged to the 6 held-out ones.

usage: GT_DIR=gt_v2 ./venv/bin/python emb_loo.py tfidf potion8m minilm bge-small egemma [--veto RUN ...]
GROUP=show holds out whole shows. Embeddings are cached in emb/<name>/<ep>.npy, per-line held-out
probabilities in embp_<GT_DIR>_<GROUP>/<name>/<ep>.json.
"""
import json, os, re, sys, time
import numpy as np
import adtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

GT = os.environ.get("GT_DIR", "gt")
# GROUP=show holds out every episode of the same show together (same-day downloads share inserted ads).
GROUP = os.environ.get("GROUP", "episode")


def group_of(ep):
    return re.sub(r"\d+$", "", ep) if GROUP == "show" else ep
MODELS = {
    "potion8m": ("model2vec", "minishlab/potion-base-8M"),
    "potion32m": ("model2vec", "minishlab/potion-base-32M"),
    "minilm": ("st", "sentence-transformers/all-MiniLM-L6-v2"),
    "bge-small": ("st", "BAAI/bge-small-en-v1.5"),
    "egemma": ("st", "unsloth/embeddinggemma-300m"),
}
# Anything else (e.g. gemma_probe.py outputs) must already be cached in emb/<name>/.
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

eps = {}
for ep, meta in adtest.EPISODES.items():
    if not os.path.exists(f"{adtest.ROOT}/{GT}/{ep}.json"):
        continue
    dur, gt = adtest.load_gt(ep)
    lines = adtest.load_lines("moonshine", ep)
    lab = np.array([any(a <= (l["t"] + l["e"]) / 2 <= b for a, b, t in gt if t == "ad") for l in lines])
    neu = np.array([any(a <= (l["t"] + l["e"]) / 2 <= b for a, b, t in gt if t != "ad") for l in lines])
    eps[ep] = dict(lines=lines, y=lab, neu=neu, dur=dur, gt=gt, split=meta.get("split", "dev"))


def texts(ep):
    L = [l["text"] for l in eps[ep]["lines"]]
    return L, [" ".join(L[max(0, i - 2):i + 3]) for i in range(len(L))]


_enc = {}


def encoder(name):
    if name not in _enc:
        if name not in MODELS:
            raise SystemExit(f"no cached embeddings for {name}; run gemma_probe.py first")
        kind, rid = MODELS[name]
        if kind == "model2vec":
            from model2vec import StaticModel
            m = StaticModel.from_pretrained(rid)
            _enc[name] = lambda xs: m.encode(xs)
        else:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(rid, device="cpu")
            kw = {"prompt_name": "Classification"} if name == "egemma" and "Classification" in m.prompts else {}
            _enc[name] = lambda xs: m.encode(xs, batch_size=64, normalize_embeddings=True, **kw)
    return _enc[name]


def embed(name, ep):
    path = f"emb/{name}/{ep}.npy"
    if os.path.exists(path):
        return np.load(path)
    cur, ctx = texts(ep)
    t0 = time.time()
    enc = encoder(name)
    X = np.hstack([enc(cur), enc(ctx)]).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, X)
    print(f"  embedded {name} {ep}: {len(cur)} lines in {time.time() - t0:.1f}s", flush=True)
    return X


def loo_probs(name):
    out = {}
    if name == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import hstack
        for held in eps:
            tr = [e for e in eps if group_of(e) != group_of(held)]
            cur = sum((texts(e)[0] for e in tr), []); ctx = sum((texts(e)[1] for e in tr), [])
            y = np.concatenate([eps[e]["y"] for e in tr]); m = ~np.concatenate([eps[e]["neu"] for e in tr])
            v1 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(cur)
            v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(ctx)
            X = hstack([v1.transform(cur), v2.transform(ctx)]).tocsr()
            clf = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(X[m], y[m])
            c2, x2 = texts(held)
            out[held] = clf.predict_proba(hstack([v1.transform(c2), v2.transform(x2)]).tocsr())[:, 1]
        return out
    E = {e: embed(name, e) for e in eps}
    for held in eps:
        tr = [e for e in eps if group_of(e) != group_of(held)]
        X = np.vstack([E[e] for e in tr]); y = np.concatenate([eps[e]["y"] for e in tr])
        m = ~np.concatenate([eps[e]["neu"] for e in tr])
        sc = StandardScaler().fit(X[m])
        clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc.transform(X[m]), y[m])
        out[held] = clf.predict_proba(sc.transform(E[held]))[:, 1]
    return out


def intervals(ep, p, th):
    sm = np.convolve(p, np.ones(3) / 3, mode="same")
    return adtest.to_intervals(eps[ep]["lines"], list(sm > th), eps[ep]["dur"])


def evaluate(probs, split, th):
    return adtest.summarize([adtest.score(intervals(e, probs[e], th), eps[e]["gt"])
                             for e in eps if eps[e]["split"] == split])


def line_metrics(probs, split):
    y = np.concatenate([eps[e]["y"][~eps[e]["neu"]] for e in eps if eps[e]["split"] == split])
    p = np.concatenate([probs[e][~eps[e]["neu"]] for e in eps if eps[e]["split"] == split])
    return roc_auc_score(y, p), average_precision_score(y, p)


def f1_of(s):
    return s["f1"] if isinstance(s, dict) else float(str(s).split("F1 ")[1].split()[0])


def veto(run, probs, th):
    res = {}
    for split in ("dev", "test"):
        base, vet = [], []
        for e in eps:
            if eps[e]["split"] != split or not os.path.exists(f"results/{run}/{e}.json"):
                continue
            a = [(adtest.mmss(s), adtest.mmss(t)) for s, t in json.load(open(f"results/{run}/{e}.json"))["pred"]]
            b = intervals(e, probs[e], th)
            base.append(adtest.score(a, eps[e]["gt"]))
            vet.append(adtest.score([x for x in a if any(adtest.overlap(x, q) > 0 for q in b)], eps[e]["gt"]))
        res[split] = (adtest.summarize(base), adtest.summarize(vet))
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    runs = args[args.index("--veto") + 1:] if "--veto" in args else []
    names = args[:args.index("--veto")] if "--veto" in args else args
    for name in names:
        t0 = time.time()
        probs = loo_probs(name)
        d = f"embp_{GT}_{GROUP}/{name}"
        os.makedirs(d, exist_ok=True)
        for e, p in probs.items():
            json.dump([round(float(x), 4) for x in p], open(f"{d}/{e}.json", "w"))
        dev = {th: evaluate(probs, "dev", th) for th in THRESHOLDS}
        best = max(THRESHOLDS, key=lambda th: f1_of(dev[th]))
        auc_d, ap_d = line_metrics(probs, "dev"); auc_t, ap_t = line_metrics(probs, "test")
        print(f"== {name}  ({time.time() - t0:.0f}s)  line AUC dev {auc_d:.3f} test {auc_t:.3f} | "
              f"AP dev {ap_d:.3f} test {ap_t:.3f} | threshold (picked on dev) {best}")
        print(f"   alone dev  {dev[best]}")
        print(f"   alone test {evaluate(probs, 'test', best)}")
        for run in runs:
            r = veto(run, probs, best)
            for split in ("dev", "test"):
                print(f"   veto {run.split('__')[0][:18]}{run.split('quotes')[-1]} {split}: {r[split][0]} -> {r[split][1]}")
        sys.stdout.flush()
