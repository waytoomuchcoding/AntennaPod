"""Distillation: train the cheap line classifier on teacher (E4B copy6) labels for the unlabelled extra episodes,
then test it on the 13 human-labelled episodes it never saw.

Training sets compared (features from emb_loo.py's cache, e.g. bge-small; extra episodes are embedded here):
  human     leave-one-show-out on the 13 labelled episodes (the 13.1 baseline)
  teacher   only the extra episodes with teacher labels -> every labelled episode is unseen
  both      teacher episodes + the labelled episodes of the other shows
Scored per split (dev / test) with the threshold or HMM knobs picked on dev, alone and as a veto.
usage: GT_DIR=gt_v2 GROUP=show ./venv/bin/python distill.py bge-small [TEACHER_RUN]
"""
import json, os, sys
import numpy as np
import adtest
import emb_loo
from emb_loo import eps, group_of, embed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

name = sys.argv[1]
TEACHER = sys.argv[2] if len(sys.argv) > 2 else "gemma-4-E4B-it.litertlm__moonshine__quotes_teacher"
VETO_RUN = "gemma-4-E2B-it.litertlm__moonshine__quotes_copy6_v2_verified"

# teacher-labelled extra episodes, registered in emb_loo's episode table with split "extra"
extra = []
for e, meta in adtest.EPISODES.items():
    f = f"results/{TEACHER}/{e}.json"
    if meta.get("split") != "extra" or not os.path.exists(f):
        continue
    r = json.load(open(f))
    lines = adtest.load_lines("moonshine", e)
    ivs = [(adtest.mmss(s), adtest.mmss(t)) for s, t in r["pred"]]
    y = np.array([any(a <= (l["t"] + l["e"]) / 2 <= b for a, b in ivs) for l in lines])
    eps[e] = dict(lines=lines, y=y, neu=np.zeros(len(lines), bool), dur=r["audio_s"], gt=[], split="extra")
    extra.append(e)
labelled = [e for e in eps if eps[e]["split"] in ("dev", "test")]
print(f"{len(extra)} teacher episodes, {sum(eps[e]['y'].sum() for e in extra)} of "
      f"{sum(len(eps[e]['y']) for e in extra)} lines labelled ad by the teacher")

E = {e: embed(name, e) for e in eps}


def fit_predict(train, held):
    X = np.vstack([E[e] for e in train]); y = np.concatenate([eps[e]["y"] for e in train])
    m = ~np.concatenate([eps[e]["neu"] for e in train])
    sc = StandardScaler().fit(X[m])
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc.transform(X[m]), y[m])
    return clf.predict_proba(sc.transform(E[held]))[:, 1]


def train_set(mode, held):
    others = [e for e in labelled if group_of(e) != group_of(held)]
    return {"human": others, "teacher": extra, "both": extra + others}[mode]


import seq_smooth as S   # smoothing + scoring shared with seq_smooth.py

for mode in ("human", "teacher", "both"):
    P = {}
    for held in labelled:
        P[held] = fit_predict(train_set(mode, held), held)
    yd = np.concatenate([eps[e]["y"][~eps[e]["neu"]] for e in labelled if eps[e]["split"] == "test"])
    pd = np.concatenate([P[e][~eps[e]["neu"]] for e in labelled if eps[e]["split"] == "test"])
    print(f"== {name} trained on {mode}: test line AUC {roc_auc_score(yd, pd):.3f} AP {average_precision_score(yd, pd):.3f}")
    S.report(P, eps, VETO_RUN)
    os.makedirs(f"embp_distill/{name}_{mode}", exist_ok=True)
    for e, p in P.items():
        json.dump([round(float(x), 4) for x in p], open(f"embp_distill/{name}_{mode}/{e}.json", "w"))
