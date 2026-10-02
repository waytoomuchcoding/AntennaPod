"""Repetition matching: inserted ads recur word for word across episodes and shows.

For each episode, build a database of word 6-gram hashes from the ads detected in the OTHER episodes, then flag a
line when at least half of its 6-grams are in the database. Database sources:
  raw     every line of every other episode (upper bound on what repeats; includes intros/credits)
  llm     only lines inside the LLM's verified ad intervals of the other episodes (what a phone would have)
  gt      only lines inside labelled ads of the other episodes (perfect detector)
Reports how much ad time the match alone finds, its false alarms, and how many LLM-detected breaks it covers
(breaks a phone could skip the LLM for).
usage: GT_DIR=gt_v2 ./venv/bin/python repeat_match.py [RUN] [n] [min_frac]
"""
import json, os, re, sys
import adtest
from emb_loo import eps

RUN = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-E2B-it.litertlm__moonshine__quotes_copy6_v2_verified"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5


def words(t):
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", t.lower()).split() if w]


def grams(ws):
    return {hash(tuple(ws[i:i + N])) for i in range(len(ws) - N + 1)}


def line_grams(ep):
    # shingles over the running word stream, assigned to the line where they start
    out, stream, owner = [], [], []
    for i, l in enumerate(eps[ep]["lines"]):
        w = words(l["text"]); stream += w; owner += [i] * len(w)
    per = [set() for _ in eps[ep]["lines"]]
    for k in range(len(stream) - N + 1):
        per[owner[k]].add(hash(tuple(stream[k:k + N])))
    return per


LG = {e: line_grams(e) for e in eps}


def llm_iv(e):
    f = f"results/{RUN}/{e}.json"
    return [(adtest.mmss(s), adtest.mmss(t)) for s, t in json.load(open(f))["pred"]] if os.path.exists(f) else []


def in_iv(l, ivs):
    m = (l["t"] + l["e"]) / 2
    return any(a <= m <= b for a, b in ivs)


def db_for(held, source):
    db = set()
    for e in eps:
        if e == held:
            continue
        ivs = llm_iv(e) if source == "llm" else [(a, b) for a, b, t in eps[e]["gt"] if t == "ad"]
        for i, l in enumerate(eps[e]["lines"]):
            if source == "raw" or in_iv(l, ivs):
                db |= LG[e][i]
    return db


for source in ("raw", "llm", "gt"):
    scores, covered, total = [], 0, 0
    for e in eps:
        db = db_for(e, source)
        flag = [len(g) > 0 and len(g & db) / len(g) >= FRAC for g in LG[e]]
        iv = adtest.to_intervals(eps[e]["lines"], flag, eps[e]["dur"])
        scores.append(adtest.score(iv, eps[e]["gt"]))
        for x in llm_iv(e):                       # LLM breaks the match alone already covers (>=50%)
            total += 1
            covered += sum(adtest.overlap(x, q) for q in iv) >= 0.5 * (x[1] - x[0])
    print(f"{source:4s} n={N} frac={FRAC}: {adtest.summarize(scores)}  LLM breaks covered {covered}/{total}")
