"""Combine several LLM labelling runs per line (README 16.2). Each run: <dir>/<ep>.json with line ranges.
Per line: ad if at least K runs say ad; self_promo if at least K say ad or self_promo; else content.
Writes lab/out_vote/<ep>.json (same format) and, with --score, compares single runs, union, majority and
intersection against human labels (GT_DIR).
usage: GT_DIR=gt_v2 ./venv/bin/python vote_labels.py lab/out,lab/out_r2,lab/out_r3 ep1,ep2,... [--score] [--k 2]"""
import json, os, sys
import adtest

ROOT = adtest.ROOT
dirs = sys.argv[1].split(",")
eps = sys.argv[2].split(",")
K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else (len(dirs) // 2 + 1)


def per_line(path, n):
    lab = [0] * n
    for s in json.load(open(path)).get("segments", []):
        a, b, t = int(s["from"]), int(s["to"]), s.get("type")
        for i in range(max(a, 0), min(b, n - 1) + 1):
            lab[i] = max(lab[i], 2 if t == "ad" else 1 if t == "self_promo" else 0)
    return lab


def to_segments(lab):
    segs, i = [], 0
    while i < len(lab):
        if lab[i]:
            j = i
            while j + 1 < len(lab) and lab[j + 1] == lab[i]:
                j += 1
            segs.append({"from": i, "to": j, "type": "ad" if lab[i] == 2 else "self_promo"})
            i = j + 1
        else:
            i += 1
    return segs


def combine(runs, k):
    return [2 if sum(r[i] == 2 for r in runs) >= k else 1 if sum(r[i] >= 1 for r in runs) >= k else 0
            for i in range(len(runs[0]))]


def intervals(lines, lab):
    return adtest.to_intervals(lines, [x == 2 for x in lab], lines[-1]["e"] + 1, gap=0, min_len=0)


os.makedirs(f"{ROOT}/lab/out_vote", exist_ok=True)
rows = {}
for ep in eps:
    lines = adtest.load_lines("moonshine", ep)
    runs = [per_line(f"{ROOT}/{d}/{ep}.json", len(lines)) for d in dirs if os.path.exists(f"{ROOT}/{d}/{ep}.json")]
    if len(runs) < len(dirs):
        print(ep, "missing runs", len(runs)); continue
    vote = combine(runs, K)
    json.dump({"labeller": f"haiku x{len(runs)} vote>={K}", "segments": to_segments(vote)},
              open(f"{ROOT}/lab/out_vote/{ep}.json", "w"))
    if "--score" in sys.argv:
        gt = adtest.load_gt(ep)[1]
        variants = {f"run {i + 1}": r for i, r in enumerate(runs)}
        variants.update({"union (>=1)": combine(runs, 1), f"majority (>={K})": vote,
                         f"all ({len(runs)})": combine(runs, len(runs))})
        for name, lab in variants.items():
            rows.setdefault(name, []).append(adtest.score(intervals(lines, lab), gt))
for name, sc in rows.items():
    s = adtest.summarize(sc)
    left = sum(x["gt"] - x["tp"] for x in sc) / sum(x["breaks"] for x in sc)
    skip = sum(x["pred"] - x["tp"] for x in sc) / sum(x["breaks"] for x in sc)
    print(f"{name:14s} P {s['precision']:.3f} R {s['recall']:.3f} F1 {s['f1']:.3f} breaks {s['breaks_found']} "
          f"FA {s['false_alarms']} | per break: ad left {left:.1f}s, content skipped {skip:.1f}s")
