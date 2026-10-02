"""Re-score every saved run against the current gt/ labels (labels were corrected after some runs finished)."""
import json, os, sys
import adtest
rows = []
for run in sorted(os.listdir(f"{adtest.ROOT}/results")):
    d = f"{adtest.ROOT}/results/{run}"
    if not os.path.isdir(d) or (run.endswith("_v2") or "_v2_" in run) != (os.environ.get("GT_DIR") == "gt_v2"):
        continue
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    if not files or (len(files) < 6 and "--all" not in sys.argv):
        continue
    scores, stats, audio = [], {}, 0
    for f in files:
        r = json.load(open(f"{d}/{f}"))
        dur, gt = adtest.load_gt(f[:-5])
        scores.append(adtest.score([(adtest.mmss(s), adtest.mmss(e)) for s, e in r["pred"]], gt))
        audio += dur or 1
        for k, v in r.get("stats", {}).items():
            stats[k] = stats.get(k, 0) + v
    s = adtest.summarize(scores)
    import random
    rnd = random.Random(0)
    boots = sorted(adtest.summarize([rnd.choice(scores) for _ in scores])["f1"] for _ in range(2000))
    s["ci"] = f"{boots[50]:.2f}-{boots[1949]:.2f}"
    h = audio / 3600
    rows.append((s["f1"], run, s, round(stats.get("calls", 0) / h), round(stats.get("in_tokens", 0) / h), round(stats.get("out_tokens", 0) / h), len(files)))
print(f"{'run':62s} {'eps':>3s} {'P':>5s} {'R':>5s} {'F1':>5s} {'95% CI':>9s} {'breaks':>6s} {'FA':>3s} {'calls/h':>7s} {'in/h':>6s} {'out/h':>5s}")
for f1, run, s, c, i, o, n in sorted(rows, reverse=True):
    print(f"{run[:62]:62s} {n:3d} {s['precision']:5.3f} {s['recall']:5.3f} {s['f1']:5.3f} {s['ci']:>9s} {s['breaks_found']:>6s} {s['false_alarms']:3d} {c:7d} {i:6d} {o:5d}")
