"""Files for LLM labelling by subagents (README 16).
  export: lab/in/<ep>.txt   numbered, timestamped transcript lines ("L12 [03:41] text")
  import: lab/out/<ep>.json (written by the labeller: {"segments": [{"from": 12, "to": 30, "type": "ad"}], ...})
          -> gt_<name>/<ep>.json in the usual label format (times from the line timestamps)
usage: ./venv/bin/python labtools.py export ep1,ep2,...   |   ./venv/bin/python labtools.py import gt_haiku [eps]
"""
import json, os, sys
import adtest

ROOT = adtest.ROOT


def export(ep):
    lines = adtest.load_lines("moonshine", ep)
    os.makedirs(f"{ROOT}/lab/in", exist_ok=True)
    m = adtest.EPISODES.get(ep, {})
    with open(f"{ROOT}/lab/in/{ep}.txt", "w") as f:
        f.write(f"# podcast: {m.get('podcast', '?')}\n# episode: {m.get('title', '?')}\n# lines: {len(lines)}\n")
        for i, l in enumerate(lines):
            f.write(f"L{i} [{adtest.fmt(l['t'])}] {l['text']}\n")


def import_(ep, out_dir):
    lines = adtest.load_lines("moonshine", ep)
    lab = json.load(open(f"{ROOT}/lab/out/{ep}.json"))
    dur = lines[-1]["e"] + 1 if lines else 0
    segs = []
    for s in lab.get("segments", []):
        a, b = int(s["from"]), int(s["to"])
        if not (0 <= a <= b < len(lines)) or s.get("type") not in ("ad", "self_promo"):
            continue
        end = lines[b + 1]["t"] if b + 1 < len(lines) else lines[b]["e"]
        segs.append({"s": adtest.fmt(lines[a]["t"]), "e": adtest.fmt(end), "type": s["type"]})
    os.makedirs(f"{ROOT}/{out_dir}", exist_ok=True)
    json.dump({"duration": dur, "segments": segs, "labeller": lab.get("labeller", "?")},
              open(f"{ROOT}/{out_dir}/{ep}.json", "w"))
    return len(segs)


if __name__ == "__main__":
    if sys.argv[1] == "export":
        for ep in sys.argv[2].split(","):
            export(ep)
    else:
        out = sys.argv[2]
        eps = sys.argv[3].split(",") if len(sys.argv) > 3 else [f[:-5] for f in os.listdir(f"{ROOT}/lab/out")]
        print(sum(import_(e, out) for e in eps), "segments imported")
