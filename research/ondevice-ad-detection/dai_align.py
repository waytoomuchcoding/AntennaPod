"""Find dynamically inserted ads by aligning transcripts of two downloads of the same episode (different
User-Agents, fetch_variants.py). Content is identical across copies; inserted ads differ. Stretches of the base
copy (>= MIN_S seconds) with no counterpart in a variant copy are labelled "inserted ad".
Scores those automatic labels against the human labels of the base copy.
usage: GT_DIR=gt_v2 ./venv/bin/python dai_align.py daily1,conan1,... [variants ua1,ua2] [--write gt_auto]"""
import difflib, json, os, re, sys
import adtest

MIN_S = float(os.environ.get("MIN_S", 15))   # inserted ads are >= 15 s; shorter diffs are mostly ASR noise
GAP_S = 5.0
# MODE=all: a stretch must differ from every variant (ASR noise is random, inserted ads differ in each copy)
MODE = os.environ.get("MODE", "all")


def words(ep):
    out = []                                   # (word, time) with times spread evenly inside each ASR segment
    for l in open(f"{adtest.ROOT}/tx/moonshine/{ep}.jsonl"):
        s = json.loads(l)
        if "text" not in s:
            continue
        ws = [w for w in re.sub(r"[^a-z0-9' ]", " ", s["text"].lower()).split()]
        for i, w in enumerate(ws):
            out.append((w, s["t"] + (s["e"] - s["t"]) * (i + 0.5) / len(ws)))
    return out


def unmatched(base, var):
    sm = difflib.SequenceMatcher(None, [w for w, _ in base], [w for w, _ in var], autojunk=False)
    hit = [False] * len(base)
    for a, b, n in sm.get_matching_blocks():
        if n >= 4:                              # ignore tiny coincidental matches inside different ads
            for k in range(a, a + n):
                hit[k] = True
    iv, start = [], None
    for k, (w, t) in enumerate(base + [("", 1e9)]):
        miss = k < len(base) and not hit[k]
        if miss and start is None:
            start = t
        elif not miss and start is not None:
            iv.append((start, base[k - 1][1])); start = None
    return iv


def merge_raw(iv):
    out = []
    for s, e in sorted(iv):
        if out and s - out[-1][1] <= GAP_S:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def intersect(per):
    cur = per[0]
    for nxt in per[1:]:
        cur = [(max(a, c), min(b, d)) for a, b in cur for c, d in nxt if min(b, d) > max(a, c)]
    return cur


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s - out[-1][1] <= GAP_S:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return [(s, e) for s, e in out if e - s >= MIN_S]


if __name__ == "__main__":
    eps = sys.argv[1].split(",")
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ["ua1", "ua2"]
    write = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None
    scores = []
    for ep in eps:
        base = words(ep)
        per = [merge_raw(unmatched(base, words(f"{ep}__{v}"))) for v in variants
               if os.path.exists(f"{adtest.ROOT}/tx/moonshine/{ep}__{v}.jsonl")]
        iv = merge(intersect(per) if MODE == "all" else sum(per, []))
        dur, gt = adtest.load_gt(ep)
        sc = adtest.score(iv, gt); scores.append(sc)
        ads = [(a, b) for a, b, t in gt if t == "ad"]
        missed = [f"{adtest.fmt(a)}-{adtest.fmt(b)}" for a, b in ads if sum(adtest.overlap(p, (a, b)) for p in iv) < 0.5 * (b - a)]
        fa = [f"{adtest.fmt(s)}-{adtest.fmt(e)}" for s, e in iv
              if sum(adtest.overlap((s, e), (a, b)) for a, b, _ in gt) < 0.5 * (e - s)]
        print(f"{ep}: {adtest.summarize([sc])}\n   breaks not inserted (baked in or same ad in all copies): {missed}\n"
              f"   differing stretches that are not labelled ad: {fa}", flush=True)
        if write:
            os.makedirs(write, exist_ok=True)
            json.dump({"duration": dur, "segments": [{"s": adtest.fmt(s), "e": adtest.fmt(e), "type": "ad"} for s, e in iv]},
                      open(f"{write}/{ep}.json", "w"))
    print("ALL", adtest.summarize(scores))
