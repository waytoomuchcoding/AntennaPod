"""Re-score saved block runs with different post-processing, without calling the LLM again."""
import json, os, sys, itertools
import adtest

def load_run(run, asr):
    out = {}
    for f in sorted(os.listdir(f"{adtest.ROOT}/results/{run}")):
        ep = f[:-5]
        r = json.load(open(f"{adtest.ROOT}/results/{run}/{f}"))
        lines = adtest.load_lines(asr, ep)
        is_ad = [False] * len(lines)
        blocks = adtest.blocks_of(lines, 30)
        for e in r["log"]:
            if "block" in e:
                ad = e.get("ad", bool(e["out"]) and not e["out"].upper().strip(".*\"' ").startswith("NONE"))
                for i in blocks[e["block"]]:
                    is_ad[i] = ad
        out[ep] = (lines, is_ad, blocks)
    return out


def postprocess(lines, is_ad, blocks, duration, gap, pad, fill):
    is_ad = list(is_ad)
    if fill:
        bad = [is_ad[b[0]] for b in blocks]
        for k in range(1, len(blocks) - 1):
            if not bad[k] and bad[k - 1] and any(bad[k + 1:k + 1 + fill]):
                for i in blocks[k]:
                    is_ad[i] = True
    iv = adtest.to_intervals(lines, is_ad, duration, gap=gap)
    return [(max(0, s - pad), min(duration, e + pad)) for s, e in iv]


def evaluate(data, gap=20, pad=0, fill=0):
    scores = []
    for ep, (lines, is_ad, blocks) in data.items():
        duration, gt = adtest.load_gt(ep)
        scores.append(adtest.score(postprocess(lines, is_ad, blocks, duration, gap, pad, fill), gt))
    return adtest.summarize(scores)


if __name__ == "__main__":
    run, asr = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "moonshine"
    data = load_run(run, asr)
    print("episodes:", list(data))
    for gap, pad, fill in itertools.product([20, 45, 75], [0, 5, 10], [0, 1, 2]):
        print(gap, pad, fill, json.dumps(evaluate(data, gap, pad, fill)))
