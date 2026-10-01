"""Vote across several saved runs: a second of audio is an ad when at least k runs marked it.
Usage: vote.py <out_run> <k> <run1> <run2> ... ; writes results/<out_run>/ so verify.py can run on it."""
import json, os, sys
import adtest


def main(out_run, k, *runs):
    k = int(k)
    os.makedirs(f"{adtest.ROOT}/results/{out_run}", exist_ok=True)
    eps = sorted(set.intersection(*(set(os.listdir(f"{adtest.ROOT}/results/{r}")) for r in runs)))
    scores = []
    for f in eps:
        ep = f[:-5]
        dur, gt = adtest.load_gt(ep)
        votes = [0] * (int(dur) + 1)
        stats = {}
        for r in runs:
            res = json.load(open(f"{adtest.ROOT}/results/{r}/{f}"))
            for key, v in res["stats"].items():
                stats[key] = round(stats.get(key, 0) + v, 1)
            for s, e in res["pred"]:
                for t in range(int(adtest.mmss(s)), min(int(adtest.mmss(e)), int(dur)) + 1):
                    votes[t] += 1
        iv = [(t, t + 1) for t, v in enumerate(votes) if v >= k]
        pred = [x for x in adtest.merge(iv, 20) if x[1] - x[0] >= 10]
        sc = adtest.score(pred, gt)
        scores.append(sc)
        json.dump({"episode": ep, "pred": [[adtest.fmt(s), adtest.fmt(e)] for s, e in pred], "score": sc,
                   "stats": stats, "audio_s": dur, "runs": list(runs), "k": k, "log": []},
                  open(f"{adtest.ROOT}/results/{out_run}/{f}", "w"), indent=1)
    summ = adtest.summarize(scores)
    summ["run"] = out_run
    print(json.dumps(summ))


if __name__ == "__main__":
    main(*sys.argv[1:])
