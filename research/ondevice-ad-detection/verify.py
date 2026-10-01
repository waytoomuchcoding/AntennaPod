"""Verification pass over a saved run: ask the model once per predicted interval whether its text is an ad;
drop intervals it rejects. Writes a new run directory '<run>_verified'."""
import json, os, sys, time
import adtest, llm as llm_mod

VERIFY_PROMPT = adtest.VERIFY_PROMPT


def main(model, run, asr="moonshine", max_words=350):
    m = llm_mod.load(model)
    out_run = run + "_verified"
    os.makedirs(f"{adtest.ROOT}/results/{out_run}", exist_ok=True)
    scores = []
    for f in sorted(os.listdir(f"{adtest.ROOT}/results/{run}")):
        ep = f[:-5]
        r = json.load(open(f"{adtest.ROOT}/results/{run}/{f}"))
        lines = adtest.load_lines(asr, ep)
        dur, gt = adtest.load_gt(ep)
        before = dict(m.stats)
        kept, log = [], []
        for s, e in r["pred"]:
            s, e = adtest.mmss(s), adtest.mmss(e)
            words = " ".join(l["text"] for l in lines if l["t"] >= s - 1 and l["t"] < e).split()
            if len(words) > max_words:
                words = words[:max_words // 2] + ["..."] + words[-max_words // 2:]
            out = m.generate(VERIFY_PROMPT.format(current=" ".join(words), **adtest.EPISODES[ep]), max_out=16,
                             system=adtest.SYSTEM)
            ok = bool(out.strip()) and not out.strip().upper().strip(".*\"' ").startswith("NONE")
            log.append({"interval": [adtest.fmt(s), adtest.fmt(e)], "out": out.strip(), "keep": ok})
            if ok:
                kept.append((s, e))
        sc = adtest.score(kept, gt)
        scores.append(sc)
        st = {k: round(m.stats[k] - before[k], 1) for k in before}
        r2 = dict(r, pred=[[adtest.fmt(s), adtest.fmt(e)] for s, e in kept], score=sc,
                  stats={k: round(r["stats"][k] + st[k], 1) for k in st}, verify_log=log)
        json.dump(r2, open(f"{adtest.ROOT}/results/{out_run}/{f}", "w"), indent=1)
        print(ep, json.dumps(adtest.summarize([sc])), [x for x in log if not x["keep"]], flush=True)
    summ = adtest.summarize(scores)
    summ["run"] = out_run
    print("SUMMARY", json.dumps(summ))
    with open(f"{adtest.ROOT}/results/summary.jsonl", "a") as fh:
        fh.write(json.dumps(summ) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
