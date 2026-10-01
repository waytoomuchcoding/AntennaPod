"""Case bench: run the copy strategy on single windows centred on known problem spots (and some tricky true ads),
so prompt/context variants can be compared in minutes instead of hours.
Usage: bench.py <model> <variant-name> [key=value ...]   (keys are strategy_quotes kwargs, plus budget=)"""
import json, os, sys, time
import adtest, llm as llm_mod

CASES = [
    # (episode, centre "mm:ss", kind, note)
    ("daily1", "27:20", "FN", "Rinse narrative ad at the end"),
    ("daily1", "09:25", "FP", "hackers quoted: 'Hey, look, we extort victims'"),
    ("daily2", "02:20", "FP", "reporting that plays Kalshi TV ad clips"),
    ("freak1", "56:30", "FP", "hosts plug their own TV show in the outro"),
    ("morbid1", "04:00", "FP", "host chat about Halloween costumes / KIFF"),
    ("morbid1", "1:21:00", "FP", "koala chat in the outro, then final ad"),
    ("conan1", "71:00", "FP", "credits and subscribe plug before final ads"),
    ("dateline4", "41:00", "FP", "social-media plug before final ad"),
    ("daily1", "14:00", "TP", "stacked mid-roll: YouTube, Link, Rinse"),
    ("crimejunkie1", "14:00", "TP", "host-read movie promos (Carrie, Verity)"),
    ("conan1", "62:30", "TP", "conversational host-read ads (T-Mobile, Alka-Seltzer)"),
    ("huberman1", "21:40", "TP", "host-read BetterHelp in a science conversation"),
    ("dateline1", "15:50", "TP", "Dateline break with podcast promo"),
]


def window_lines(llm, ep, centre, budget):
    """Contiguous lines around `centre` whose transcript fits in `budget` tokens."""
    lines = adtest.load_lines("moonshine", ep)
    c = adtest.mmss(centre)
    mid = min(range(len(lines)), key=lambda i: abs(lines[i]["t"] - c))
    toks = [llm.count(l["text"]) + 1 for l in lines]
    lo = hi = mid
    total = toks[mid]
    while True:
        grew = False
        for side in (-1, 1):
            j = lo - 1 if side < 0 else hi + 1
            if 0 <= j < len(lines) and total + toks[j] <= budget:
                total += toks[j]
                lo, hi = (j, hi) if side < 0 else (lo, j)
                grew = True
        if not grew:
            return lines[lo:hi + 1]


def main(model, variant, *args):
    kw = {"mode": "copy", "iters": 6, "min_words": 5}
    for a in args:
        k, v = a.split("=", 1)
        kw[k] = int(v) if v.lstrip("-").isdigit() else float(v) if v.replace(".", "", 1).isdigit() else v
    budget = int(kw.pop("budget", 1200))
    region_default = int(kw.pop("region", 0))
    llm = llm_mod.load(model)
    llm.thinking = int(kw.pop("think", 0))
    if kw.pop("dump", 0):
        os.environ["DUMP_PROMPTS"] = f"{adtest.ROOT}/bench/dump_{variant}.jsonl"
        open(os.environ["DUMP_PROMPTS"], "w").close()
    os.makedirs(f"{adtest.ROOT}/bench", exist_ok=True)
    rows, t0 = [], time.time()
    case_filter = os.environ.get("BENCH_ONLY")
    for ep, centre, kind, note in CASES:
        if case_filter and case_filter not in f"{ep} {centre}":
            continue
        region = int(kw.pop("region", 0)) if "region" in kw else region_default
        lines = window_lines(llm, ep, centre, region or budget)
        focus = int(kw.get("focus", 0))
        only = None
        if focus:
            only = (lines[0]["t"], lines[-1]["t"])
            lines = window_lines(llm, ep, centre, budget + 2 * focus)
        log = []
        if os.environ.get("DUMP_PROMPTS"):
            with open(os.environ["DUMP_PROMPTS"], "a") as f:
                f.write(json.dumps({"case": f"{ep} {centre} {kind}: {note}"}) + "\n")
        is_ad = adtest.strategy_quotes(llm, lines, adtest.EPISODES[ep], budget=budget if region else 10 ** 6, log=log,
                                       episode=ep, only=only, **kw)
        if only:
            keep = [i for i, l in enumerate(lines) if only[0] <= l["t"] <= only[1]]
            lines = [lines[i] for i in keep]
            is_ad = [is_ad[i] for i in keep]
        lo, hi = lines[0]["t"], lines[-1]["e"]
        pred = adtest.to_intervals(lines, is_ad, hi)
        _, gt = adtest.load_gt(ep)
        ads = [(max(a, lo), min(b, hi)) for a, b, t in gt if t == "ad" and b > lo and a < hi]
        neutral = [(max(a, lo), min(b, hi)) for a, b, t in gt if t != "ad" and b > lo and a < hi]
        tp = sum(adtest.overlap(p, g) for p in pred for g in ads)
        fp = sum(p[1] - p[0] for p in pred) - tp - sum(adtest.overlap(p, n) for p in pred for n in neutral)
        fn = sum(b - a for a, b in ads) - tp
        ok = fp < 10 and fn < 10
        rows.append({"case": f"{ep} {centre}", "kind": kind, "fp_s": round(max(fp, 0)), "fn_s": round(max(fn, 0)),
                     "ok": ok, "pred": [[adtest.fmt(s), adtest.fmt(e)] for s, e in pred],
                     "outs": [l["out"][:80] for l in log]})
        print(f"{'PASS' if ok else 'FAIL'} {kind} {ep:12s} {centre:>7s} fp={max(fp,0):4.0f}s fn={max(fn,0):4.0f}s  {note}", flush=True)
    n_ok = sum(r["ok"] for r in rows)
    summary = {"variant": variant, "model": llm.name, "budget": budget, "kw": kw, "passed": f"{n_ok}/{len(rows)}",
               "fp_s": sum(r["fp_s"] for r in rows), "fn_s": sum(r["fn_s"] for r in rows),
               "calls": llm.stats["calls"], "out_tokens": llm.stats["out_tokens"],
               "think_tokens": llm.stats.get("think_tokens", 0), "minutes": round((time.time() - t0) / 60, 1)}
    print("BENCH", json.dumps(summary))
    with open(f"{adtest.ROOT}/bench/results.jsonl", "a") as f:
        f.write(json.dumps({**summary, "rows": rows}) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
