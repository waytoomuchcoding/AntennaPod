"""Block-level prompt iteration: accuracy of single block decisions on a fixed sample of blocks."""
import json, random, re, sys, time
import adtest, llm as llm_mod

PROMPTS = {}

PROMPTS["product"] = adtest.BLOCK_PROMPT_PRODUCT

PROMPTS["nocontext"] = """Here is a 30-second part of the transcript of the podcast "{podcast}", episode "{title}":

{current}

Podcasts are interrupted by advertisements: a sponsor message or ad that tries to get the listener to buy, try, download, subscribe to or listen to a specific product, service, app, brand or other show, usually with a call to action such as a website, a promo code or "try it free". Reporting, interviews or discussion about companies, money or products is NOT an advertisement.

Does this part contain an advertisement? If it does, answer with the name of the advertised product or brand. If it does not, answer NONE. Answer with the name or NONE only."""

PROMPTS["listfirst"] = """Here is a 30-second part of the transcript of the podcast "{podcast}", episode "{title}", with the text just before and after it for context.

[before] {before}

[PART] {current}

[after] {after}

Task: decide whether [PART] contains any advertising. Advertising means any sponsor message or commercial that promotes a product, service, company, app, website or another podcast or show to the listener (for example "visit example.com", "use code", "try it free", "listen wherever you get your podcasts", "terms apply"). The podcast's own story, reporting, interviews or discussion is not advertising, even when it mentions companies.

If [PART] contains advertising, answer with the advertised brand. Otherwise answer NONE."""

PROMPTS["system"] = None  # uses SYSTEM + nocontext body


SYSTEM = ("You detect advertisements in podcast transcripts. An advertisement is a sponsor message or commercial "
          "that promotes a product, service, brand, website, app or another show to the listener. The podcast's own "
          "content (story, reporting, interviews, discussion, intro, credits) is not an advertisement, even when it "
          "talks about companies or products. Answer with the advertised brand, or NONE.")


def sample(asr, episodes, n_content=70, seed=0):
    rnd = random.Random(seed)
    items = []
    for ep in episodes:
        dur, gt = adtest.load_gt(ep)
        lines = adtest.load_lines(asr, ep)
        blocks = adtest.blocks_of(lines, 30)
        for k, b in enumerate(blocks):
            s, e = lines[b[0]]["t"], lines[b[-1]]["e"]
            frac = sum(adtest.overlap((s, e), (g0, g1)) for g0, g1, t in gt if t == "ad") / max(e - s, 1)
            neutral = sum(adtest.overlap((s, e), (g0, g1)) for g0, g1, t in gt if t != "ad")
            if neutral > 0 or 0.1 < frac < 0.5:
                continue
            items.append({"ep": ep, "k": k, "label": frac >= 0.5,
                          "before": " ".join(lines[i]["text"] for i in blocks[k - 1]) if k else "(start of episode)",
                          "current": " ".join(lines[i]["text"] for i in b),
                          "after": " ".join(lines[i]["text"] for i in blocks[k + 1]) if k + 1 < len(blocks) else "(end of episode)",
                          **adtest.EPISODES[ep]})
    ads = [x for x in items if x["label"]]
    content = [x for x in items if not x["label"]]
    return ads + rnd.sample(content, min(n_content, len(content)))


def is_none(out):
    return not out.strip() or out.strip().upper().strip(".*\"' []").startswith("NONE")


def run(model, prompt, asr="moonshine", episodes="daily1,daily2,dateline1,dateline2,sysk1,crimejunkie1,freak1"):
    m = llm_mod.load(model)
    items = sample(asr, episodes.split(","))
    tp = fp = fn = tn = 0
    errors = []
    t0 = time.time()
    for x in items:
        if prompt == "system":
            out = m.generate(PROMPTS["nocontext"].format(**x), max_out=16, system=SYSTEM)
        else:
            out = m.generate(PROMPTS[prompt].format(**x), max_out=16)
        pred = not is_none(out)
        tp += pred and x["label"]; fp += pred and not x["label"]; fn += (not pred) and x["label"]; tn += (not pred) and not x["label"]
        if pred != x["label"]:
            errors.append(f'{"FN" if x["label"] else "FP"} {x["ep"]}#{x["k"]} {out.strip()[:25]!r} | {x["current"][:110]}')
    res = {"model": m.name, "prompt": prompt, "n": len(items), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
           "recall": round(tp / max(tp + fn, 1), 3), "fp_rate": round(fp / max(fp + tn, 1), 3),
           "s_per_call": round((time.time() - t0) / len(items), 2)}
    print(json.dumps(res), flush=True)
    for e in errors:
        print("   ", e)
    with open("results/blockdev.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], *sys.argv[3:])
