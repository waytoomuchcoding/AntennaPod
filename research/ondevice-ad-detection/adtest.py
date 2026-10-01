"""On-device ad detection experiments: transcript (from transcribe.py) -> small LLM -> ad intervals -> score."""
import argparse, json, os, re, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
EPISODES = json.load(open(f"{ROOT}/episodes.json"))


def mmss(x):
    if isinstance(x, (int, float)):
        return float(x)
    parts = [float(p) for p in x.split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]


def load_gt(ep):
    g = json.load(open(f"{ROOT}/gt/{ep}.json"))
    return g["duration"], [(mmss(s["s"]), mmss(s["e"]), s["type"]) for s in g["segments"]]


def load_lines(asr, ep, max_words=25, max_gap=1.5):
    """Merge VAD segments into lines of up to ~max_words words; a line never spans a pause > max_gap s."""
    lines, cur = [], None
    for l in open(f"{ROOT}/tx/{asr}/{ep}.jsonl"):
        s = json.loads(l)
        if cur and s["t"] - cur["e"] <= max_gap and len(cur["text"].split()) + len(s["text"].split()) <= max_words:
            cur["text"] += " " + s["text"]
            cur["e"] = s["e"]
        else:
            if cur:
                lines.append(cur)
            cur = dict(s)
    if cur:
        lines.append(cur)
    return lines


def fmt(t):
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


def merge(iv, gap):
    out = []
    for s, e in sorted(iv):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


# ---------------------------------------------------------------- strategy: ranges
RANGES_PROMPT = """Below is part of the transcript of the podcast "{podcast}", episode "{title}". Each line starts with a line number and a timestamp.

Find the advertisements in it: sponsor messages, host-read ads, ads for products or services, promos for other podcasts or shows, and network announcements. Several ads usually play back to back; report such an ad break as one range. Do not report the episode's own content, such as its story, interviews, intro, teasers or credits.

Answer with one line per ad break in the form FIRST-LAST, using the line numbers of the first and last line of the break. If there are no ads, answer NONE.

Transcript:
{transcript}"""


def strategy_ranges(llm, lines, meta, budget=2600, overlap=0.3, log=None):
    """Windows of up to `budget` transcript tokens, overlapping by `overlap`. A line is an ad if any window says so."""
    toks = [llm.count(f"{i} [{fmt(l['t'])}] {l['text']}\n") for i, l in enumerate(lines)]
    is_ad = [False] * len(lines)
    start = 0
    while start < len(lines):
        end, total = start, 0
        while end < len(lines) and total + toks[end] <= budget:
            total += toks[end]
            end += 1
        tx = "".join(f"{i - start + 1} [{fmt(lines[i]['t'])}] {lines[i]['text']}\n" for i in range(start, end))
        out = llm.generate(RANGES_PROMPT.format(transcript=tx, **meta))
        if log is not None:
            log.append({"window": [start, end], "out": out})
        for a, b in re.findall(r"(\d+)\s*[-–]\s*(\d+)", out):
            a, b = int(a) - 1 + start, int(b) - 1 + start
            for i in range(max(a, start), min(b, end - 1) + 1):
                is_ad[i] = True
        if end >= len(lines):
            break
        n = end - start
        start = start + max(1, int(n * (1 - overlap)))
    return is_ad


# ---------------------------------------------------------------- strategy: quotes
COPY_PROMPT = """Below is part of the transcript of the podcast "{podcast}", episode "{title}".

<transcript>
{transcript}
</transcript>

Find the first advertisement in this transcript and copy its complete text, word for word, exactly as it is written in the transcript. Output only the copied advertisement text. If the transcript contains no advertisement, answer NONE."""

COPY_FOCUS_PROMPT = """Below is part of the transcript of the podcast "{podcast}", episode "{title}". Only the part between <search> and </search> needs to be checked; the text around it is there for context.

<transcript>
{transcript}
</transcript>

Find the first advertisement inside the <search> part and copy its complete text, word for word, exactly as it is written in the transcript. Output only the copied advertisement text. If the <search> part contains no advertisement, answer NONE."""

QUOTES_PROMPT = """Below is part of the transcript of the podcast "{podcast}", episode "{title}".

<transcript>
{transcript}
</transcript>

Find every advertisement in this transcript. For each one, copy its first few words and its last few words exactly as they are written in the transcript, in this format:
START: <first 5 to 8 words of the ad> || END: <last 5 to 8 words of the ad>

Write one line per advertisement, in order. Ads that play back to back may be reported as one. If the transcript contains no advertisement, answer NONE."""

QUOTES_SYSTEM = ("You find advertisements in podcast transcripts. An advertisement is a sponsor message or commercial "
                 "that promotes a product, service, brand, website, app or another podcast or show to the listener, "
                 "usually with a call to action such as a web address, a promo code, a free trial or where to listen. "
                 "The podcast's own content (story, reporting, interviews, discussion, intro, credits) is not an "
                 "advertisement, even when it talks about companies or products. You only copy words that appear "
                 "in the transcript.")


VERIFY_PROMPT = """Here is a part of the transcript of the podcast "{podcast}", episode "{title}":

{current}

Podcasts are interrupted by advertisements: a sponsor message or ad that tries to get the listener to buy, try, download, subscribe to or listen to a specific product, service, app, brand or other show, usually with a call to action such as a website, a promo code or "try it free". Reporting, interviews or discussion about companies, money or products is NOT an advertisement.

Does this part contain an advertisement? If it does, answer with the name of the advertised product or brand. If it does not, answer NONE. Answer with the name or NONE only."""


def _norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def find_quote(words, quote, lo, hi, from_end=False):
    """Index into `words` (within [lo, hi)) where `quote` best matches, by fraction of quote words matched in order."""
    q = [_norm(w) for w in quote.split() if _norm(w)]
    if not q:
        return None, 0
    best, best_score = None, 0.0
    n = len(q)
    for i in range(lo, max(lo, hi - 1) + 1):
        seg = words[i:i + n + 2]
        j = hits = 0
        first = last = None
        for w in q:
            for k in range(j, min(len(seg), j + 3)):
                if seg[k] == w:
                    hits += 1
                    j = k + 1
                    first = k if first is None else first
                    last = k
                    break
        s = hits / n
        if s > best_score + 1e-9:
            best, best_score = (i + last if from_end else i + first), s
    return best, best_score


ANNOT_SYSTEM = (" The transcript also contains annotations that are not spoken words: \"S1:\", \"S2:\" and so on "
                "mark who is speaking (S1 is the voice heard most in the whole episode, S2 the second most, and so on), "
                "[pause Ns] marks N seconds of silence and [music] marks music. Advertisements often begin after a pause "
                "or with music and are often read by a voice that is rarely heard elsewhere in the episode, but hosts "
                "sometimes read ads too. Never copy annotations.")


def line_annotations(episode, lines, annot):
    """Per-line prefix with speaker changes, pauses and music, from diar/<ep>.json and feats/<ep>.json."""
    pre = [""] * len(lines)
    if not annot:
        return pre
    spk = [None] * len(lines)
    if "s" in annot:
        segs = json.load(open(f"{ROOT}/diar/{episode}.json"))
        talk = {}
        for s, e, k in segs:
            talk[k] = talk.get(k, 0) + e - s
        rank = {k: n + 1 for n, (k, _) in enumerate(sorted(talk.items(), key=lambda kv: -kv[1]))}
        for i, l in enumerate(lines):
            ov = {}
            for s, e, k in segs:
                o = min(e, l["e"]) - max(s, l["t"])
                if o > 0:
                    ov[k] = ov.get(k, 0) + o
            spk[i] = rank[max(ov, key=ov.get)] if ov else None
    music = json.load(open(f"{ROOT}/feats/{episode}.json"))["music"] if "m" in annot else None
    prev_spk, prev_music = None, False
    for i, l in enumerate(lines):
        p = ""
        gap = l["t"] - lines[i - 1]["e"] if i else 0
        if "g" in annot and gap >= 2:
            p += f"[pause {round(gap)}s]\n"
        if music is not None:
            lo = int(lines[i - 1]["e"]) if i else 0
            sec = music[lo:int(l["e"]) + 1]
            on = bool(sec) and sum(v > 0.5 for v in sec) >= max(1, len(sec) // 2)
            if on and not prev_music:
                p += "[music]\n"
            prev_music = on
        if spk[i] is not None and spk[i] != prev_spk:
            p += f"S{spk[i]}: "
        prev_spk = spk[i] if spk[i] is not None else prev_spk
        pre[i] = p
    line_annotations.speakers = spk
    return pre


SYSTEMS = {
    "v2": ("You find advertisements in podcast transcripts. An advertisement is a paid message from a sponsor or "
           "another company: a commercial or sponsor read that promotes a product, service, brand, website, app, "
           "film, TV show or another podcast to the listener. Hosts sometimes read ads themselves, often starting "
           "with phrases like \"this episode is sponsored by\", and some ads are told as a little story that only "
           "names the brand and its web address at the end. Ads usually end with a call to action such as a web "
           "address, a promo code, a free trial or where to listen.\n"
           "These are NOT advertisements: the episode's own story, reporting, interviews and conversation; hosts "
           "chatting about shows, books or products they personally like; ads or clips that the episode quotes or "
           "discusses as part of its reporting; the show's credits, sign-off, listener mail, plugs for the show's own "
           "social media, email, newsletter, YouTube channel or other episodes, and requests to subscribe or rate.\n"
           "You only copy words that appear in the transcript. If you are not sure that something is an "
           "advertisement, answer NONE."),
}


def strategy_quotes(llm, lines, meta, budget=1200, overlap=0.3, min_match=0.5, iters=1, min_words=3, max_span=120,
                    marker=1, mode="quotes", offset=0.0, verify_each=0, annot="", episode=None, sys="",
                    marker_fallback=0, focus=0, only=None, veto=0.0, max_vetoes=3, log=None):
    """Plain-text windows; the model quotes the first/last words of each ad; quotes are matched back to words.
    With iters > 1, found ads are replaced by a marker and the window is asked again until NONE or iters passes."""
    words, word_line = [], []
    for li, l in enumerate(lines):
        for w in l["text"].split():
            words.append(w)
            word_line.append(li)
    norm = [_norm(w) for w in words]
    removed = [False] * len(words)
    is_ad = [False] * len(lines)
    clfp = None
    if veto:
        probs = json.load(open(f"{ROOT}/clfp/{episode}.json"))
        clfp = [probs.get(str(l["t"]), 0.0) for l in lines]
    prefix = line_annotations(episode, lines, annot)
    speakers = getattr(line_annotations, "speakers", [None] * len(lines)) if "s" in annot else [None] * len(lines)
    system = SYSTEMS.get(sys, QUOTES_SYSTEM) + (ANNOT_SYSTEM if annot else "")
    toks_per_word = llm.count(" ".join(words[:2000])) / max(1, min(2000, len(words)))
    win = int(budget / toks_per_word)
    start = -int(win * offset)
    if only:
        start = next(i for i in range(len(words)) if lines[word_line[i]]["t"] >= only[0])
        win = next((i for i in range(len(words)) if lines[word_line[i]]["t"] > only[1]), len(words)) - start
    while start < len(words):
        end = min(len(words), start + win)
        rejects = 0
        for it in range(iters):
            visible = [i for i in range(max(0, start), end) if not removed[i]]
            if not visible:
                break
            def render(idx):
                t, prev, shown = "", None, set()
                for i in idx:
                    if prev is not None and i != prev + 1:
                        t += "\n[advertisement removed]\n" if marker else "\n"
                    elif prev is not None:
                        t += "\n" if word_line[i] != word_line[prev] else " "
                    if word_line[i] not in shown:
                        p = prefix[word_line[i]]
                        sp = speakers[word_line[i]]
                        if not shown and sp is not None and f"S{sp}: " not in p:
                            p += f"S{sp}: "
                        shown.add(word_line[i])
                        if not t.endswith(" "):
                            t += p
                    t += words[i]
                    prev = i
                return t
            tx = render(visible)
            if focus:
                fw = int(focus / toks_per_word)
                before = [i for i in range(max(0, max(0, start) - fw), max(0, start)) if not removed[i]]
                after = [i for i in range(end, min(len(words), end + fw)) if not removed[i]]
                tx = (render(before) + "\n" if before else "") + "<search>\n" + tx + "\n</search>" + \
                     ("\n" + render(after) if after else "")
                ctx_norm = [norm[i] for i in before + after]
            else:
                ctx_norm = []
            vnorm = [norm[i] for i in visible]
            prompt = (COPY_FOCUS_PROMPT if focus else COPY_PROMPT) if mode == "copy" else QUOTES_PROMPT
            out = llm.generate(prompt.format(transcript=tx, **meta), system=system)
            if marker_fallback and "advertisement removed" in out.lower() and \
                    not re.sub(r"\[[^\]]*\]", " ", out).strip():
                out = llm.generate(prompt.format(transcript=tx.replace("\n[advertisement removed]\n", "\n"), **meta),
                                   system=system)
            entry = {"window": [max(0, start), end], "iter": it, "out": out, "matches": []}
            pos, found, stop = 0, 0, False
            if mode == "copy":
                clean = re.sub(r"\bS\d+:", " ", re.sub(r"\[[^\]]*\]|</?search>", " ", out))
                cw = clean.replace("\n", " ").split()
                pairs = [] if not cw or out.strip().upper().strip(".*\"' ").startswith("NONE") else \
                    [(" ".join(cw[:8]), " ".join(cw[-8:]))]
            else:
                pairs = [mm.groups() for mm in (re.search(r"START:\s*(.*?)\s*\|\|?\s*END:\s*(.*)", line, re.I)
                                                for line in out.splitlines()) if mm]
            for pair in pairs:
                m = type("M", (), {"group": staticmethod(lambda k, p=pair: p[k - 1])})
                a, sa = find_quote(vnorm, m.group(1).strip(" \"'<>"), pos, len(vnorm))
                if a is None or sa < min_match:
                    entry["matches"].append([m.group(1)[:40], "nomatch-start", round(sa, 2)])
                    continue
                if focus and ctx_norm:
                    _, sc = find_quote(ctx_norm, m.group(1).strip(" \"'<>"), 0, len(ctx_norm))
                    if sc >= sa:
                        entry["matches"].append([m.group(1)[:40], "from-context", round(sc, 2)])
                        continue
                hi = min(len(vnorm), a + int(1.5 * len(cw)) + 5) if mode == "copy" else len(vnorm)
                b, sb = find_quote(vnorm, m.group(2).strip(" \"'<>"), a, hi, from_end=True)
                if b is None or sb < min_match:
                    entry["matches"].append([m.group(1)[:40], "nomatch-end", round(sb, 2)])
                    continue
                ga, gb = visible[a], visible[min(b, len(visible) - 1)]
                short = min(len([w for w in m.group(k).split() if _norm(w)]) for k in (1, 2)) < min_words
                if short or lines[word_line[gb]]["e"] - lines[word_line[ga]]["t"] > max_span:
                    entry["matches"].append([m.group(1)[:40], "rejected", fmt(lines[word_line[ga]]["t"]),
                                             fmt(lines[word_line[gb]]["e"])])
                    continue
                if verify_each:
                    span = words[ga:gb + 1]
                    if len(span) > 350:
                        span = span[:175] + ["..."] + span[-175:]
                    vout = llm.generate(VERIFY_PROMPT.format(current=" ".join(span), **meta), max_out=16,
                                        system=SYSTEM)
                    if not vout.strip() or vout.strip().upper().strip(".*\"' ").startswith("NONE"):
                        entry["matches"].append([fmt(lines[word_line[ga]]["t"]), "rejected-verify",
                                                 fmt(lines[word_line[gb]]["e"])])
                        for i in range(ga, gb + 1):
                            removed[i] = True
                        rejects += 1
                        found += 1
                        stop = rejects >= verify_each
                        break
                if clfp is not None and max(clfp[word_line[ga]:word_line[gb] + 1]) < veto:
                    for i in range(ga, gb + 1):
                        removed[i] = True
                    entry["matches"].append([fmt(lines[word_line[ga]]["t"]), "vetoed", fmt(lines[word_line[gb]]["e"])])
                    rejects += 1
                    found += 1
                    stop = rejects >= max_vetoes
                    pos = b
                    continue
                for i in range(ga, gb + 1):
                    removed[i] = True
                for li in range(word_line[ga], word_line[gb] + 1):
                    is_ad[li] = True
                entry["matches"].append([fmt(lines[word_line[ga]]["t"]), fmt(lines[word_line[gb]]["e"]),
                                         round(sa, 2), round(sb, 2)])
                pos = b
                found += 1
            if log is not None:
                log.append(entry)
            if not found or stop:
                break
        if end >= len(words) or only:
            break
        start += max(1, int(win * (1 - overlap)))
    return is_ad


# ---------------------------------------------------------------- strategy: blocks
BLOCK_PROMPT = """You are finding advertisements in the podcast "{podcast}", episode "{title}". Below are three consecutive parts of its transcript.

BEFORE:
{before}

CURRENT:
{current}

AFTER:
{after}

Is the CURRENT part an advertisement? Advertisements include sponsor messages, host-read ads, ads for products or services, promos for other podcasts or shows, and network announcements. The episode's own content, such as its story, interviews, intro, teasers or credits, is not an advertisement. If the CURRENT part is partly an advertisement, answer AD.

Answer with one word: AD or CONTENT."""


def blocks_of(lines, block_s):
    blocks, cur = [], []
    for i, l in enumerate(lines):
        if cur and l["e"] - lines[cur[0]]["t"] > block_s:
            blocks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        blocks.append(cur)
    return blocks


BLOCK_PROMPT_PRODUCT = """Below are three consecutive parts of the transcript of the podcast "{podcast}", episode "{title}".

BEFORE:
{before}

CURRENT:
{current}

AFTER:
{after}

Podcasts are interrupted by advertisements: a sponsor message or ad that tries to get the listener to buy, try, download, subscribe to or listen to a specific product, service, app, brand or other show, usually with a call to action such as a website, a promo code or "try it free". Reporting, interviews or discussion about companies, money or products is NOT an advertisement.

Does the CURRENT part contain an advertisement? If it does, answer with the name of the advertised product or brand. If it does not, answer NONE. Answer with the name or NONE only."""

BLOCK_PROMPT_SYSTEM = """Here is a 30-second part of the transcript of the podcast "{podcast}", episode "{title}":

{current}

Podcasts are interrupted by advertisements: a sponsor message or ad that tries to get the listener to buy, try, download, subscribe to or listen to a specific product, service, app, brand or other show, usually with a call to action such as a website, a promo code or "try it free". Reporting, interviews or discussion about companies, money or products is NOT an advertisement.

Does this part contain an advertisement? If it does, answer with the name of the advertised product or brand. If it does not, answer NONE. Answer with the name or NONE only."""

SYSTEM = ("You detect advertisements in podcast transcripts. An advertisement is a sponsor message or commercial "
          "that promotes a product, service, brand, website, app or another show to the listener. The podcast's own "
          "content (story, reporting, interviews, discussion, intro, credits) is not an advertisement, even when it "
          "talks about companies or products. Answer with the advertised brand, or NONE.")

BLOCK_PROMPTS = {"yesno": BLOCK_PROMPT, "product": BLOCK_PROMPT_PRODUCT, "system": BLOCK_PROMPT_SYSTEM}


def strategy_blocks(llm, lines, meta, block_s=30, log=None, prompt="yesno", refine=0, reuse=None, episode=None):
    blocks = blocks_of(lines, block_s)
    text = lambda b: " ".join(lines[i]["text"] for i in b) if b else "(none)"
    is_ad = [False] * len(lines)
    if reuse:
        prev = json.load(open(f"{ROOT}/results/{reuse}/{episode}.json"))["log"]
        for entry in prev:
            if "block" in entry:
                out = entry["out"]
                ad = entry.get("ad", bool(out) and not out.upper().strip(".*\"' ").startswith("NONE"))
                for i in blocks[entry["block"]]:
                    is_ad[i] = ad
                if log is not None:
                    log.append(dict(entry, ad=ad))
        blocks_iter = []
    else:
        blocks_iter = list(enumerate(blocks))
    for k, b in blocks_iter:
        out = llm.generate(BLOCK_PROMPTS[prompt].format(before=text(blocks[k - 1]) if k else "(start of episode)",
                                               current=text(b),
                                               after=text(blocks[k + 1]) if k + 1 < len(blocks) else "(end of episode)",
                                               **meta), max_out=8 if prompt == "yesno" else 16,
                           system=SYSTEM if prompt == "system" else None)
        if prompt == "yesno":
            ad = "AD" in out.upper().replace("ADVERT", "AD")
        else:
            ad = bool(out.strip()) and not out.strip().upper().strip(".*\"' ").startswith("NONE")
        if log is not None:
            log.append({"block": k, "t": lines[b[0]]["t"], "out": out.strip()})
        for i in b:
            is_ad[i] = ad
        if log is not None:
            log[-1]["ad"] = ad
    if refine:
        refine_boundaries(llm, lines, blocks, is_ad, meta, log, refine)
    return is_ad


START_PROMPT = """Below are consecutive numbered lines from the transcript of the podcast "{podcast}", episode "{title}". Somewhere in these lines the episode's own content stops and an advertisement (a sponsor message or an ad for a product, service or other show) begins.

{numbered}

Which line is the first line of the advertisement? Answer with the line number only."""

END_PROMPT = """Below are consecutive numbered lines from the transcript of the podcast "{podcast}", episode "{title}". The first lines are an advertisement (a sponsor message or an ad for a product, service or other show). Somewhere in these lines the advertisements end and the episode's own content resumes.

{numbered}

Which line is the first line of the episode's own content after the advertisements? If the advertisements continue until the last line, answer 0. Answer with the line number only."""


def refine_boundaries(llm, lines, blocks, is_ad, meta, log, refine=1):
    """For each run of ad blocks, ask for the exact first ad line (among the block before and the first ad block)
    and the first content line after it (among the last ad block and the block after)."""
    block_ad = [is_ad[b[0]] for b in blocks]
    runs, k = [], 0
    while k < len(blocks):
        if block_ad[k]:
            s = k
            while k + 1 < len(blocks) and block_ad[k + 1]:
                k += 1
            runs.append((s, k))
        k += 1
    numbered = lambda idx: "\n".join(f"{n + 1}. {lines[i]['text']}" for n, i in enumerate(idx))
    for s, e in runs:
        if s > 0 and refine in (1, 2):
            idx = blocks[s - 1] + blocks[s]
            out = llm.generate(START_PROMPT.format(numbered=numbered(idx), **meta), max_out=8)
            m = re.search(r"\d+", out)
            if m and 1 <= int(m.group()) <= len(idx):
                first = int(m.group()) - 1
                for n, i in enumerate(idx):
                    is_ad[i] = n >= first
            if log is not None:
                log.append({"refine_start": lines[idx[0]]["t"], "out": out.strip()})
        if e + 1 < len(blocks) and refine in (1, 3):
            idx = blocks[e] + blocks[e + 1]
            out = llm.generate(END_PROMPT.format(numbered=numbered(idx), **meta), max_out=8)
            m = re.search(r"\d+", out)
            if m and 1 <= int(m.group()) <= len(idx):
                resume = int(m.group()) - 1
                for n, i in enumerate(idx):
                    is_ad[i] = n < resume
            elif m and int(m.group()) == 0:
                for i in idx:
                    is_ad[i] = True
            if log is not None:
                log.append({"refine_end": lines[idx[0]]["t"], "out": out.strip()})


# ---------------------------------------------------------------- lines -> intervals, scoring
TIGHT_ENDS = False


def to_intervals(lines, is_ad, duration, gap=20, min_len=10):
    iv = []
    for i, l in enumerate(lines):
        if is_ad[i]:
            end = lines[i + 1]["t"] if i + 1 < len(lines) else duration
            if TIGHT_ENDS:
                end = min(end, l["e"] + 2.0)
            iv.append((l["t"], end))
    return [x for x in merge(iv, gap) if x[1] - x[0] >= min_len]


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def score(pred, gt):
    """Seconds-based precision/recall for ads. Self-promo regions are neutral (neither required nor penalized).
    Also break-level: a GT ad break counts as found when >=50% of it is covered; a predicted interval is a
    false alarm when <50% of it overlaps any GT ad/promo."""
    ads = [(s, e) for s, e, t in gt if t == "ad"]
    neutral = [(s, e) for s, e, t in gt if t != "ad"]
    tp = sum(overlap(p, g) for p in pred for g in ads)
    pred_len = sum(p[1] - p[0] for p in pred) - sum(overlap(p, n) for p in pred for n in neutral)
    gt_len = sum(e - s for s, e in ads)
    found = sum(1 for g in ads if sum(overlap(p, g) for p in pred) >= 0.5 * (g[1] - g[0]))
    false_alarms = sum(1 for p in pred if sum(overlap(p, g) for g in ads + neutral) < 0.5 * (p[1] - p[0]))
    return {"tp": tp, "pred": max(pred_len, 0), "gt": gt_len, "breaks": len(ads), "found": found,
            "false_alarms": false_alarms}


def summarize(scores):
    tp = sum(s["tp"] for s in scores)
    pr = sum(s["pred"] for s in scores)
    gt = sum(s["gt"] for s in scores)
    p = tp / pr if pr else 0
    r = tp / gt if gt else 0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0,
            "breaks_found": f"{sum(s['found'] for s in scores)}/{sum(s['breaks'] for s in scores)}",
            "false_alarms": sum(s["false_alarms"] for s in scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to .litertlm, or http://host:port#name for llama-server")
    ap.add_argument("--asr", default="moonshine")
    ap.add_argument("--strategy", default="ranges")
    ap.add_argument("--episodes", default="")
    ap.add_argument("--split", default="dev", help="dev, test or all; used when --episodes is empty")
    ap.add_argument("--tag", default="")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tight-ends", action="store_true", help="an ad line ends 2 s after its last word, not at the next line")
    ap.add_argument("--arg", action="append", default=[], help="strategy kwargs, key=value")
    a = ap.parse_args()
    if not a.episodes:
        a.episodes = ",".join(e for e, m in EPISODES.items() if a.split in ("all", m.get("split", "dev"))
                              and os.path.exists(f"{ROOT}/gt/{e}.json")
                              and os.path.exists(f"{ROOT}/tx/{a.asr}/{e}.jsonl"))
    import llm as llm_mod
    global TIGHT_ENDS
    TIGHT_ENDS = a.tight_ends
    llm = llm_mod.load(a.model)
    if a.temp > 0:
        llm.sampling = (a.temp, a.seed)
    conv = lambda v: float(v) if v.replace(".", "").isdigit() and "." in v else int(v) if v.isdigit() else v
    kwargs = {k: conv(v) for k, v in (x.split("=") for x in a.arg)}
    fn = {"ranges": strategy_ranges, "blocks": strategy_blocks, "quotes": strategy_quotes}[a.strategy]
    run = f"{llm.name}__{a.asr}__{a.strategy}{a.tag}"
    os.makedirs(f"{ROOT}/results/{run}", exist_ok=True)
    scores = []
    for ep in a.episodes.split(","):
        duration, gt = load_gt(ep)
        lines = load_lines(a.asr, ep)
        before = dict(llm.stats)
        log = []
        t0 = time.time()
        extra = {"episode": ep} if a.strategy in ("blocks", "quotes") else {}
        is_ad = fn(llm, lines, EPISODES[ep], log=log, **extra, **kwargs)
        pred = to_intervals(lines, is_ad, duration)
        sc = score(pred, gt)
        scores.append(sc)
        st = {k: round(llm.stats[k] - before[k], 1) for k in before}
        res = {"episode": ep, "pred": [[fmt(s), fmt(e)] for s, e in pred],
               "gt": [[fmt(s), fmt(e), t] for s, e, t in gt], "score": sc, "stats": st,
               "wall_s": round(time.time() - t0, 1), "audio_s": duration, "log": log}
        json.dump(res, open(f"{ROOT}/results/{run}/{ep}.json", "w"), indent=1)
        print(ep, json.dumps(summarize([sc])), "pred", res["pred"], "stats", st, flush=True)
    summ = summarize(scores)
    summ["run"] = run
    print("SUMMARY", json.dumps(summ), flush=True)
    with open(f"{ROOT}/results/summary.jsonl", "a") as f:
        f.write(json.dumps(summ) + "\n")


if __name__ == "__main__":
    main()
