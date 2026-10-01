"""Second, independent labeler for the ground truth: a large cloud model reads the whole transcript and lists
ad segments. Disagreements with gt/*.json are printed for manual review (the cloud labels are not used directly)."""
import json, os, sys, urllib.request, urllib.error
import adtest

KEY = open(os.path.expanduser("~/.config/antennapod-test/gemini-api-key")).read().strip()
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.1-pro-preview"

PROMPT = """Below is the full timestamped transcript (automatic speech recognition, so it has errors) of the podcast "{podcast}", episode "{title}". The episode is {dur} long.

Label every segment that is an advertisement: sponsor reads, host-read ads, commercials, promos for other podcasts, shows, films or products, and network ads. Report each ad break as one segment, from the first line of the first ad to the last line of the last ad. Separately label "self_promo" segments where the show promotes itself (its own newsletter, app, other formats of the same show, merchandise). Do not label the episode's own content, teasers of the episode, or credits.

Return JSON: a list of objects with "start" and "end" (MM:SS or H:MM:SS, the timestamps of the first and last line of the segment), "type" ("ad" or "self_promo") and "what" (the advertised brands/shows).

Transcript:
{transcript}"""


def call(prompt):
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        json.dumps(body).encode(), {"Content-Type": "application/json", "x-goog-api-key": KEY})
    import time
    for attempt in range(3):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503) or attempt == 2:
                raise
            time.sleep(20 * (attempt + 1))
    return "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"] if not p.get("thought"))


for ep, meta in adtest.EPISODES.items():
    if len(sys.argv) > 2 and meta.get("split", "dev") != sys.argv[2]:
        continue
    dur, gt = adtest.load_gt(ep)
    lines = adtest.load_lines("moonshine", ep)
    tx = "\n".join(f"[{adtest.fmt(l['t'])}] {l['text']}" for l in lines)
    path = f"{adtest.ROOT}/gt/check_{ep}.json"
    if not os.path.exists(path):
        segs = json.loads(call(PROMPT.format(transcript=tx, dur=adtest.fmt(dur), **meta)))
        json.dump(segs, open(path, "w"), indent=1)
    segs = json.load(open(path))
    cloud = [(adtest.mmss(s["start"]), adtest.mmss(s["end"]), s["type"], s.get("what", "")) for s in segs]
    print(f"== {ep}")
    for s, e, t, w in cloud:
        ov = sum(adtest.overlap((s, e + 5), (a, b)) for a, b, tt in gt)
        mark = "" if ov >= 0.5 * (e + 5 - s) else "   <-- NOT IN GT"
        print(f"  cloud {t:10s} {adtest.fmt(s)}-{adtest.fmt(e)} {w[:50]}{mark}")
    for a, b, t in gt:
        ov = sum(adtest.overlap((a, b), (s, e + 5)) for s, e, tt, w in cloud)
        if ov < 0.5 * (b - a):
            print(f"  gt    {t:10s} {adtest.fmt(a)}-{adtest.fmt(b)}   <-- NOT IN CLOUD")
