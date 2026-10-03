"""Label episodes with the app's own Gemini ad-detection step (AdSegmentIndexer.detectAdsInWindow): same model,
prompt, response schema, 20-minute windows every 15 minutes and 5 s merge, applied to our Moonshine transcripts
(the app transcribes with Gemini first; we skip that to save quota). Resumable per episode.
Key: $GEMINI_API_KEY or ~/.config/antennapod-test/gemini-api-key (never logged or committed).
Writes gt_gemini/<ep>.json (label format; ad + self_promo). Stops cleanly when the daily quota is exhausted.
usage: ./venv/bin/python gemini_label.py ep1,ep2,...   |   ./venv/bin/python gemini_label.py --bulk  (all lab/in/b_*)
"""
import json, os, sys, time, urllib.error, urllib.request
import adtest

ROOT = adtest.ROOT
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
WINDOW_S, STEP_S, MERGE_GAP_S = 20 * 60, 15 * 60, 5.0
MIN_INTERVAL_S = float(os.environ.get("GEMINI_MIN_INTERVAL", 4))     # stay under the per-minute limit
OUT = f"{ROOT}/gt_gemini"
KEY = os.environ.get("GEMINI_API_KEY") or open(os.path.expanduser("~/.config/antennapod-test/gemini-api-key")).read().strip()
_last = [0.0]


class QuotaExhausted(Exception):
    pass


def fmt(s):
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


def prompt_for(lines, episode, podcast):
    transcript = "".join(f"{i} [{fmt(l['t'])}] {l['text']}\n" for i, l in enumerate(lines))
    return ("You are labeling ad breaks in a podcast so that a player can skip them. "
            f"Below is a numbered, timestamped transcript of the episode \"{episode}\" from the podcast \"{podcast}\".\n\n"
            "Mark every segment that is not part of the episode's actual content:\n"
            "- \"ad\": sponsor reads, host-read ads, pre-roll/mid-roll/post-roll ads, movie trailers, "
            "promos for other podcasts or products, network idents such as \"This is an iHeart podcast\", "
            "and legal disclaimers of ads.\n"
            "- \"self_promo\": the show promoting itself: its own newsletter, membership, Patreon, "
            "merchandise, YouTube channel, live shows or asking listeners to rate/subscribe.\n\n"
            "Rules:\n"
            "- An ad break usually contains several ads back to back. Report the whole break as ONE segment, "
            "from the first line of the first ad to the last line of the last ad. "
            "Never split a break because the sponsor changes.\n"
            "- Do NOT mark the episode's own content, including its cold open, intro, teasers of the episode "
            "(\"after the break...\", \"coming up...\"), credits or listener mail.\n"
            "- Ads often start with phrases like \"support for this show comes from\", "
            "\"this episode is sponsored by\" or \"brought to you by\", "
            "and end right before the host returns to the topic.\n"
            "- Ads at the very start or end of the episode are common; include them.\n\n"
            "Return the first and last line number and the type of each segment.\n\nTranscript:\n" + transcript)


def generate(prompt):
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": {"type": "ARRAY", "items": {
                                     "type": "OBJECT",
                                     "properties": {"first_line": {"type": "INTEGER"}, "last_line": {"type": "INTEGER"},
                                                    "type": {"type": "STRING", "enum": ["ad", "self_promo"]}},
                                     "required": ["first_line", "last_line", "type"]}}}}
    for attempt in range(6):
        wait = MIN_INTERVAL_S - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        req = urllib.request.Request(URL, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "x-goog-api-key": KEY})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
            text = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])
            return json.loads(text), r.get("usageMetadata", {})
        except urllib.error.HTTPError as ex:
            msg = ex.read().decode(errors="replace")[:400]
            if ex.code == 429 and ("PerDay" in msg or "per day" in msg.lower()):
                raise QuotaExhausted(msg)
            if ex.code in (429, 500, 503) and attempt < 5:
                time.sleep(30 * (attempt + 1)); continue
            raise RuntimeError(f"HTTP {ex.code}: {msg}")
        except (KeyError, json.JSONDecodeError, urllib.error.URLError) as ex:
            if attempt < 5:
                time.sleep(10 * (attempt + 1)); continue
            raise


def label(ep):
    lines = adtest.load_lines("moonshine", ep)
    m = adtest.EPISODES.get(ep, {})
    dur = max(m.get("duration_s", 0), lines[-1]["e"] + 1)
    segs, usage = [], {"in": 0, "out": 0, "calls": 0}
    start = 0.0
    while True:
        window = [l for l in lines if start <= l["t"] < start + WINDOW_S]
        after = [l for l in lines if l["t"] >= start + WINDOW_S]
        end = after[0]["t"] if after else dur
        if window:
            res, u = generate(prompt_for(window, m.get("title", ""), m.get("podcast", "")))
            usage["in"] += u.get("promptTokenCount", 0); usage["out"] += u.get("candidatesTokenCount", 0)
            usage["calls"] += 1
            for s in res:
                a, b = s["first_line"], s["last_line"]
                if 0 <= a <= b < len(window):
                    e = window[b + 1]["t"] if b + 1 < len(window) else end
                    segs.append((window[a]["t"], e, s["type"]))
        if start + WINDOW_S > max(dur, lines[-1]["t"]):
            break
        start += STEP_S
    merged = []
    for s, e, t in sorted(segs):
        if merged and merged[-1][2] == t and s <= merged[-1][1] + MERGE_GAP_S:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e), t)
        else:
            merged.append((s, e, t))
    os.makedirs(OUT, exist_ok=True)
    json.dump({"duration": dur, "labeller": MODEL, "usage": usage,
               "segments": [{"s": fmt(s), "e": fmt(e), "type": t} for s, e, t in merged]},
              open(f"{OUT}/{ep}.json", "w"))
    return usage


if __name__ == "__main__":
    once = "--bulk" not in sys.argv
    while True:
        eps = (sys.argv[1].split(",") if once else
               sorted(f[:-4] for f in os.listdir(f"{ROOT}/lab/in") if f.startswith("b_")))
        todo = [e for e in eps if not os.path.exists(f"{OUT}/{e}.json")]
        for ep in todo:
            try:
                u = label(ep)
                print(f"{time.strftime('%H:%M:%S')} {ep}: {u['calls']} calls, {u['in']} in / {u['out']} out tokens", flush=True)
            except QuotaExhausted as ex:
                print(f"{time.strftime('%H:%M:%S')} daily quota exhausted, sleeping 1 h: {str(ex)[:150]}", flush=True)
                time.sleep(3600); break
            except Exception as ex:
                print(f"{time.strftime('%H:%M:%S')} {ep} failed: {ex!r}"[:300], flush=True)
        if once:
            break
        if not todo:
            if "ALL DONE" in open(f"{ROOT}/logs/bulk.txt").read():
                break
            time.sleep(120)
