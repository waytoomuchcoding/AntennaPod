"""Unlabelled episodes for distillation: the newest 20-90 minute episode of each show below (none of them is in
the labelled set, so a student trained on them never sees the labelled shows or their same-day ads).
Feeds are looked up through the iTunes search API. Writes episodes_extra.json and wav/x_<slug>.wav."""
import json, os, re, subprocess, urllib.parse, urllib.request, xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "AntennaPod/3.0"}
SHOWS = ["Call Her Daddy", "Pod Save America", "Hidden Brain", "Radiolab", "This American Life", "Armchair Expert",
         "My Favorite Murder", "Up First", "The Bill Simmons Podcast", "Science Vs", "99% Invisible",
         "How I Built This", "Revisionist History", "Office Ladies", "The Mel Robbins Podcast", "SmartLess",
         "Pardon My Take", "The Ezra Klein Show", "Hard Fork", "Today, Explained", "Stuff You Missed in History Class",
         "Casefile True Crime", "Rotten Mango", "Last Podcast on the Left", "The Diary Of A CEO", "Search Engine",
         "Wait Wait... Don't Tell Me!", "The Moth", "Throughline", "Behind the Bastards", "Lore",
         "Normal Gossip", "The Prof G Pod", "Acquired", "Fresh Air"]
OUT = f"{ROOT}/episodes_extra.json"
meta = json.load(open(OUT)) if os.path.exists(OUT) else {}


def get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def secs(d):
    if not d:
        return 0
    p = [float(x) for x in d.strip().split(":") if x.replace(".", "").isdigit()]
    return sum(v * 60 ** i for i, v in enumerate(reversed(p)))


ITUNES = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"
for show in SHOWS:
    slug = "x_" + re.sub(r"[^a-z0-9]+", "", show.lower())[:16]
    if slug in meta and os.path.exists(f"{ROOT}/wav/{slug}.wav"):
        continue
    try:
        res = json.loads(get("https://itunes.apple.com/search?" + urllib.parse.urlencode(
            {"term": show, "media": "podcast", "limit": 3})))["results"]
        feed = next(r["feedUrl"] for r in res if r.get("feedUrl"))
        root = ET.fromstring(get(feed))
        item = next(i for i in root.findall("channel/item")
                    if 20 * 60 <= secs(i.findtext(ITUNES + "duration")) <= 90 * 60 and i.find("enclosure") is not None)
        mp3, wav = f"{ROOT}/audio/{slug}.mp3", f"{ROOT}/wav/{slug}.wav"
        open(mp3, "wb").write(get(item.find("enclosure").get("url"), timeout=900))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp3, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav],
                       check=True)
        meta[slug] = {"podcast": root.findtext("channel/title"), "title": item.findtext("title"), "feed": feed,
                      "split": "extra", "downloaded": "2026-10-01"}
        json.dump(meta, open(OUT, "w"), indent=1)
        print(slug, "ok", meta[slug]["podcast"], "|", meta[slug]["title"], flush=True)
    except Exception as ex:
        print(slug, "FAILED", show, repr(ex)[:150], flush=True)
