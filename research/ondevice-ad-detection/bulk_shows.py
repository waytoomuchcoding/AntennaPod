"""Pick a large, diverse set of unlabelled episodes for LLM labelling (README 16): search the iTunes podcast
directory for many genre terms, skip every show already in episodes.json / episodes_extra.json (keeps the
evaluation shows unseen), take up to PER_SHOW recent 20-90 min episodes per show until TARGET_H hours.
Writes episodes_bulk.json (split "bulk"); prints the hosting-provider mix.
usage: ./venv/bin/python bulk_shows.py [target_hours] [per_show]"""
import collections, json, os, re, sys, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET_H = float(sys.argv[1]) if len(sys.argv) > 1 else 520
PER_SHOW = int(sys.argv[2]) if len(sys.argv) > 2 else 2
UA = {"User-Agent": "AntennaPod/3.0"}
IT = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"
TERMS = ["true crime", "comedy", "news", "politics", "business", "investing", "technology", "science", "history",
         "health", "fitness", "nutrition", "mental health", "sports", "football", "basketball", "baseball", "soccer",
         "nfl", "society culture", "relationships", "parenting", "education", "self improvement", "entrepreneur",
         "marketing", "personal finance", "real estate", "film", "tv recap", "music", "video games", "food",
         "travel", "religion", "spirituality", "fiction", "horror stories", "kids", "arts", "design", "books",
         "comedy interview", "celebrity", "pop culture", "daily news", "economics", "law", "crypto", "ai",
         "outdoors", "cars", "wrestling", "golf", "beauty", "fashion", "true crime documentary", "paranormal",
         "conspiracy", "philosophy", "language learning", "careers", "dating", "wellness", "medicine"]

known = {}
for f in ("episodes.json", "episodes_extra.json"):
    if os.path.exists(f"{ROOT}/{f}"):
        known.update(json.load(open(f"{ROOT}/{f}")))
known_feeds = {m["feed"] for m in known.values()}
known_names = {m["podcast"].lower()[:20] for m in known.values()}
out = json.load(open(f"{ROOT}/episodes_bulk.json")) if os.path.exists(f"{ROOT}/episodes_bulk.json") else {}
seen_feeds = known_feeds | {m["feed"] for m in out.values()}


def get(url, timeout=30):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def secs(d):
    if not d:
        return 0
    try:
        p = [float(x) for x in d.strip().split(":")]
    except ValueError:
        return 0
    return sum(v * 60 ** i for i, v in enumerate(reversed(p)))


hours = sum(m["duration_s"] for m in out.values()) / 3600
for term in TERMS:
    if hours >= TARGET_H:
        break
    try:
        res = json.loads(get("https://itunes.apple.com/search?" + urllib.parse.urlencode(
            {"term": term, "media": "podcast", "limit": 50, "country": "US"})))["results"]
    except Exception as ex:
        print("search failed", term, ex); time.sleep(10); continue
    time.sleep(3)                                        # iTunes API rate limit (~20/min)
    for r in res:
        feed = r.get("feedUrl")
        if not feed or feed in seen_feeds or r.get("collectionName", "").lower()[:20] in known_names:
            continue
        seen_feeds.add(feed)
        try:
            root = ET.fromstring(get(feed))
        except Exception:
            continue
        lang = (root.findtext("channel/language") or "en").lower()
        if not lang.startswith("en"):
            continue
        n = 0
        for item in root.findall("channel/item"):
            enc = item.find("enclosure")
            d = secs(item.findtext(IT + "duration"))
            if enc is None or not 20 * 60 <= d <= 90 * 60 or "audio" not in (enc.get("type") or "audio"):
                continue
            slug = "b_" + re.sub(r"[^a-z0-9]+", "", r["collectionName"].lower())[:14] + f"_{n}"
            if slug in out:
                n += 1; continue
            url = enc.get("url")
            out[slug] = {"podcast": root.findtext("channel/title") or r["collectionName"], "title": item.findtext("title"),
                         "feed": feed, "url": url, "host": urllib.parse.urlparse(url).netloc, "duration_s": d,
                         "genre": r.get("primaryGenreName", ""), "term": term, "split": "bulk"}
            hours += d / 3600; n += 1
            if n >= PER_SHOW:
                break
        if hours >= TARGET_H:
            break
    json.dump(out, open(f"{ROOT}/episodes_bulk.json", "w"), indent=1)
    print(f"{term:24s} -> {len(out)} episodes, {hours:.0f} h", flush=True)

hosts = collections.Counter(re.sub(r"^.*?([a-z0-9-]+\.[a-z]+)$", r"\1", m["host"]) for m in out.values())
print("shows", len({m["feed"] for m in out.values()}), "hosts:", hosts.most_common(25))
