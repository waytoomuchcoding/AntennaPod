"""Which feeds insert ads per download (DAI) and which bake them into the audio?

For every episode we already downloaded (episodes.json + episodes_extra.json), re-request the enclosure now with two
different User-Agents (1-byte range requests, so nothing is downloaded) and compare the reported file size with
the copy on disk. Changed size between our download and now, or between the two User-Agents => dynamic insertion.
Writes logs/dai_survey.json and prints one line per episode.
"""
import json, os, urllib.request, xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
UAS = ["AntennaPod/3.0", "Mozilla/5.0 (Linux; Android 16; Pixel 10 Pro Fold) PodcastAddict/2024.1"]
eps = json.load(open(f"{ROOT}/episodes.json"))
if os.path.exists(f"{ROOT}/episodes_extra.json"):
    eps.update(json.load(open(f"{ROOT}/episodes_extra.json")))


def size(url, ua):
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")
        return (int(cr.split("/")[-1]) if cr else int(r.headers.get("Content-Length", -1))), r.geturl()


feeds, out = {}, {}
for ep, m in eps.items():
    try:
        if m["feed"] not in feeds:
            feeds[m["feed"]] = ET.fromstring(urllib.request.urlopen(urllib.request.Request(
                m["feed"], headers={"User-Agent": UAS[0]}), timeout=60).read())
        item = next((i for i in feeds[m["feed"]].findall("channel/item") if i.findtext("title") == m["title"]), None)
        if item is None:
            print(f"{ep:22s} not in feed any more"); continue
        url = item.find("enclosure").get("url")
        (s1, final), (s2, _) = size(url, UAS[0]), size(url, UAS[1])
        local = os.path.getsize(f"{ROOT}/audio/{ep}.mp3") if os.path.exists(f"{ROOT}/audio/{ep}.mp3") else None
        host = final.split("/")[2]
        dai = s1 != s2 or (local is not None and local != s1)
        out[ep] = dict(podcast=m["podcast"], host=host, local=local, now=s1, other_ua=s2, dai=dai)
        print(f"{ep:22s} {'DAI ' if dai else 'same'} local {local} now {s1} other-UA {s2}  {host}  ({m['podcast'][:30]})",
              flush=True)
    except Exception as ex:
        print(f"{ep:22s} error {repr(ex)[:100]}")
json.dump(out, open(f"{ROOT}/logs/dai_survey.json", "w"), indent=1)
