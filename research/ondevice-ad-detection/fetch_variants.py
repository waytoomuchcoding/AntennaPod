"""Download extra copies of episodes with different User-Agents. Feeds with dynamic ad insertion stitch different
ads into identical content per request, so aligning the transcripts of two copies finds the inserted ads
(see dai_align.py). Copies are saved as audio/<ep>__<k>.mp3 and wav/<ep>__<k>.wav.
usage: ./venv/bin/python fetch_variants.py daily1,conan1,sysk1 [k1,k2,...]"""
import json, os, subprocess, sys, urllib.request, xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
UAS = {"ua1": "Mozilla/5.0 (Linux; Android 16; Pixel 10 Pro Fold) PodcastAddict/2024.1",
       "ua2": "Overcast/3.0 (+http://overcast.fm/; iOS podcast app)",
       "ua3": "PocketCasts/1.0 (Pocket Casts Feed Parser; +http://pocketcasts.com/)",
       "ua4": "Spotify/8.9.0 Android/34 (Pixel 9)",
       "ua5": "AppleCoreMedia/1.0.0.21A350 (iPhone; U; CPU OS 18_0 like Mac OS X; en_us)"}
eps = json.load(open(f"{ROOT}/episodes.json"))
if os.path.exists(f"{ROOT}/episodes_extra.json"):
    eps.update(json.load(open(f"{ROOT}/episodes_extra.json")))
names = sys.argv[1].split(",")
keys = sys.argv[2].split(",") if len(sys.argv) > 2 else ["ua1", "ua2"]
feeds = {}
for ep in names:
    m = eps[ep]
    if m["feed"] not in feeds:
        feeds[m["feed"]] = ET.fromstring(urllib.request.urlopen(urllib.request.Request(
            m["feed"], headers={"User-Agent": "AntennaPod/3.0"}), timeout=60).read())
    item = next((i for i in feeds[m["feed"]].findall("channel/item") if i.findtext("title") == m["title"]), None)
    if item is None:
        print(ep, "not in feed"); continue
    for k in keys:
        mp3, wav = f"{ROOT}/audio/{ep}__{k}.mp3", f"{ROOT}/wav/{ep}__{k}.wav"
        if os.path.exists(wav):
            continue
        data = urllib.request.urlopen(urllib.request.Request(item.find("enclosure").get("url"),
                                                             headers={"User-Agent": UAS[k]}), timeout=900).read()
        open(mp3, "wb").write(data)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp3, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav],
                       check=True)
        print(ep, k, len(data), "bytes", flush=True)
