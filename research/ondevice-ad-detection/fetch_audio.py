"""Download every episode in episodes.json (by exact title from its RSS feed) and convert it to 16 kHz mono WAV.
NOTE: most of these feeds use dynamic ad insertion, so a fresh download may contain different ads than the
labelled audio in gt/ (downloaded 2026-09-30/10-01). Check the labels against the new transcript before trusting scores."""
import json, os, subprocess, urllib.request, xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "AntennaPod/3.0"}
os.makedirs(f"{ROOT}/audio", exist_ok=True)
os.makedirs(f"{ROOT}/wav", exist_ok=True)
for ep, m in json.load(open(f"{ROOT}/episodes.json")).items():
    wav = f"{ROOT}/wav/{ep}.wav"
    if os.path.exists(wav):
        continue
    root = ET.fromstring(urllib.request.urlopen(urllib.request.Request(m["feed"], headers=UA), timeout=60).read())
    item = next((i for i in root.findall("channel/item") if i.findtext("title") == m["title"]), None)
    if item is None:
        print(ep, "not in feed any more:", m["title"])
        continue
    mp3 = f"{ROOT}/audio/{ep}.mp3"
    open(mp3, "wb").write(urllib.request.urlopen(urllib.request.Request(item.find("enclosure").get("url"), headers=UA), timeout=900).read())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp3, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav], check=True)
    print(ep, "ok")
