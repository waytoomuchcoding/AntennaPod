"""Bulk pipeline for LLM-labelled training data (README 16). Resumable: every step is skipped when its output
exists, so after a restart just run it again. Per episode of episodes_bulk.json:
  1. download two copies (default User-Agent and "ua1"), convert to 16 kHz mono WAV
  2. Moonshine transcript  -> tx/moonshine/<ep>.jsonl
  3. audio alignment of the two copies (audio_align.py) -> gt_auto_bulk/<ep>.json (inserted ads, high precision)
  4. labelling input for the LLM labellers -> lab/in/<ep>.txt
  5. delete the audio (only transcripts and labels are kept)
Downloads run ahead of transcription by at most AHEAD episodes and pause when free disk < MIN_FREE_GB.
Order: round-robin over shows, so the first episodes are already diverse. Progress: logs/bulk.txt
usage: setsid nohup ./venv/bin/python bulk.py > /dev/null 2>&1 &
"""
import json, os, queue, shutil, subprocess, threading, time, traceback, urllib.request
import adtest
import audio_align
import labtools

ROOT = adtest.ROOT
AHEAD, MIN_FREE_GB, THREADS = 8, 4.0, 6
UAS = {"": "AntennaPod/3.0", "ua1": "Mozilla/5.0 (Linux; Android 16; Pixel 10 Pro Fold) PodcastAddict/2024.1"}
eps = json.load(open(f"{ROOT}/episodes_bulk.json"))
os.makedirs(f"{ROOT}/gt_auto_bulk", exist_ok=True)
lock = threading.Lock()


def log(msg):
    with lock, open(f"{ROOT}/logs/bulk.txt", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def order():
    by_show = {}
    for e, m in eps.items():
        by_show.setdefault(m["feed"], []).append(e)
    lists, out = list(by_show.values()), []
    for i in range(max(len(l) for l in lists)):
        out += [l[i] for l in lists if i < len(l)]
    return out


def done(e):
    return os.path.exists(f"{ROOT}/lab/in/{e}.txt") or os.path.exists(f"{ROOT}/logs/bulk_failed/{e}")


def fail(e, why):
    os.makedirs(f"{ROOT}/logs/bulk_failed", exist_ok=True)
    open(f"{ROOT}/logs/bulk_failed/{e}", "w").write(why)
    log(f"FAIL {e}: {why[:200]}")
    cleanup(e)


def cleanup(e):
    for k in UAS:
        sfx = f"__{k}" if k else ""
        for p in (f"{ROOT}/audio/{e}{sfx}.mp3", f"{ROOT}/wav/{e}{sfx}.wav"):
            if os.path.exists(p):
                os.remove(p)


def download(e):
    for k, ua in UAS.items():
        sfx = f"__{k}" if k else ""
        mp3, wav = f"{ROOT}/audio/{e}{sfx}.mp3", f"{ROOT}/wav/{e}{sfx}.wav"
        if os.path.exists(wav):
            continue
        data = urllib.request.urlopen(urllib.request.Request(eps[e]["url"], headers={"User-Agent": ua}), timeout=900).read()
        open(mp3, "wb").write(data)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp3, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                        wav + ".part.wav"], check=True)
        os.replace(wav + ".part.wav", wav)
        os.remove(mp3)


ready = queue.Queue()
todo = [e for e in order() if not done(e)]
log(f"START {len(todo)} episodes to do, {len(eps) - len(todo)} done")
slots = threading.Semaphore(AHEAD)


def downloader(items):
    for e in items:
        slots.acquire()
        while shutil.disk_usage(ROOT).free / 1e9 < MIN_FREE_GB:
            time.sleep(30)
        try:
            download(e)
            ready.put(e)
        except Exception as ex:
            fail(e, f"download: {ex!r}")
            slots.release()


workers = [threading.Thread(target=downloader, args=(todo[i::3],), daemon=True) for i in range(3)]
for w in workers:
    w.start()
finished = 0
while finished < len(todo):
    try:
        e = ready.get(timeout=60)
    except queue.Empty:
        if not any(w.is_alive() for w in workers) and ready.empty():
            break
        continue
    finished += 1
    try:
        t0 = time.time()
        tx = f"{ROOT}/tx/moonshine/{e}.jsonl"
        if not os.path.exists(tx):
            subprocess.run([f"{ROOT}/venv/bin/python", f"{ROOT}/transcribe.py", "moonshine", f"wav/{e}.wav",
                            f"tx/moonshine/{e}.jsonl.part", str(THREADS)], cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(tx + ".part", tx)
        a = audio_align.envelope(f"{ROOT}/wav/{e}.wav")
        iv, n = audio_align.missing_in(a, audio_align.envelope(f"{ROOT}/wav/{e}__ua1.wav"))
        json.dump({"duration": len(a) * audio_align.HOP / audio_align.SR, "anchors": n,
                   "segments": [{"s": adtest.fmt(s), "e": adtest.fmt(t), "type": "ad"} for s, t in iv]},
                  open(f"{ROOT}/gt_auto_bulk/{e}.json", "w"))
        labtools.export(e)
        cleanup(e)
        log(f"ok {e} ({eps[e]['duration_s'] / 60:.0f} min audio, {len(iv)} inserted ads) in {time.time() - t0:.0f}s")
    except Exception:
        fail(e, traceback.format_exc())
    finally:
        slots.release()
log("ALL DONE")
