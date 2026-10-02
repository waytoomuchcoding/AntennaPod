"""Find dynamically inserted ads by aligning the AUDIO of two downloads of the same episode.

Content is the same recording in every download; inserted ads differ. Steps:
 1. loudness envelope in 10 ms buckets (dB) for copy A and copy B
 2. cut A into 2 s anchor pieces; find each piece's best match in B (normalised cross-correlation of the
    envelope); content = runs of >= 3 consecutive pieces (corr >= 0.9) with the same time shift
 3. A buckets not covered by a matched piece are "not in B": inserted ads (or ads that differ)
 4. edge refinement: from each matched run, extend bucket by bucket while A and B still agree (|dB diff| small),
    so the ad edge is found to ~10 ms
With several copies, an A stretch counts as an ad if it is missing from ANY other copy (MODE=any) or from ALL
(MODE=all). Scores against human labels when they exist.
usage: GT_DIR=gt_v2 ./venv/bin/python audio_align.py daily1,conan1 [ua1,ua2] [--write DIR]
"""
import json, os, sys, time
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import adtest

SR, HOP = 16000, 160                      # 10 ms buckets
PIECE = 200                               # 2 s anchor pieces
MIN_AD_S = float(os.environ.get("MIN_AD_S", 5))
MODE = os.environ.get("MODE", "any")


def envelope(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR
    n = len(x) // HOP
    return 10 * np.log10((x[:n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-10)


def match_pieces(a, b):
    """For each 2 s piece of a: (start bucket in a, best start bucket in b, corr)."""
    bm = b - b.mean()
    out = []
    # sliding sums of b for normalisation
    c1 = np.concatenate([[0], np.cumsum(b)]); c2 = np.concatenate([[0], np.cumsum(b * b)])
    for s in range(0, len(a) - PIECE, PIECE):
        p = a[s:s + PIECE]
        if p.std() < 1.0:                 # silence / flat: ambiguous, skip
            continue
        p = (p - p.mean()) / p.std()
        num = fftconvolve(b, p[::-1], mode="valid")       # sum_k b[j+k] p[k]
        m = (c1[PIECE:] - c1[:-PIECE]) / PIECE
        var = (c2[PIECE:] - c2[:-PIECE]) / PIECE - m * m
        corr = np.where(var >= 1.0, num / (PIECE * np.sqrt(np.maximum(var, 1.0))), 0.0)  # flat/silent B windows: no match
        j = int(np.argmax(corr))
        out.append((s, j, float(corr[j])))
    return out


def chain(matches, min_corr=0.98):
    """Longest chain with increasing a and b positions among good matches (content keeps its order)."""
    good = [m for m in matches if m[2] >= min_corr]
    if not good:
        return []
    best = [1] * len(good); prev = [-1] * len(good)
    for i in range(len(good)):
        for k in range(max(0, i - 50), i):             # local window is enough: chain is nearly linear
            if good[k][1] < good[i][1] and best[k] + 1 > best[i]:
                best[i], prev[i] = best[k] + 1, k
    i = int(np.argmax(best)); out = []
    while i >= 0:
        out.append(good[i]); i = prev[i]
    return out[::-1]


def runs(matches, min_corr=0.9, min_run=3, tol=5):
    """Content pieces: consecutive pieces whose best match has the same time shift (within 50 ms). Copies are
    re-encoded, so correlation alone is not decisive (content often scores 0.93-0.99); a steady shift is."""
    out, cur = [], []
    for m in matches:
        ok = m[2] >= min_corr
        if ok and cur and m[0] - cur[-1][0] == PIECE and abs((m[1] - m[0]) - (cur[-1][1] - cur[-1][0])) <= tol:
            cur.append(m); continue
        if len(cur) >= min_run:
            out += cur
        cur = [m] if ok else []
    if len(cur) >= min_run:
        out += cur
    return out


def missing_in(a, b, tol_db=4.0):
    """Intervals (seconds) of a that have no counterpart in b."""
    ch = runs(match_pieces(a, b))
    covered = np.zeros(len(a), bool)
    for s, j, _ in ch:
        off = j - s
        covered[s:s + PIECE] = True
        # edge refinement: extend this run outwards bucket by bucket while a and b agree
        k = s - 1
        while k >= 0 and 0 <= k + off < len(b) and not covered[k] and abs(a[k] - b[k + off]) < tol_db:
            covered[k] = True; k -= 1
        k = s + PIECE
        while k < len(a) and k + off < len(b) and not covered[k] and abs(a[k] - b[k + off]) < tol_db:
            covered[k] = True; k += 1
    iv, start = [], None
    for t, c in enumerate(np.append(covered, True)):
        if not c and start is None:
            start = t
        elif c and start is not None:
            if (t - start) * HOP / SR >= MIN_AD_S:
                iv.append((start * HOP / SR, t * HOP / SR))
            start = None
    return iv, len(ch)


def combine(per, mode):
    if mode == "all":
        cur = per[0]
        for nxt in per[1:]:
            cur = [(max(a, c), min(b, d)) for a, b in cur for c, d in nxt if min(b, d) - max(a, c) >= MIN_AD_S]
        return cur
    out = []
    for s, e in sorted(sum(per, [])):
        if out and s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


if __name__ == "__main__":
    eps = sys.argv[1].split(",")
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ["ua1", "ua2"]
    write = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None
    scores = []
    for ep in eps:
        t0 = time.time()
        a = envelope(f"{adtest.ROOT}/wav/{ep}.wav")
        per, info = [], []
        for v in variants:
            p = f"{adtest.ROOT}/wav/{ep}__{v}.wav"
            if os.path.exists(p):
                iv, n = missing_in(a, envelope(p)); per.append(iv); info.append(n)
        iv = combine(per, MODE)
        line = f"{ep}: {len(iv)} ad stretches, {sum(e - s for s, e in iv) / 60:.1f} min, anchors {info}, {time.time() - t0:.0f}s"
        gt_path = f"{adtest.ROOT}/{os.environ.get('GT_DIR', 'gt')}/{ep}.json"
        if os.path.exists(gt_path) and json.load(open(gt_path))["segments"]:
            dur, gt = adtest.load_gt(ep)
            sc = adtest.score(iv, gt); scores.append(sc)
            ads = [(a_, b_) for a_, b_, t in gt if t == "ad"]
            missed = [f"{adtest.fmt(x)}-{adtest.fmt(y)}" for x, y in ads if sum(adtest.overlap(p, (x, y)) for p in iv) < 0.5 * (y - x)]
            fa = [f"{adtest.fmt(s)}-{adtest.fmt(e)}" for s, e in iv if sum(adtest.overlap((s, e), (x, y)) for x, y, _ in gt) < 0.5 * (e - s)]
            line += f"\n   {adtest.summarize([sc])}\n   labelled ads not found: {missed}\n   found but not labelled: {fa}"
        print(line, flush=True)
        if write:
            os.makedirs(write, exist_ok=True)
            json.dump({"duration": len(a) * HOP / SR, "variants": variants, "mode": MODE,
                       "segments": [{"s": adtest.fmt(s), "e": adtest.fmt(e), "type": "ad"} for s, e in iv]},
                      open(f"{write}/{ep}.json", "w"))
    if scores:
        print("ALL", adtest.summarize(scores))
