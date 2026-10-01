"""Use audio cues after the LLM: snap predicted ad edges to nearby pauses / speaker changes, optionally drop
predictions spoken only by the episode's main voice. Re-scores a saved run; no LLM calls."""
import json, os, sys
import adtest


def boundaries(ep, lines, use_pause=True, use_spk=True, min_gap=1.5):
    """Candidate cut points (seconds): starts of lines that follow a pause and/or a speaker change."""
    spk = None
    if use_spk and os.path.exists(f"{adtest.ROOT}/diar/{ep}.json"):
        adtest.line_annotations(ep, lines, "s")
        spk = adtest.line_annotations.speakers
    cuts = [0.0]
    for i in range(1, len(lines)):
        gap = lines[i]["t"] - lines[i - 1]["e"]
        if (use_pause and gap >= min_gap) or (spk and spk[i] != spk[i - 1]):
            cuts.append((lines[i - 1]["e"] + lines[i]["t"]) / 2)
    return cuts, spk


def snap(pred, cuts, max_shift):
    out = []
    for s, e in pred:
        cs = min(cuts, key=lambda c: abs(c - s))
        ce = min(cuts, key=lambda c: abs(c - e))
        s2 = cs if abs(cs - s) <= max_shift else s
        e2 = ce if abs(ce - e) <= max_shift else e
        out.append((s2, e2) if e2 > s2 else (s, e))
    return out


def run(run_name, asr="moonshine", **kw):
    scores = []
    for f in sorted(os.listdir(f"{adtest.ROOT}/results/{run_name}")):
        ep = f[:-5]
        r = json.load(open(f"{adtest.ROOT}/results/{run_name}/{f}"))
        dur, gt = adtest.load_gt(ep)
        lines = adtest.load_lines(asr, ep)
        pred = [(adtest.mmss(s), adtest.mmss(e)) for s, e in r["pred"]]
        if kw.get("max_shift"):
            cuts, _ = boundaries(ep, lines, kw.get("pause", True), kw.get("spk", True))
            pred = snap(pred, cuts, kw["max_shift"])
        if kw.get("drop_main"):
            adtest.line_annotations(ep, lines, "s")
            spk = adtest.line_annotations.speakers
            keep = []
            for s, e in pred:
                sp = [spk[i] for i, l in enumerate(lines) if s <= l["t"] < e and spk[i] is not None]
                if sp and all(x == 1 for x in sp):
                    continue
                keep.append((s, e))
            pred = keep
        scores.append(adtest.score(pred, gt))
    return adtest.summarize(scores)


if __name__ == "__main__":
    base = sys.argv[1]
    print("baseline                 ", run(base))
    for ms in (5, 10, 20):
        for pause, spk in ((True, False), (False, True), (True, True)):
            print(f"snap {ms:2d}s pause={pause!s:5} spk={spk!s:5}", run(base, max_shift=ms, pause=pause, spk=spk))
    print("drop main-voice-only      ", run(base, drop_main=True))
