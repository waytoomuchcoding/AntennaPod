"""Per-second audio features with YAMNet (LiteRT, the same .tflite MediaPipe's Android Audio Classifier uses):
music probability, speech probability and loudness (dBFS). Output: feats/<ep>.json"""
import json, sys, time
import numpy as np, soundfile as sf
from ai_edge_litert.interpreter import Interpreter

classes = [l.split(",")[2].strip() for l in open("asr/yamnet_class_map.csv").read().splitlines()[1:]]
MUSIC = [i for i, c in enumerate(classes) if c in ("Music",) or "music" in c.lower() and "instrument" not in c.lower()]


def main(ep, threads=4):
    x, sr = sf.read(f"wav/{ep}.wav", dtype="float32")
    it = Interpreter("asr/yamnet.tflite", num_threads=threads)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    n = len(x) // sr
    music, speech, loud = [], [], []
    t0 = time.time()
    for s in range(n):
        w = x[s * sr: s * sr + 15600]
        if len(w) < 15600:
            w = np.pad(w, (0, 15600 - len(w)))
        it.set_tensor(inp, w)
        it.invoke()
        p = it.get_tensor(out)[0]
        music.append(round(float(max(p[i] for i in MUSIC)), 3))
        speech.append(round(float(p[0]), 3))
        rms = float(np.sqrt(np.mean(x[s * sr:(s + 1) * sr] ** 2)) + 1e-9)
        loud.append(round(20 * np.log10(rms), 1))
    json.dump({"music": music, "speech": speech, "loud": loud}, open(f"feats/{ep}.json", "w"))
    print(ep, f"{n}s audio in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    for ep in sys.argv[1:]:
        main(ep)
