"""Audio embeddings per transcript line, so the text classifier pipeline (emb_loo.py, seq_smooth.py) can be reused
unchanged. Each line gets [vector of its own audio, mean vector of lines i-2..i+2], saved as emb/<name>/<ep>.npy.

  spectral  numpy log-mel (64 bands) mean + std over the line, plus loudness: "production style" only, ~0 cost
  whisper   openai/whisper-base encoder (74M total), run on 30 s chunks, encoder frames (20 ms) mean-pooled per line
  clap      laion/clap-htsat-unfused audio tower (audio-caption model), 10 s window centred on the line (48 kHz)
  ast       MIT/ast-finetuned-audioset-10-10-0.4593 (AudioSet sound events), 10.24 s window centred on the line,
            pooled embedding
  gemma_audio  Gemma 4 E2B's audio encoder (305M), 30 s chunks, frames mean-pooled per line
usage: GT_DIR=gt_v2 ./venv/bin/python audio_emb.py <spectral|whisper|clap|ast|gemma_audio>
"""
import os, sys, time
import numpy as np
import soundfile as sf
import torch
import adtest
from emb_loo import eps

name = sys.argv[1]
torch.set_num_threads(6)
SR = 16000


def ctx_mean(V):
    return np.vstack([V[max(0, i - 2):i + 3].mean(0) for i in range(len(V))])


def spans(lines):
    return [(l["t"], max(l["e"], l["t"] + 1.0)) for l in lines]


def window(x, centre, sec, sr):
    n = int(sec * sr); a = int(centre * sr) - n // 2
    w = x[max(0, a):max(0, a) + n]
    return np.pad(w, (0, n - len(w)))


if name == "spectral":
    fb = None
    def line_vecs(x, lines):
        global fb
        nfft, hop = 512, 160
        if fb is None:   # 64 mel triangles, 50 Hz - 8 kHz
            mel = lambda f: 2595 * np.log10(1 + f / 700); imel = lambda m: 700 * (10 ** (m / 2595) - 1)
            pts = imel(np.linspace(mel(50), mel(8000), 66)); bins = np.floor((nfft + 1) * pts / SR).astype(int)
            fb = np.zeros((64, nfft // 2 + 1))
            for i in range(64):
                a, b, c = bins[i], bins[i + 1], bins[i + 2]
                fb[i, a:b] = (np.arange(a, b) - a) / max(b - a, 1); fb[i, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
        out = []
        for s, e in spans(lines):
            seg = x[int(s * SR):int(e * SR)]
            fr = np.lib.stride_tricks.sliding_window_view(seg, nfft)[::hop] * np.hanning(nfft)
            lm = np.log(fb @ (np.abs(np.fft.rfft(fr, axis=1)) ** 2).T + 1e-8)
            out.append(np.concatenate([lm.mean(1), lm.std(1), [10 * np.log10(np.mean(seg ** 2) + 1e-10)]]))
        return np.array(out)

elif name == "whisper":
    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    enc = WhisperModel.from_pretrained("openai/whisper-base").encoder.eval()
    def line_vecs(x, lines):
        frames = []
        with torch.no_grad():
            for k in range(0, len(x), 30 * SR):
                f = fe(x[k:k + 30 * SR], sampling_rate=SR, return_tensors="pt").input_features
                frames.append(enc(f).last_hidden_state[0].numpy())       # 1500 frames x 512, 20 ms each
        H = np.vstack(frames)
        return np.array([H[int(s / 0.02):max(int(e / 0.02), int(s / 0.02) + 1)].mean(0) for s, e in spans(lines)])

elif name == "clap":
    from scipy.signal import resample_poly
    from transformers import ClapModel, ClapProcessor
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    def line_vecs(x, lines):
        out = []
        with torch.no_grad():
            for k in range(0, len(lines), 16):
                ws = [resample_poly(window(x, (s + e) / 2, 10, SR), 3, 1) for s, e in spans(lines[k:k + 16])]
                inp = proc(audios=ws, sampling_rate=48000, return_tensors="pt")
                out.append(model.get_audio_features(**inp).numpy())
        return np.vstack(out)

elif name == "ast":
    from transformers import ASTFeatureExtractor, ASTModel
    rid = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = ASTFeatureExtractor.from_pretrained(rid)
    model = ASTModel.from_pretrained(rid).eval()
    def line_vecs(x, lines):
        out = []
        with torch.no_grad():
            for k in range(0, len(lines), 16):
                ws = [window(x, (s + e) / 2, 10.24, SR) for s, e in spans(lines[k:k + 16])]
                out.append(model(**fe(ws, sampling_rate=SR, return_tensors="pt")).pooler_output.numpy())
        return np.vstack(out)

elif name == "gemma_audio":
    # Gemma 4 E2B's own audio encoder (305M, USM-style conformer), weights extracted from google/gemma-4-E2B-it
    # (only the audio tensors: models/gemma4_e2b_audio/, see README 14.5). Run on 30 s chunks, frames mean-pooled.
    import json
    from transformers.models.gemma4 import Gemma4AudioConfig, Gemma4AudioModel, Gemma4AudioFeatureExtractor
    d = f"{adtest.ROOT}/models/gemma4_e2b_audio"
    gm = Gemma4AudioModel(Gemma4AudioConfig(**json.load(open(f"{d}/config.json"))["audio_config"]))
    idx = json.load(open(f"{d}/index.json")); raw = open(f"{d}/span.bin", "rb").read()
    sd = {k[len("model.audio_tower."):]: torch.frombuffer(bytearray(raw[v["data_offsets"][0] - idx["lo"]:
                                                                         v["data_offsets"][1] - idx["lo"]]),
                                                           dtype=torch.bfloat16).reshape(v["shape"])
          for k, v in idx["tensors"].items() if k.startswith("model.audio_tower.")}
    gm.load_state_dict(sd, strict=True); gm = gm.float().eval(); del raw, sd
    pc = json.load(open(f"{d}/processor_config.json"))
    gfe = Gemma4AudioFeatureExtractor(**{k: v for k, v in pc.get("feature_extractor", pc).items()
                                         if k != "feature_extractor_type"})
    def line_vecs(x, lines):
        frames, step = [], 30 * SR
        with torch.no_grad():
            for k in range(0, len(x), step):
                chunk = x[k:k + step]
                if len(chunk) < SR:
                    chunk = np.pad(chunk, (0, SR - len(chunk)))
                f = gfe([chunk], sampling_rate=SR, return_tensors="pt")
                mask = next((f[m] for m in ("input_features_mask", "attention_mask") if m in f), None)
                o = gm(input_features=f["input_features"], attention_mask=mask)
                enc, emask = o.last_hidden_state, o.attention_mask
                h = enc[0][emask[0]].numpy() if emask is not None else enc[0].numpy()
                frames.append((k / SR, len(chunk) / SR / max(len(h), 1), h))
        H = np.vstack([h for _, _, h in frames]); dt = frames[0][1]
        return np.array([H[int(s / dt):max(int(e / dt), int(s / dt) + 1)].mean(0) for s, e in spans(lines)])

total = 0.0
for ep in eps:
    path = f"emb/{name}/{ep}.npy"
    if os.path.exists(path):
        continue
    x, sr = sf.read(f"{adtest.ROOT}/wav/{ep}.wav", dtype="float32")
    assert sr == SR
    t = time.time()
    V = line_vecs(x, eps[ep]["lines"])
    dt = time.time() - t; total += dt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.hstack([V, ctx_mean(V)]).astype(np.float32))
    print(f"  {name} {ep}: {len(V)} lines, {len(x) / SR / 60:.0f} min audio in {dt:.0f}s", flush=True)
print(f"done {name}: {total:.0f}s for {sum(eps[e]['dur'] for e in eps) / 3600:.1f} h audio")
