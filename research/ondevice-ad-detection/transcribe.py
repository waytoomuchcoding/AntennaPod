"""Transcribe a 16 kHz mono WAV with sherpa-onnx: Silero VAD splits speech, an offline ASR model
recognizes each speech segment. Output: one JSON line per segment {"t": start_seconds, "e": end, "text": ...}.
This mirrors what an Android app would do with sherpa-onnx's Java/Kotlin API."""
import json, sys, time
import numpy as np, soundfile as sf, sherpa_onnx

import os
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asr") + "/"


def make_recognizer(name, threads):
    if name == "parakeet":
        d = A + "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/"
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=d + "encoder.int8.onnx", decoder=d + "decoder.int8.onnx", joiner=d + "joiner.int8.onnx",
            tokens=d + "tokens.txt", model_type="nemo_transducer", num_threads=threads)
    if name == "moonshine":
        d = A + "sherpa-onnx-moonshine-base-en-int8/"
        return sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=d + "preprocess.onnx", encoder=d + "encode.int8.onnx",
            uncached_decoder=d + "uncached_decode.int8.onnx", cached_decoder=d + "cached_decode.int8.onnx",
            tokens=d + "tokens.txt", num_threads=threads)
    if name.startswith("whisper-"):
        size = name.split("-", 1)[1]
        d = A + f"sherpa-onnx-whisper-{size}/"
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=d + f"{size}-encoder.int8.onnx", decoder=d + f"{size}-decoder.int8.onnx",
            tokens=d + f"{size}-tokens.txt", num_threads=threads)
    raise ValueError(name)


def main(model, wav, out, threads=6, limit_s=None):
    samples, sr = sf.read(wav, dtype="float32")
    assert sr == 16000
    if limit_s:
        samples = samples[: int(limit_s * sr)]
    rec = make_recognizer(model, threads)
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = A + "silero_vad.onnx"
    cfg.silero_vad.min_silence_duration = 0.25
    cfg.silero_vad.max_speech_duration = 20 if not model.startswith("whisper") else 28
    cfg.sample_rate = sr
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
    t0 = time.time()
    segs = []

    def drain():
        while not vad.empty():
            seg = vad.front
            st = rec.create_stream()
            st.accept_waveform(sr, seg.samples)
            rec.decode_stream(st)
            text = st.result.text.strip()
            if text:
                segs.append({"t": round(seg.start / sr, 2), "e": round((seg.start + len(seg.samples)) / sr, 2),
                             "text": text})
            vad.pop()

    win = cfg.silero_vad.window_size
    for i in range(0, len(samples), win):
        vad.accept_waveform(samples[i:i + win])
        drain()
    vad.flush()
    drain()
    dt = time.time() - t0
    with open(out, "w") as f:
        for s in segs:
            f.write(json.dumps(s) + "\n")
    audio_s = len(samples) / sr
    print(json.dumps({"model": model, "wav": wav, "audio_s": round(audio_s), "wall_s": round(dt, 1),
                      "rtf": round(dt / audio_s, 4), "threads": threads, "segments": len(segs)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 6,
         float(sys.argv[5]) if len(sys.argv) > 5 else None)
