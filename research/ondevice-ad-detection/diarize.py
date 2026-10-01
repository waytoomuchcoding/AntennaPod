"""Offline speaker diarization with sherpa-onnx (pyannote segmentation 3.0 + speaker embeddings).
Output: diar/<ep>.json list of [start, end, speaker]."""
import json, sys, time
import soundfile as sf, sherpa_onnx

EMB = {"titanet": "asr/nemo_en_titanet_small.onnx", "eres2net": "asr/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"}


def main(ep, emb="titanet", threshold=0.6, threads=6):
    x, sr = sf.read(f"wav/{ep}.wav", dtype="float32")
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model="asr/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"), num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB[emb], num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    t0 = time.time()
    res = sd.process(x).sort_by_start_time()
    segs = [[round(r.start, 2), round(r.end, 2), r.speaker] for r in res]
    json.dump(segs, open(f"diar/{ep}.json", "w"))
    spk = {}
    for s, e, k in segs:
        spk[k] = spk.get(k, 0) + e - s
    print(ep, f"{len(x)/sr:.0f}s audio in {time.time()-t0:.1f}s; speakers (s):",
          {k: round(v) for k, v in sorted(spk.items(), key=lambda kv: -kv[1])})


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or []), *(float(a) for a in sys.argv[3:4]))
