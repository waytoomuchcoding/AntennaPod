#!/bin/bash
# One-time setup on an arm64 or x86_64 Linux box (tested on an arm64 Lima VM, Ubuntu 24.04, CPU only).
set -e
cd "$(dirname "$0")"
sudo apt-get install -y ffmpeg cmake build-essential python3-venv
python3 -m venv venv
./venv/bin/pip install sherpa-onnx numpy soundfile huggingface_hub litert-lm ai-edge-litert scikit-learn scipy

# Speech models (sherpa-onnx), VAD, diarization, YAMNet
mkdir -p asr && cd asr
B=https://github.com/k2-fsa/sherpa-onnx/releases/download
for m in sherpa-onnx-moonshine-base-en-int8 sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8 sherpa-onnx-whisper-base.en sherpa-onnx-whisper-tiny.en; do
  curl -sL $B/asr-models/$m.tar.bz2 | tar xj
done
curl -sLO $B/asr-models/silero_vad.onnx
curl -sL $B/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 | tar xj
curl -sLO $B/speaker-recongition-models/nemo_en_titanet_small.onnx
curl -sL -o yamnet.tflite https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite
curl -sL -o yamnet_class_map.csv https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv
cd ..

# LLMs: Gemma 4 in Google's on-device LiteRT-LM format; Gemma 3n as GGUF (Google's 3n LiteRT files are gated)
./venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
for repo, f in [("litert-community/gemma-4-E2B-it-litert-lm", "gemma-4-E2B-it.litertlm"),
                ("litert-community/gemma-4-E4B-it-litert-lm", "gemma-4-E4B-it.litertlm"),
                ("unsloth/gemma-3n-E2B-it-GGUF", "gemma-3n-E2B-it-Q4_0.gguf"),
                ("unsloth/gemma-3n-E4B-it-GGUF", "gemma-3n-E4B-it-Q4_0.gguf")]:
    print(hf_hub_download(repo, f, local_dir="models"))
PY

# llama.cpp, only needed for the Gemma 3n GGUF runs
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
cmake --build llama.cpp/build -j --target llama-server

# Audio and transcripts (see README: dynamic ad insertion may change the ads)
./venv/bin/python fetch_audio.py
for m in moonshine; do mkdir -p tx/$m; for w in wav/*.wav; do e=$(basename $w .wav); ./venv/bin/python transcribe.py $m $w tx/$m/$e.jsonl; done; done
