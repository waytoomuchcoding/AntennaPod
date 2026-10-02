#!/bin/bash
# One-time setup on an arm64 or x86_64 Linux box (tested on arm64 Lima VMs: Ubuntu 24.04 CPU only, Fedora 44 with Venus GPU).
set -e
cd "$(dirname "$0")"
if command -v dnf > /dev/null; then
  sudo dnf install -y ffmpeg-free cmake gcc gcc-c++ python3-devel aria2 vulkan-headers vulkan-loader-devel glslc \
    glslang spirv-tools spirv-headers-devel
else
  sudo apt-get install -y ffmpeg cmake build-essential python3-venv aria2 libvulkan-dev glslc spirv-headers
fi
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install sherpa-onnx numpy soundfile huggingface_hub litert-lm ai-edge-litert scikit-learn scipy

# Speech models (sherpa-onnx), VAD, diarization, YAMNet
mkdir -p asr && cd asr
B=https://github.com/k2-fsa/sherpa-onnx/releases/download
for m in sherpa-onnx-moonshine-base-en-int8 sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8 sherpa-onnx-whisper-base.en sherpa-onnx-whisper-tiny.en; do
  [ -d $m ] || curl -sL $B/asr-models/$m.tar.bz2 | tar xj
done
[ -f silero_vad.onnx ] || curl -sLO $B/asr-models/silero_vad.onnx
[ -d sherpa-onnx-pyannote-segmentation-3-0 ] || curl -sL $B/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 | tar xj
[ -f nemo_en_titanet_small.onnx ] || curl -sLO $B/speaker-recongition-models/nemo_en_titanet_small.onnx
[ -f yamnet.tflite ] || curl -sL -o yamnet.tflite https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite
[ -f yamnet_class_map.csv ] || curl -sL -o yamnet_class_map.csv https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv
cd ..

# LLMs: Gemma 4 in Google's on-device LiteRT-LM format; Gemma 3n as GGUF (Google's 3n LiteRT files are gated)
./dl_models.sh

# llama.cpp, only needed for the Gemma 3n GGUF runs
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DGGML_VULKAN=ON
cmake --build llama.cpp/build -j --target llama-server llama-bench

# Audio and transcripts (see README: dynamic ad insertion may change the ads)
./venv/bin/python fetch_audio.py
./transcribe_all.sh
