#!/bin/bash
cd "$(dirname "$0")/models"
for p in gemma-4-E2B-it-litert-lm/gemma-4-E2B-it-gpu.litertlm gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm \
         gemma-4-E4B-it-litert-lm/gemma-4-E4B-it-gpu.litertlm gemma-4-E4B-it-litert-lm/gemma-4-E4B-it.litertlm; do
  f=${p#*/}
  [ -f "$f" ] && [ ! -f "$f.aria2" ] && continue
  aria2c --console-log-level=warn --summary-interval=0 -x16 -s16 -k4M --file-allocation=none -o "$f" "https://huggingface.co/litert-community/${p%/*}/resolve/main/$f" \
    && echo "OK $f" || echo "FAIL $f"
done
echo MODELS_DONE
