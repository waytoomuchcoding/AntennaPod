#!/bin/bash
cd "$(dirname "$0")"
mkdir -p models && cd models
for p in litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm \
         litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it-gpu.litertlm \
         litert-community/gemma-4-E4B-it-litert-lm/gemma-4-E4B-it.litertlm \
         litert-community/gemma-4-E4B-it-litert-lm/gemma-4-E4B-it-gpu.litertlm \
         unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf \
         unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf \
         unsloth/gemma-3n-E2B-it-GGUF/gemma-3n-E2B-it-Q4_0.gguf \
         unsloth/gemma-3n-E4B-it-GGUF/gemma-3n-E4B-it-Q4_0.gguf; do
  f=${p##*/}
  [ -f "$f" ] && [ ! -f "$f.aria2" ] && continue
  aria2c --console-log-level=warn --summary-interval=0 -x16 -s16 -k4M --file-allocation=none -o "$f" \
    "https://huggingface.co/${p%/*}/resolve/main/$f" && echo "OK $f" || echo "FAIL $f"
done
echo MODELS_DONE
