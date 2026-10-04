#!/bin/bash
# Idempotent relaunch after a container restart (README 16): starts whichever of the long-running jobs is not
# running. Safe to call any number of times. Uses logs/ps_match.sh (never pkill -f, which matches itself).
cd "$(dirname "$0")"
running() { [ -n "$(logs/ps_match.sh "$1")" ]; }
if ! grep -q 'ALL DONE' logs/bulk.txt 2>/dev/null && ! running 'bulk.py'; then
  MIN_AD_S=10 setsid nohup ./venv/bin/python bulk.py > logs/bulk_stdout.txt 2>&1 &
  echo "started bulk.py"
fi
if ! running 'gemini_label.py'; then
  setsid nohup ./venv/bin/python gemini_label.py --bulk >> logs/gemini_bulk.txt 2>&1 &
  echo "started gemini_label.py"
fi
# optional extra job: ./relaunch.sh train  (bulk training evaluation, needs the EmbeddingGemma GPU server)
if [ "$1" = "train" ] && ! grep -qE '^bulk \(' logs/train_bulk_1.txt 2>/dev/null; then
  if ! running 'port 8093'; then
    setsid nohup llama.cpp/build/bin/llama-server -m models/embeddinggemma-300M-Q8_0.gguf --embeddings -ngl 99 -t 2 \
        -c 8192 -b 8192 -ub 8192 -np 1 --port 8093 --no-webui > logs/egemma_gguf_server.txt 2>&1 &
    for i in $(seq 60); do curl -s localhost:8093/health | grep -q ok && break; sleep 2; done
    echo "started embedding server"
  fi
  if ! running 'train_bulk.py'; then
    BULK_MAX=${BULK_MAX:-80} GT_DIR=gt_v2 GROUP=show setsid nohup nice ./venv/bin/python train_bulk.py 8093 > logs/train_bulk_1.txt 2>&1 &
    echo "started train_bulk.py"
  fi
fi
