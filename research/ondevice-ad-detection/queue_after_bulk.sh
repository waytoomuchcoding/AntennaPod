#!/bin/bash
# Heavy experiments, strictly one at a time, only after bulk.py has finished transcribing (README 16.4: the
# Moonshine transcriber alone peaks at ~3.8-4.6 GB on this 5.9 GB VM; running anything heavy next to it caused
# the OOM kills). The Gemini labeller (light, ~50 MB) keeps running alongside. Progress: logs/queue.txt
cd "$(dirname "$0")"
log() { echo "$(date +%T) $*" >> logs/queue.txt; }
running() { [ -n "$(logs/ps_match.sh "$1")" ]; }
log "waiting for bulk.py"
until grep -q 'ALL DONE' logs/bulk.txt 2>/dev/null && ! running 'bulk.py'; do sleep 120; done
log "bulk done; starting EmbeddingGemma server"
setsid llama.cpp/build/bin/llama-server -m models/embeddinggemma-300M-Q8_0.gguf --embeddings -ngl 99 -t 2 \
    -c 8192 -b 8192 -ub 8192 -np 1 --port 8093 --no-webui > logs/egemma_gguf_server.txt 2>&1 &
srv=$!
until curl -s localhost:8093/health | grep -q ok; do kill -0 $srv 2>/dev/null || { log "server failed"; exit 1; }; sleep 2; done

# 1. linear classifier: human vs human+bulk vs bulk, all Gemini-labelled bulk episodes so far (union labels)
LABELS=union GT_DIR=gt_v2 GROUP=show nice ./venv/bin/python train_bulk.py 8093 > logs/train_bulk_all.txt 2>&1
log "train_bulk (all labelled bulk) rc=$?"
kill $srv; wait $srv 2>/dev/null

# 2. sequence model (BiGRU 32, 15 epochs) with and without the bulk episodes
python3 - <<'EOF'
import os, numpy as np
for f in os.listdir("emb/egemma_q8_gpu"):
    if not os.path.exists(f"emb/egemma_q8_ctr/{f}"):
        X = np.load(f"emb/egemma_q8_gpu/{f}")
        np.save(f"emb/egemma_q8_ctr/{f}", np.hstack([X, X[:, 768:] - X[:, 768:].mean(0)]).astype(np.float32))
EOF
for b in "" 9999; do
  d=$(BULK=$b GT_DIR=gt_v2 GROUP=show nice ./venv/bin/python seq_model.py egemma_q8_ctr 32 15 2>/dev/null | tail -1)
  GT_DIR=gt_v2 GROUP=show nice ./venv/bin/python eval_all.py scores $d 2>/dev/null | grep -v Warn >> logs/seq_bulk.txt
  log "seq_model bulk=${b:-0} done"
done
log "QUEUE DONE"
