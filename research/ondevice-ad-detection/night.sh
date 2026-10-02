#!/bin/bash
# Overnight chain (2026-10-01), resumable: every step skips work whose output already exists, so after a
# container restart just run ./night.sh again. Progress in logs/night.txt.
#  1. EmbeddingGemma leave-one-show-out (CPU)  ||  Gemma E2B hidden-state probes, full depth and layer 20 (GPU)
#  2. Moonshine transcripts of the extra episodes
#  3. clean CPU-vs-GPU llama-bench on an idle machine
#  4. E4B copy6 teacher labels for the extra episodes (distillation)
cd "$(dirname "$0")"
export GT_DIR=gt_v2 GROUP=show
log() { echo "$(date +%T) $*" >> logs/night.txt; }
killport() { for pid in $(logs/ps_match.sh "port $1" | awk '{print $1}'); do kill $pid; done; sleep 3; }
probe() {  # name, extra server args
  local n=$(ls emb/$1 2>/dev/null | wc -l)
  [ "$n" -ge 13 ] && return
  killport 8092
  llama.cpp/build/bin/llama-server -m models/gemma-4-E2B-it-Q4_K_M.gguf --embeddings --pooling none -ngl 99 -t 2 \
      -c 2048 -b 2048 -ub 2048 --port 8092 --no-webui $2 > logs/probe_server_$1.txt 2>&1 &
  local pid=$!
  until curl -s localhost:8092/health | grep -q ok; do
    kill -0 $pid 2>/dev/null || { log "probe $1 server failed to start"; return; }; sleep 2; done
  ./venv/bin/python gemma_probe.py $1 8092 >> logs/probe_$1.txt 2>&1
  log "probe $1 rc=$?"
  killport 8092
}
log START
( [ "$(ls emb/egemma | wc -l)" -ge 13 ] && grep -q '== egemma' logs/emb_egemma.txt 2>/dev/null ||
  ./venv/bin/python emb_loo.py egemma --veto gemma-4-E2B-it.litertlm__moonshine__quotes_copy6_v2_verified \
      > logs/emb_egemma.txt 2>&1; log "egemma done" ) &
probe probe_e2b_last ""
# Layer-20 probe: --override-kv block_count fails (per-layer arrays must have 35 entries); needs a truncated GGUF.
wait
log "embeddings and probes done"

./transcribe_all.sh > logs/tx_extra.txt 2>&1
log "transcription done: $(ls tx/moonshine/x_*.jsonl | wc -l) extra transcripts"

if ! grep -q 'tg64' logs/clean_bench.txt 2>/dev/null; then
  { date +%T; uptime; free -m; } > logs/clean_bench.txt
  timeout -s KILL 1800 llama.cpp/build/bin/llama-bench -m models/gemma-4-E2B-it-Q4_K_M.gguf -ngl 0,99 -t 6 \
      -p 1200 -n 64 -r 3 >> logs/clean_bench.txt 2>&1
  log "clean bench done rc=$?"
fi

T=results/gemma-4-E4B-it.litertlm__moonshine__quotes_teacher
todo=$(./venv/bin/python -c "
import json,os
print(','.join(e for e in json.load(open('episodes_extra.json'))
               if os.path.exists(f'tx/moonshine/{e}.jsonl') and not os.path.exists(f'$T/{e}.json')))")
if [ -n "$todo" ]; then
  GT_DIR=gt_extra ./venv/bin/python adtest.py --model models/gemma-4-E4B-it.litertlm --episodes "$todo" \
      --strategy quotes --arg mode=copy --arg iters=6 --arg min_words=5 --tag _teacher >> logs/teacher.txt 2>&1
  log "teacher done rc=$?"
fi
log ALL_DONE

# 5. distillation students (appended 01:40): LR on frozen embeddings, then a fine-tuned bge-small
if [ -n "$(ls results/gemma-4-E4B-it.litertlm__moonshine__quotes_teacher 2>/dev/null)" ]; then
  for m in egemma bge-small; do ./venv/bin/python distill.py $m > logs/distill_$m.txt 2>&1; log "distill $m rc=$?"; done
  ./venv/bin/python finetune.py > logs/finetune_bge.txt 2>&1; log "finetune rc=$?"
fi
log STUDENTS_DONE
