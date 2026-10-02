#!/bin/bash
cd "$(dirname "$0")"
for w in wav/*.wav; do
  e=$(basename $w .wav)
  [ -s tx/moonshine/$e.jsonl ] && continue
  ./venv/bin/python transcribe.py moonshine $w tx/moonshine/$e.jsonl 6 || echo "TXFAIL $e"
done
echo TX_DONE
