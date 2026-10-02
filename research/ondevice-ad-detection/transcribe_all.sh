#!/bin/bash
cd "$(dirname "$0")"
for w in wav/*.wav; do
  e=$(basename $w .wav)
  [ -s tx/moonshine/$e.jsonl ] && continue
  # write to .part and rename, so a run killed midway is redone instead of looking finished
  ./venv/bin/python transcribe.py moonshine $w tx/moonshine/$e.jsonl.part 6 && mv tx/moonshine/$e.jsonl.part tx/moonshine/$e.jsonl || echo "TXFAIL $e"
done
echo TX_DONE
