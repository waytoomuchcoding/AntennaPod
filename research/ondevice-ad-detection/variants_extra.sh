#!/bin/bash
# Two more copies (different User-Agents) of every extra episode, then transcripts, for dai_align.py labels.
cd "$(dirname "$0")"
eps=$(./venv/bin/python -c "import json;print(','.join(json.load(open('episodes_extra.json'))))")
./venv/bin/python fetch_variants.py "$eps" ua1,ua2 > logs/fetch_variants_extra.txt 2>&1
./transcribe_all.sh > logs/tx_variants_extra.txt 2>&1
echo "$(date +%T) VARIANTS_DONE" >> logs/night.txt
