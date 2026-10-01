#!/bin/bash
cd "$(dirname "$0")"
M=models/gemma-4-E2B-it.litertlm
./venv/bin/python bench.py $M slide400 region=1200 budget=400 dump=1
./venv/bin/python bench.py $M slide400_veto region=1200 budget=400 veto=0.15 dump=1
./venv/bin/python bench.py $M slide400_think region=1200 budget=400 think=512 dump=1
./venv/bin/python bench.py $M slide400_veto_think region=1200 budget=400 veto=0.15 think=512 dump=1
