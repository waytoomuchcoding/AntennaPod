#!/bin/bash
cd "$(dirname "$0")"
while [ -n "$(ps -eo args | grep '^./venv/bin/python bench.py' )" ]; do sleep 15; done
./venv/bin/python bench.py models/gemma-4-E2B-it.litertlm focus600 budget=600 focus=1200 dump=1
./venv/bin/python bench.py models/gemma-4-E2B-it.litertlm focus1200 budget=1200 focus=1000 dump=1
