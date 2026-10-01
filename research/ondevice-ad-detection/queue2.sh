#!/bin/bash
# Each line: <model>|<shell command using {spec}>. Starts llama-server for .gguf models. One at a time.
cd "$(dirname "$0")"
while IFS='|' read -r model cmd; do
  [ -z "$model" ] && continue
  srv=
  if [[ "$model" == *.gguf ]]; then
    name=$(basename $model .gguf)
    llama.cpp/build/bin/llama-server -m $model -c 4608 -t 6 --port 8090 -np 1 --no-webui --cache-ram 0 > logs_server_$name.txt 2>&1 &
    srv=$!
    until curl -s localhost:8090/health | grep -q ok; do sleep 2; done
    spec="http://localhost:8090#$name.gguf"
  else
    spec=$model
  fi
  c=${cmd//\{spec\}/$spec}
  echo "=== $c" >> logs_queue2.txt
  bash -c "$c" 2>&1 | grep -v "^W0\|^I0\|ERROR: third\|XNNPack" >> logs_queue2.txt
  [ -n "$srv" ] && kill $srv && wait $srv 2>/dev/null
done < "$1"
echo "QUEUE DONE" >> logs_queue2.txt
