#!/bin/bash
# Wait (max ~9.5 min) for a new SUMMARY or a new Traceback in logs_queue2.txt.
cd "$(dirname "$0")"
n=$(grep -c '^SUMMARY' logs_queue2.txt); t=$(grep -c 'Traceback' logs_queue2.txt)
timeout 570 bash -c "until [ \$(grep -c '^SUMMARY' logs_queue2.txt) -gt $n ] || [ \$(grep -c Traceback logs_queue2.txt) -gt $t ]; do sleep 20; done"
date +%H:%M; grep -E '^SUMMARY|Traceback' logs_queue2.txt | tail -2 | cut -c1-220
