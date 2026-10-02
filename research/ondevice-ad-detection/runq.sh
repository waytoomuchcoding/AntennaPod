#!/bin/bash
cd "$(dirname "$0")"
export GT_DIR=${GT_DIR:-gt_v2}
q=${1:-queue.txt}
mkdir -p logs
while true; do
  line=$(grep -m1 -v '^#\|^$' "$q" 2>/dev/null)
  [ -z "$line" ] && break
  sed -i "0,/^$(printf '%s' "$line" | sed 's/[]\/$*.^[]/\\&/g')$/s//# done: &/" "$q"
  echo "$(date +%T) JOB_START $line" >> logs/runq.txt
  bash -c "$line" >> logs/runq_out.txt 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "$(date +%T) JOB_END $line" >> logs/runq.txt; else echo "$(date +%T) JOB_FAIL rc=$rc $line" >> logs/runq.txt; fi
done
echo "$(date +%T) QUEUE_EMPTY" >> logs/runq.txt
