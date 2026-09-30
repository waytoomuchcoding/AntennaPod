#!/bin/bash
# Usage: tap.sh "<regex>" [index]
# Taps the center of the view whose text, content-desc or resource-id matches the regex,
# e.g. 'text="Settings"' or 'id/butPlay'. Index picks among several matches (default 1).
export ANDROID_SERIAL=localhost:5555
b=$(adb exec-out uiautomator dump /dev/tty 2>/dev/null | grep -o '<node [^>]*>' | grep -E "$1" \
    | sed -n "${2:-1}p" | grep -o 'bounds="[^"]*"' | grep -oE '[0-9]+' | tr '\n' ' ')
if [ -z "$b" ]; then
    echo "not found: $1"
    exit 1
fi
set -- $b
adb shell input tap $(( ($1 + $3) / 2 )) $(( ($2 + $4) / 2 ))
