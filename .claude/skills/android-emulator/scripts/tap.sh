#!/bin/bash
# Usage: tap.sh "<visible text>" [attempts]
# Taps the center of the first view with exactly this text, retrying until it appears.
ADB=/opt/android-sdk/platform-tools/adb
for i in $(seq 1 "${2:-20}"); do
    $ADB shell uiautomator dump /sdcard/ui.xml > /dev/null 2>&1
    b=$($ADB shell cat /sdcard/ui.xml 2>/dev/null | grep -o "text=\"$1\"[^>]*bounds=\"[^\"]*\"" | head -1 \
        | grep -o 'bounds="[^"]*"' | grep -o '[0-9]\+' | tr '\n' ' ')
    if [ -n "$b" ]; then
        set -- "$1" $b
        $ADB shell input tap $(( ($2 + $4) / 2 )) $(( ($3 + $5) / 2 ))
        echo "tapped $1"
        exit 0
    fi
    sleep 5
done
echo "not found: $1"
exit 1
