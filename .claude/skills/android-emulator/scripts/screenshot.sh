#!/bin/bash
# Usage: screenshot.sh <output.png>
ADB=/opt/android-sdk/platform-tools/adb
timeout 120 $ADB shell screencap -p /sdcard/screenshot.png && timeout 60 $ADB pull /sdcard/screenshot.png "$1" > /dev/null
