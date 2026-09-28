#!/bin/bash
# Boots the AVD "test" in software emulation with network access through the container's proxy.
ADB=/opt/android-sdk/platform-tools/adb
WIPE=""
[ "$1" = "--wipe" ] && WIPE="-wipe-data"
PORT=${HTTPS_PROXY##*:}
$ADB emu kill > /dev/null 2>&1 && sleep 5
nohup /opt/android-sdk/emulator/emulator -avd test -no-window -no-audio -no-boot-anim -accel off \
    -gpu swiftshader_indirect -no-snapshot -cores 4 $WIPE -http-proxy "$HTTPS_PROXY" \
    > /tmp/emulator.log 2>&1 &
echo "Waiting for boot (this takes about 10 minutes without KVM)..."
until [ "$($ADB shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; do sleep 10; done
# System services keep failing for a while after boot_completed on the slow emulator
sleep 120
$ADB shell settings put global http_proxy 10.0.2.2:$PORT
$ADB shell settings put global hide_error_dialogs 1
echo "Emulator ready, proxy 10.0.2.2:$PORT"
