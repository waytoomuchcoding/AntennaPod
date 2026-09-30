#!/bin/bash
# Starts the redroid container (creating it on first use) and waits until Android has booted.
# Usage: start-redroid.sh [--wipe]
set -e
IMAGE=redroid/redroid:14.0.0_64only-latest
DATA=$HOME/redroid-data
export ANDROID_SERIAL=localhost:5555

sudo /usr/local/sbin/redroid-binderfs
if [ "$1" = "--wipe" ]; then
    sudo docker rm -f redroid > /dev/null 2>&1 || true
    sudo rm -rf "$DATA"
fi
if sudo docker inspect redroid > /dev/null 2>&1; then
    sudo docker start redroid > /dev/null
else
    sudo docker run -itd --privileged --name redroid --restart unless-stopped \
        -v "$DATA":/data \
        -v /dev/binderfs/binder:/dev/binder \
        -v /dev/binderfs/hwbinder:/dev/hwbinder \
        -v /dev/binderfs/vndbinder:/dev/vndbinder \
        -p 5555:5555 $IMAGE \
        androidboot.redroid_width=1080 androidboot.redroid_height=1920 androidboot.redroid_dpi=420 \
        androidboot.redroid_gpu_mode=guest > /dev/null
fi
for i in $(seq 1 90); do
    adb connect localhost:5555 > /dev/null 2>&1 || true
    if [ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; then
        echo "redroid ready after ${i}s at localhost:5555"
        exit 0
    fi
    sleep 1
done
echo "redroid did not boot, check: sudo dmesg | grep 'init:' | tail -40"
exit 1
