#!/bin/bash
set -e
SDK=/opt/android-sdk
if [ ! -x $SDK/cmdline-tools/latest/bin/sdkmanager ]; then
    mkdir -p $SDK/cmdline-tools
    curl -sSfL -o /tmp/clt.zip https://dl.google.com/android/repository/commandlinetools-linux-13114758_latest.zip
    unzip -q /tmp/clt.zip -d $SDK/cmdline-tools
    mv $SDK/cmdline-tools/cmdline-tools $SDK/cmdline-tools/latest
    rm /tmp/clt.zip
fi
export ANDROID_HOME=$SDK ANDROID_SDK_ROOT=$SDK
yes | $SDK/cmdline-tools/latest/bin/sdkmanager --licenses > /dev/null 2>&1 || true
$SDK/cmdline-tools/latest/bin/sdkmanager "platform-tools" "platforms;android-36" "build-tools;36.0.0" \
    "emulator" "system-images;android-30;default;x86_64" > /dev/null
if [ ! -d ~/.android/avd/test.avd ]; then
    echo no | $SDK/cmdline-tools/latest/bin/avdmanager create avd -n test \
        -k "system-images;android-30;default;x86_64" -d pixel_4 > /dev/null
    echo "hw.ramSize=3072" >> ~/.android/avd/test.avd/config.ini
fi
echo "sdk.dir=$SDK" > "$(git rev-parse --show-toplevel)/local.properties"
echo "SDK ready in $SDK"
