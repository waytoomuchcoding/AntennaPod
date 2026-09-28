---
name: android-emulator
description: Set up the Android SDK and an emulator in a Claude Code cloud container, give it working network access, install and launch AntennaPod, drive the UI and take screenshots. Use when asked to run the app, take screenshots or test a change on a device.
---

# Running AntennaPod in an Android emulator

The cloud container has no KVM, so the emulator runs in software emulation. It works, but everything is slow
(about 10 minutes to boot, several seconds per screenshot). Plan for that and run long waits in the background.

## 1. Install the SDK (once per container)

```
.claude/skills/android-emulator/scripts/setup-sdk.sh
```

Installs the Google command line tools, platform 36, build tools, the emulator and an Android 11 (API 30)
x86_64 system image into `/opt/android-sdk`, creates the AVD `test` and writes `local.properties`.
Only Google packages from `dl.google.com` are used.

Build with `ANDROID_HOME=/opt/android-sdk ./gradlew :app:assembleDebug`. Maven Central sometimes answers
`429 Too Many Requests` during the first build; just run the build again until all dependencies are cached.

## 2. Boot the emulator with network access

```
.claude/skills/android-emulator/scripts/start-emulator.sh          # add --wipe for a clean device
```

Networking in the container goes through an HTTPS proxy (`$HTTPS_PROXY`, usually `http://127.0.0.1:<port>`)
that forwards to the environment's egress gateway. Two settings are needed, and the script does both:

1. Start the emulator with `-http-proxy "$HTTPS_PROXY"`.
2. Set Android's own proxy: `adb shell settings put global http_proxy 10.0.2.2:<port>`
   (`10.0.2.2` is the host as seen from the emulator).

The second one matters: with only `-http-proxy`, the emulator resolves DNS itself and asks the proxy to
CONNECT to raw IP addresses, which the gateway drops ("Connection reset" in the app). With Android's proxy
setting, apps send the hostname and HTTPS works without installing any certificates.
If requests still fail, `curl -sS "$HTTPS_PROXY/__agentproxy/status"` lists recent relay failures.

The script also disables the "isn't responding" dialogs (`settings put global hide_error_dialogs 1`),
which otherwise pop up constantly on the slow emulator. The emulator may be killed when the session is idle;
check `adb devices` and start it again if needed. If `/sdcard` is missing after an unclean shutdown,
restart with `--wipe`.

## 3. Install and launch

```
adb install -r app/build/outputs/apk/play/debug/app-play-debug.apk
adb shell monkey -p de.danoeh.antennapod.debug 1
```

## 4. Screenshots and UI control

- `scripts/screenshot.sh out.png` takes a screenshot on the device and pulls it. Do not use
  `adb exec-out screencap -p > file`, it often produces truncated PNGs on the slow emulator.
- `scripts/tap.sh "Text"` finds a view by its text in a `uiautomator` dump and taps its center,
  retrying until the view appears. For icons without text, grep the dump for `content-desc` instead.
- `adb shell uiautomator dump /sdcard/ui.xml; adb shell cat /sdcard/ui.xml` gives the screen as text,
  which is much faster than a screenshot for checking state.
- Screenshots are 1080x2280. Images shown to you may be scaled, multiply coordinates accordingly.
- Snackbars disappear before a screenshot finishes; verify them through logs instead.

## 5. Logs

`adb logcat -d` without a filter can hang on this emulator. Always filter by tag and use a timeout:
`timeout 60 adb logcat -d -s MyTag:V`. Crashes: `timeout 30 adb logcat -d -b crash`.

## 6. Preparing test data without the network

`adb root` works on the `default` system image. Useful for seeding data:

- Preferences: push a file to `/data/data/<package>/shared_prefs/<package>_preferences.xml` while the app
  is stopped, then `chown` it to the app user and run `restorecon`.
- Database: pull `/data/data/<package>/databases/Antennapod.db`, edit it with Python's `sqlite3`, push it
  back, delete the `-journal` file and fix the ownership as above.
- Media files: write them below `/data/media/0/Android/data/<package>/files/`, `chown -R` them to
  `<app uid>:1078` (ext_data_rw) and run `restorecon -R`. Files pushed to `/sdcard` as root are not
  readable by the app (EACCES).
