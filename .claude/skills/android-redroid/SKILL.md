---
name: android-redroid
description: Run AntennaPod in redroid (Android in a Docker container) on an arm64 Linux VM with sudo, such as a Lima VM on Apple Silicon. Covers one-time setup, booting the device, installing the app, driving the UI, screenshots, instrumentation tests and seeding test data. Use when asked to run the app, take screenshots or test a change on a device in that environment. In the Claude Code cloud container, use the android-emulator skill instead.
---

# Running AntennaPod in redroid

redroid runs Android 14 as a container on the host kernel, so there is no emulation: it boots in about
5 seconds, installs the APK in a fraction of a second and cold-starts the app in under a second.
It needs an arm64 Linux host with sudo (tested on Ubuntu 24.04 in Lima on Apple Silicon).

## 1. Set up the host (once)

```
.claude/skills/android-redroid/scripts/setup.sh
```

The script is idempotent and sets up the following:
- Docker, JDK 21 and adb.
- The Android SDK in `~/Android/Sdk`, plus `local.properties`.
- qemu-user and amd64 libraries, because Google ships `aapt2` and the build tools only for x86_64.
- A `redroid-binderfs` systemd service.
- A 4 GB swap file if the host has no swap.
- The redroid image.

Why the unusual parts are needed:
- **64-bit only image**: use `redroid/redroid:14.0.0_64only-latest`. Apple Silicon cannot run 32-bit ARM
  code, and the regular image reboots in a loop (`boringssl_self_test32: Exec format error`).
- **binderfs permissions**: the kernel creates binder nodes in binderfs with mode 0600. Without `chmod 666`,
  `servicemanager` gets "Permission denied" on `/dev/binder` and Android never finishes booting.
- **Swap**: Gradle (around 2.5 GB with the Kotlin daemon) and Android do not fit into a 6 GB VM together.
  Without swap, Android's low memory killer ends up restarting `system_server` in a loop and the build
  slows to a crawl. If the machine becomes very slow, check `uptime` and `swapon --show`.

## 2. Start the device

```
.claude/skills/android-redroid/scripts/start-redroid.sh          # add --wipe for a clean device
```

The container restarts on its own after a reboot of the VM. Use the system `adb` (not the one from the SDK,
which is x86_64) and `export ANDROID_SERIAL=localhost:5555`.

## 3. Build, install and launch

Build and install with the commands from `AGENTS.md`, then start the app. The application ID of the debug
build is `com.waytoomuchcoding.antennapod.debug`:

```
adb shell am start -n com.waytoomuchcoding.antennapod.debug/de.danoeh.antennapod.activity.MainActivity
```

A fresh install has no notification permission. Grant it with
`adb shell pm grant <package> android.permission.POST_NOTIFICATIONS` to see notifications.
Subscribe to a feed without typing: `adb shell am start -a android.intent.action.VIEW -d "antennapod-subscribe://<feed url without https://>"`.

Instrumentation tests run on the device like on any emulator, for example
`./gradlew :app:connectedPlayDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=<test class>`.
They uninstall the app afterwards, which removes its data.

## 4. Screenshots and UI control

- `scripts/screenshot.sh out.png` saves a screenshot (1080x1920). With a crop box,
  `scripts/screenshot.sh out.png 40 1270 1040 1340` also writes `out-zoom.png`, the box scaled up 2x,
  which is useful for checking small details such as the progress bar.
- `scripts/tap.sh 'text="Settings"'` taps the first view matching the regex. It matches text,
  `content-desc` and resource IDs (`'id/butPlay'`). A second argument picks among several matches.
- `adb exec-out uiautomator dump /dev/tty` gives the screen as text. It fails while something animates,
  for example the position text during playback. Pause first with `adb shell input keyevent MEDIA_PAUSE`.
- Switch themes with `adb shell cmd uimode night yes|no`. The activity is recreated.

## 5. Preparing test data

The container's `/data` is bind-mounted at `~/redroid-data`, so app files can be edited from the host with
`sudo`. Stop the app first with `adb shell am force-stop <package>`.

- Database: `~/redroid-data/data/<package>/databases/Antennapod.db`, edit it with Python's `sqlite3`.
  `FeedMedia.downloaded` holds a timestamp, not a flag. Ad breaks are stored in `FeedMedia.ad_segments`
  as one `start end type enabled` line per break (milliseconds, type 0 = ad, 1 = show promo).
- Preferences: write `~/redroid-data/data/<package>/shared_prefs/<package>_preferences.xml` and give it the
  owner of the other files in that folder (`sudo stat -c %u:%g`).

## 6. Logs

`adb logcat -d` works normally. `adb logcat -d -b crash` shows crashes. If Android itself does not boot,
its init messages are in the host's kernel log: `sudo dmesg | grep 'init:' | tail -40`.
