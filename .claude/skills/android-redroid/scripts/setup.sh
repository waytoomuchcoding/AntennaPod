#!/bin/bash
# One-time host setup for running AntennaPod in redroid on an arm64 Ubuntu VM (e.g. Lima on Apple Silicon).
# Safe to run again. Needs sudo.
set -e
SDK=$HOME/Android/Sdk
IMAGE=redroid/redroid:14.0.0_64only-latest

sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "linux-modules-extra-$(uname -r)" docker.io \
    openjdk-21-jdk-headless adb unzip qemu-user-static binfmt-support python3-pil > /dev/null
sudo usermod -aG docker,kvm "$USER"

# Google only ships x86_64 builds of aapt2 and the build tools. Run them through qemu with amd64 libraries.
if ! dpkg --print-foreign-architectures | grep -q amd64; then
    sudo dpkg --add-architecture amd64
    grep -q '^Architectures:' /etc/apt/sources.list.d/ubuntu.sources \
        || sudo sed -i 's|^Types: deb$|Types: deb\nArchitectures: arm64|' /etc/apt/sources.list.d/ubuntu.sources
    sudo tee /etc/apt/sources.list.d/amd64.sources > /dev/null <<'EOF'
Types: deb
URIs: http://archive.ubuntu.com/ubuntu/
Suites: noble noble-updates noble-security
Components: main universe
Architectures: amd64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
    sudo apt-get update -qq
fi
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libc6:amd64 libstdc++6:amd64 zlib1g:amd64 > /dev/null

# redroid needs binder. The kernel uses binderfs, whose nodes are created with mode 0600,
# which non-root Android services cannot open. A boot-time service mounts it and opens up the nodes.
sudo tee /usr/local/sbin/redroid-binderfs > /dev/null <<'EOF'
#!/bin/sh
modprobe binder_linux devices=binder,hwbinder,vndbinder
mkdir -p /dev/binderfs
mountpoint -q /dev/binderfs || mount -t binder binder /dev/binderfs
chmod 666 /dev/binderfs/binder /dev/binderfs/hwbinder /dev/binderfs/vndbinder
EOF
sudo chmod +x /usr/local/sbin/redroid-binderfs
sudo tee /etc/systemd/system/redroid-binderfs.service > /dev/null <<'EOF'
[Unit]
Description=binderfs for redroid
Before=docker.service
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/redroid-binderfs
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now -q redroid-binderfs

# Gradle and Android together do not fit into a small VM without swap.
if ! swapon --show | grep -q .; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap -q /swapfile
    sudo swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
fi

if [ ! -x "$SDK/cmdline-tools/latest/bin/sdkmanager" ]; then
    mkdir -p "$SDK/cmdline-tools"
    curl -sSfL -o /tmp/clt.zip https://dl.google.com/android/repository/commandlinetools-linux-13114758_latest.zip
    unzip -q /tmp/clt.zip -d "$SDK/cmdline-tools"
    mv "$SDK/cmdline-tools/cmdline-tools" "$SDK/cmdline-tools/latest"
    rm /tmp/clt.zip
fi
yes | "$SDK/cmdline-tools/latest/bin/sdkmanager" --licenses > /dev/null 2>&1 || true
"$SDK/cmdline-tools/latest/bin/sdkmanager" "platforms;android-36" "build-tools;36.0.0" > /dev/null
echo "sdk.dir=$SDK" > "$(git rev-parse --show-toplevel)/local.properties"

sudo docker pull -q $IMAGE > /dev/null
echo "Setup done. Log out and back in once so the docker group applies."
