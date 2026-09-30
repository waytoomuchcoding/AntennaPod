#!/bin/bash
# Usage: screenshot.sh <output.png> [left top right bottom]
# With a crop box, also writes <output>-zoom.png: the box scaled up 2x, useful for small UI details.
export ANDROID_SERIAL=localhost:5555
adb exec-out screencap -p > "$1"
if [ -n "$5" ]; then
    python3 - "$@" <<'EOF'
import sys
from PIL import Image
out, box = sys.argv[1], tuple(int(v) for v in sys.argv[2:6])
im = Image.open(out).crop(box)
im.resize((im.width * 2, im.height * 2), Image.LANCZOS).save(out[:-4] + "-zoom.png")
EOF
fi
