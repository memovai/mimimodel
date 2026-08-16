#!/bin/bash
# Flash needle2.cact into the raw `needle` partition (offset 0x210000).
set -e
PORT="${1:-/dev/cu.usbmodem1101}"
WEIGHTS="${2:-$(dirname "$0")/../../model/needle2.cact}"
PY=$(ls -d ~/.espressif/python_env/*/bin/python | head -1)
echo "Flashing $WEIGHTS to $PORT @ 0x210000 ..."
"$PY" -m esptool --chip esp32s3 --port "$PORT" --baud 921600 \
    write_flash 0x210000 "$WEIGHTS"
