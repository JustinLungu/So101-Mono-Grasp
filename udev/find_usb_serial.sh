#!/usr/bin/env bash
# Print the USB serial number of every serial port, for udev/99-so101.rules.
# To tell the arms apart, plug in only one, run this, then plug in the other.
set -uo pipefail

shopt -s nullglob
ports=(/dev/ttyACM* /dev/ttyUSB*)
if [ ${#ports[@]} -eq 0 ]; then
  echo "No /dev/ttyACM* or /dev/ttyUSB* found. Is an arm plugged in and powered?"
  exit 1
fi

for dev in "${ports[@]}"; do
  props=$(udevadm info -q property -n "$dev")
  get() { sed -n "s/^$1=//p" <<<"$props"; }
  echo "$dev  serial=$(get ID_SERIAL_SHORT)  vendor:product=$(get ID_VENDOR_ID):$(get ID_MODEL_ID)  usb_path=$(get ID_PATH)"
done
