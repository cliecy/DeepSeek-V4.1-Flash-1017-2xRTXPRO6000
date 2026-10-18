#!/usr/bin/env bash
# Resource log during a launch: MemAvailable, MemFree, page cache, Unevictable, per-GPU VRAM (MiB). usage: monitor.sh [secs]
while true; do
  m=$(awk '/^MemAvailable/{a=$2} /^Unevictable/{u=$2} /^Cached/{c=$2} /^MemFree/{f=$2}
           END{printf "avail %.0fG free %.0fG cached %.0fG unevict %.1fG", a/1048576, f/1048576, c/1048576, u/1048576}' /proc/meminfo)
  echo "$(date +%T) $m vram $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
  sleep "${1:-20}"
done
