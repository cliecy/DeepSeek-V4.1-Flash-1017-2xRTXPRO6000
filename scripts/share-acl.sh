#!/usr/bin/env bash
# Optional: let another local account (e.g. a service user that runs the server) use this tree read-only through POSIX
# ACLs, with a shared writable kernel cache. Run as the owner after builds / pack changes; idempotent.
# usage: SHARE_USER=<account> share-acl.sh
# The other account then runs serve-bwrap.sh with the same WORK / MODELS_DIR (and, if needed, ROOTFS / DRV / RECIPE).
# Every parent directory of $WORK and $MODELS_DIR must be traversable by that account (x); add them with setfacl -m.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
U=${SHARE_USER:?set SHARE_USER}
ME=$(id -un)
# read + execute on the stack and the models, also for files created later (only entries we own; the cache holds
# files written by the other account, which inherit the cache's default ACL)
for d in "$WORK" "$MODELS_DIR"; do
  find "$d" -path "$CACHE_DIR" -prune -o -user "$ME" ! -type l -exec setfacl -m "u:$U:rX" {} +
  find "$d" -path "$CACHE_DIR" -prune -o -user "$ME" -type d -exec setfacl -m "d:u:$U:rX" {} +
done
# shared kernel cache: both accounts read/write, new entries inherit
mkdir -p "$CACHE_DIR"
find "$CACHE_DIR" -user "$ME" ! -type l -exec setfacl -m "u:$U:rwX,u:$ME:rwX" {} + 2>/dev/null || true
find "$CACHE_DIR" -user "$ME" -type d -exec setfacl -m "d:u:$U:rwX,d:u:$ME:rwX" {} + 2>/dev/null || true
# bind mount points inside the rootfs: bwrap cannot create them there when it runs as another account
mkdir -p "$ROOTFS$(dirname "$(readlink -f "$PACK")")" "$ROOTFS/root/.cache" "$ROOTFS/opt/recipe" \
  "$ROOTFS/usr/local/nvidia/lib64" "$ROOTFS/usr/local/nvidia/bin"
touch "$ROOTFS/usr/local/nvidia/bin/nvidia-smi"
echo "share-acl.sh: $U can read $WORK $MODELS_DIR and write $CACHE_DIR"
