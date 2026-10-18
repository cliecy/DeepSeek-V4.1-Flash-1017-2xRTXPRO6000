#!/usr/bin/env bash
# Rootless stand-in for `docker run` (no Docker needed): bubblewrap + an unprivileged user namespace.
# Runs a command inside the unpacked vLLM rootfs ($ROOTFS) with the host NVIDIA driver libraries, host network/IPC,
# /dev (GPUs, /dev/shm) and the given bind mounts.
# usage: ctr.sh [-b HOST:CTR]... [-r HOST:CTR]... [-e K=V]... -- CMD ARGS...
#   -b read-write bind, -r read-only bind, -e environment variable
# $ROOTFS is bound read-write, so build steps (apt, pip install) persist in it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
binds=() envs=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -b) binds+=(--bind "${2%%:*}" "${2#*:}"); shift 2 ;;
    -r) binds+=(--ro-bind "${2%%:*}" "${2#*:}"); shift 2 ;;
    -e) envs+=(--setenv "${2%%=*}" "${2#*=}"); shift 2 ;;
    --) shift; break ;;
    *) echo "ctr.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
[[ $# -gt 0 ]] || { echo "ctr.sh: no command" >&2; exit 2; }
# Collect the host driver's userspace libraries once (what nvidia-container-toolkit injects for compute,utility).
# Re-run with an empty $DRV after a driver update.
if [[ ! -e $DRV/libcuda.so.1 ]]; then
  mkdir -p "$DRV"
  LIBDIR=${NV_LIBDIR:-/usr/lib/x86_64-linux-gnu}
  for f in "$LIBDIR"/lib{cuda,cudadebugger,nvidia-ml,nvidia-ptxjitcompiler,nvidia-nvvm,nvidia-nvvm70,nvidia-gpucomp,nvidia-cfg,nvidia-allocator}.so*; do
    cp -a "$f" "$DRV/"
  done
fi
image_env=(
  --setenv PATH /usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  --setenv LD_LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/cuda/lib64
  --setenv CUDA_HOME /usr/local/cuda --setenv CUDA_VERSION 13.0.2
  --setenv VLLM_ENABLE_CUDA_COMPATIBILITY 0 --setenv HOME /root --setenv USER root
  --setenv NVIDIA_VISIBLE_DEVICES all --setenv NVIDIA_DRIVER_CAPABILITIES compute,utility
  --setenv DEBIAN_FRONTEND noninteractive
  --setenv UV_PYTHON_INSTALL_DIR /opt/uv/python --setenv UV_CACHE_DIR /opt/uv/cache --setenv UV_LINK_MODE copy
  --setenv TERM "${TERM:-xterm}"
)
exec bwrap \
  --unshare-user --uid 0 --gid 0 --unshare-pid --die-with-parent \
  --bind "$ROOTFS" / \
  --dev-bind /dev /dev --proc /proc --ro-bind /sys /sys \
  --ro-bind /etc/resolv.conf /etc/resolv.conf --ro-bind /etc/hosts /etc/hosts \
  --ro-bind "$DRV" /usr/local/nvidia/lib64 \
  --ro-bind "$(command -v nvidia-smi)" /usr/local/nvidia/bin/nvidia-smi \
  --tmpfs /tmp \
  --clearenv "${image_env[@]}" "${envs[@]}" "${binds[@]}" \
  --chdir / \
  "$@"
