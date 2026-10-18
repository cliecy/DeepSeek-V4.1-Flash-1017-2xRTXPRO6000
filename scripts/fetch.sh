#!/usr/bin/env bash
# Step 1: everything except the model weights. No root, no Docker.
#   - crane (go-containerregistry) into $WORK/bin
#   - the vLLM nightly-5555a2d8 image, exported to $ROOTFS (~23 GB after the build)
#   - the diffbot recipe (recipe/ + config.json) at its pinned revision
#   - git checkouts: sfxnz recipe, vllm-exl3, ExLlamaV3 at the pinned commits
# Idempotent: finished parts are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$WORK/bin" "$SRC"
CRANE=$WORK/bin/crane
if [[ ! -x $CRANE ]]; then
  curl -sL "https://github.com/google/go-containerregistry/releases/download/$CRANE_VERSION/go-containerregistry_Linux_x86_64.tar.gz" \
    | tar -xz -C "$WORK/bin" crane
fi
if [[ ! -f $ROOTFS/.export-done ]]; then
  mkdir -p "$ROOTFS"
  "$CRANE" export --platform linux/amd64 "$VLLM_IMAGE" - \
    | tar -x --no-same-owner --no-same-permissions -C "$ROOTFS" --exclude='dev/*'
  touch "$ROOTFS/.export-done"
fi
# Ubuntu's /usr/lib/python3.12/sitecustomize.py is an absolute symlink into /etc; bwrap cannot bind a file over it.
S=$ROOTFS/usr/lib/python3.12/sitecustomize.py
if [[ -L $S || ! -e $S ]]; then rm -f "$S"; cp "$ROOTFS/etc/python3.12/sitecustomize.py" "$S"; fi
mkdir -p "$ROOTFS/usr/local/nvidia/lib64" "$ROOTFS/usr/local/nvidia/bin" "$ROOTFS/opt/recipe" "$ROOTFS/root/.cache"
touch "$ROOTFS/usr/local/nvidia/bin/nvidia-smi"
# diffbot recipe at the pinned revision (small: scripts, patches, kernels, plans)
if [[ ! -f $WORK/diffbot/config.json ]]; then
  hf_cli download "$DIFFBOT_REPO" --revision "$DIFFBOT_REV" --include "recipe/**" config.json --local-dir "$WORK/diffbot"
fi
[[ -d $RECIPE/serve ]] || cp -r "$WORK/diffbot/recipe" "$RECIPE"
cp "$WORK/diffbot/config.json" "$WORK/config-diffbot.json"
clone() {   # clone URL DIR REF
  [[ -d $2/.git ]] || git clone -q "$1" "$2"
  git -C "$2" checkout -q "$3"
}
clone https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark "$SRC/sfxnz-recipe" "$SFXNZ_REF"
clone https://github.com/vcruz305/vllm-exl3.git "$SRC/vllm-exl3" "$VLLM_EXL3_REF"
clone https://github.com/turboderp-org/exllamav3.git "$SRC/exllamav3" "$EXLLAMAV3_REF"
echo "fetch.sh: rootfs $ROOTFS, recipe $RECIPE, sources $SRC ready"
