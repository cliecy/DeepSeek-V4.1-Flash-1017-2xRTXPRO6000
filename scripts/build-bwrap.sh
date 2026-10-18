#!/usr/bin/env bash
# Step 2: rootless equivalent of the diffbot recipe's docker/build.sh + kernels/build.sh (no Docker, no GPU needed).
# Applies the same pinned stack to the unpacked vllm/vllm-openai:nightly-5555a2d8 rootfs in place:
#   vLLM patches (recipe/patches/vllm/*.diff), vllm-exl3 d3cfd394 + recipe/patches/vllm-exl3/exl3-thor.patch,
#   ExLlamaV3 5be88657 for sm_120a, the sfxnz helpers in /opt/dsv41-patch, then the xmoe / xmoe_do kernels into
#   $RECIPE/kernels/build. Takes ~30 min on 32 cores (mostly the ExLlamaV3 CUDA build).
# usage: build-bwrap.sh [step...]   steps: apt vllm plugin exllamav3 kernels (default: all, in order)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
JOBS=${MAX_JOBS:-8}
VP=/usr/local/lib/python3.12/dist-packages
run() { nice -n 19 "$SCRIPTS/ctr.sh" -r "$RECIPE/patches:/mnt/patches" -r "$SRC/sfxnz-recipe/docker/patch:/mnt/patch" \
  -r "$SRC/vllm-exl3:/mnt/vllm-exl3" -r "$SRC/exllamav3:/mnt/exllamav3" -b "$RECIPE/kernels:/k" \
  -e MAX_JOBS=$JOBS -e TORCH_CUDA_ARCH_LIST=12.0a -- bash -c "set -euxo pipefail; $1"; }
step_apt() { run '
  rm -rf /opt/dsv41-patch && cp -r /mnt/patch /opt/dsv41-patch
  apt-get -o APT::Sandbox::User=root update
  apt-get -o APT::Sandbox::User=root install -y --no-install-recommends git patch libcusparse-dev-13-0 libcusolver-dev-13-0'; }
step_vllm() { run "
  cd $VP
  for p in sm120-thor-fixes v41-engram-post-lifetime pr59585-dsml-finite-args pr61039-length-aware-candidate-blocks \
      pr61142-dsml-drifted-openers pr60792-v41-image-order; do
    patch -p1 --forward --no-backup-if-mismatch < /mnt/patches/vllm/\$p.diff
  done
  find $VP/vllm -name __pycache__ -prune -exec rm -rf {} +
  python3 -c 'import vllm.models.deepseek_v41.decoder_replay_layers, vllm.models.deepseek_v41.attention, vllm.models.deepseek_v41.common.engram, vllm.models.deepseek_v41.nvidia.model, vllm.models.deepseek_v41.nvidia.flashinfer_sparse, vllm.parser.deepseek_v4, vllm.parser.deepseek_v41; print(\"vllm patches ok\")'"; }
step_plugin() { run '
  rm -rf /opt/vllm-exl3 && cp -r /mnt/vllm-exl3 /opt/vllm-exl3 && cd /opt/vllm-exl3
  patch -p1 < /mnt/patches/vllm-exl3/exl3-thor.patch
  python3 /opt/dsv41-patch/fix_vllm_exl3_setup.py /opt/vllm-exl3/setup.py
  VLLM_EXL3_NO_CUDA=1 python3 -m pip install --no-build-isolation --no-deps /opt/vllm-exl3
  python3 -c "import vllm_exl3; print(\"vllm_exl3 ok\")"
  rm -rf /opt/vllm-exl3'; }
step_exllamav3() { run '
  rm -rf /opt/exllamav3 && cp -r /mnt/exllamav3 /opt/exllamav3
  python3 -m pip install --no-build-isolation --no-deps /opt/exllamav3
  python3 -c "import torch, exllamav3, exllamav3_ext; assert hasattr(exllamav3_ext, \"exl3_moe\"); print(\"exllamav3 ok\")"
  rm -rf /opt/exllamav3'; }
step_kernels() { run 'cd /k && python3 build_xmoe.py xmoe xmoe && python3 build_xmoe_do.py'; }
for s in ${@:-apt vllm plugin exllamav3 kernels}; do echo "=== step $s"; "step_$s"; done
echo "build-bwrap: done ($*)"
