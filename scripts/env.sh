# Shared settings, sourced by every script in this directory. Override any variable from the environment.
#   WORK        build tree: rootfs, recipe, sources, caches, logs. Put it on a fast local disk (NVMe) if you can.
#   MODELS_DIR  model downloads and the assembled pack (~540 GB peak, ~335 GB after cleanup). All of it must be one
#               filesystem: the pack is built from hardlinks.
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-$HOME/dsv41-1017}
MODELS_DIR=${MODELS_DIR:-$WORK/models}
ROOTFS=${ROOTFS:-$WORK/rootfs}            # unpacked vLLM image
DRV=${DRV:-$WORK/nvidia-driver}           # copy of the host driver's userspace libraries (filled by ctr.sh)
RECIPE=${RECIPE:-$WORK/recipe}            # diffbot recipe/ (kernels/build is written here)
SRC=${SRC:-$WORK/src}                     # git checkouts at the pinned commits
CACHE_DIR=${CACHE_DIR:-$WORK/cache}       # TileLang / Triton / FlashInfer / DeepGEMM kernel caches
LOGS=${LOGS:-$WORK/logs}

# Model directories
PIXELML_DIR=${PIXELML_DIR:-$MODELS_DIR/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-PixelML}
BASE_DIR=${BASE_DIR:-$MODELS_DIR/DeepSeek-V4.1-Flash-1017-src}          # official shards 47-48 + sidecars
ENGRAM_DIR=${ENGRAM_DIR:-$MODELS_DIR/DeepSeek-V4.1-Flash-1017-engram-int4}
PACK=${PACK:-$MODELS_DIR/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-thor}

# Pinned upstream versions (everything measured in results/ used exactly these)
VLLM_IMAGE=vllm/vllm-openai@sha256:8896fd7b29fc4ee7ce839f146cb4757984c476ec79d0234ffa8c995dee6e339a  # nightly-5555a2d8
DIFFBOT_REPO=diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000
DIFFBOT_REV=6062be5f0f43c50ba562ef4e7087d4fc83eea36c
PIXELML_REPO=PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw
PIXELML_REV=3d095b724ed4f3abb519cc1cda64c290a0e0b0c5
BASE_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
BASE_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
VLLM_EXL3_REF=d3cfd394920360d69f820d2dc96f8292a9e10283
EXLLAMAV3_REF=5be886578ec80324c2c715269387be2058724b6e
SFXNZ_REF=460ac89c7df1eea4db8a731f1feb807f1584f359
CRANE_VERSION=v0.22.1

hf_cli() {   # Hugging Face CLI: installed `hf`, else through uv
  if command -v hf >/dev/null; then hf "$@"; else uvx --from 'huggingface_hub[cli,hf_xet]' hf "$@"; fi
}
mkdir -p "$WORK" "$LOGS" 2>/dev/null || true
