#!/usr/bin/env bash
# Step 5: serve. Rootless variant of the diffbot recipe's serve/serve.sh (MIT, Copyright (c) 2026 Diffbot): the same
# settings, but the server runs in the unpacked vllm-openai nightly-5555a2d8 rootfs through ctr.sh (bubblewrap) instead
# of `docker run`. Differences from upstream are listed in patches/serve.sh-to-serve-bwrap.sh.diff and the README.
# Serves DeepSeek-V4.1-Flash-1017 (EXL3 3.0 bpw routed experts, int4 Engram, DSpark drafter) with vLLM TP2 on
# 2x RTX PRO 6000 Blackwell (sm_120, 96 GB each), 1M-token context.
# Memory layout: the most-used routed experts live in VRAM; the coldest 4,608 target experts plus 176 drafter experts
# (plans/$PLAN; see the recipe's plans/README.md) stay in pinned host RAM and run on a side stream next to the hot ones.
# Prefill runs the VRAM experts through the decode-once kernel (kernels/xmoe_do, EXL3_DO=1). The Engram tables (int4)
# and the embedding table are pinned host RAM too, read over UVA (~190 GiB host RAM in total with PIN_ROUND_MB below).
# Text and image input (VISION=1). Foreground; Ctrl-C / SIGTERM stops the server. Requires API_KEY in the environment
# (clients send it as the Bearer token). Health check: GET /health; OpenAI-compatible API under /v1.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
REPO=$RECIPE                                # the diffbot recipe, mounted at /opt/recipe
: "${API_KEY:?set API_KEY}"
# The assembled pack (prepare_pack.sh); its parent directory is bound read-only at the same path.
PACK=$(readlink -f "$PACK")                 # bind the real path if $MODELS_DIR is a symlink
MODELS_DIR=$(dirname "$PACK")
CTR=${CTR:-$SCRIPTS/ctr.sh}
PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}                      # bind address; warm-up requests go here too
WARM_HOST=$([[ $HOST == 0.0.0.0 ]] && echo 127.0.0.1 || echo "$HOST")
SERVED=${SERVED:-deepseek-v4.1-flash-1017}
CONTAINER_MEM=${CONTAINER_MEM:-}            # optional cgroup cap via systemd-run --user (e.g. 240G); empty = none
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}      # 1M tokens (the model's full window); 524288 + KV_MEM=4294967296 = the earlier 512K setup
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_BATCHED=${MAX_BATCHED:-16384}           # prefill chunk; 16K beats 8K on fan-out and 4-stream decode
KV_MEM=${KV_MEM:-4563402752}                # 4.25 GiB fp8 KV = 1,071,558 tokens: one full 1M request (5.25 GiB = 1.32M)
NO_CUSTOM_AR=${NO_CUSTOM_AR-1}              # vLLM custom all-reduce fails CUDA graph capture here (custom_all_reduce.cuh:164)
LINEAR_BACKEND=${LINEAR_BACKEND-marlin}     # small dense MXFP8 layers; the big ones use the one-copy FP8 path below
XMOE_EXT=${XMOE_EXT-xmoe}                   # kernels/build/xmoe (kernels/build.sh); empty = stock ExLlamaV3 exl3_moe
EXL3_DO=${EXL3_DO-1}                        # 1 = decode-once prefill MoE kernel (kernels/build/xmoe_do); 0 = the fused kernel
UTIL=${UTIL:-0.92}
EAGER=${EAGER:-0}                           # 1 = no CUDA graphs
CG_REPLAY=${CG_REPLAY:-false}               # true = vLLM's decoder-replay CUDA graphs (upstream default; measured at parity here)
SPEC=${SPEC:-dspark}                        # none = no speculative decoding
SPEC_K=${SPEC_K:-5}                         # DSpark draft length; CAPTURE_SIZES must reach MAX_NUM_SEQS x (1 + SPEC_K)
FI_ARCH=${FI_ARCH:-12.0f}
THINKING=${THINKING:-default}               # default = model decides (on, effort 50); off = force thinking off
VISION=${VISION:-1}                         # 1 = load the vision encoder (0.77 GiB bf16; images via the image_url API); 0 = text only
PLAN=${PLAN:-plan-blend30-dh.json}          # host-expert plan in plans/ (agent30 / blend30 / gen30); results/ used blend30
WARMUP=${WARMUP:-1}                         # compile the remaining TileLang/Triton kernels before real traffic
[[ -f $PACK/config.json ]] || { echo "no config.json in PACK=$PACK"; exit 1; }
[[ -z ${XMOE_EXT:-} || -d $REPO/kernels/build/$XMOE_EXT ]] || { echo "kernels/build/$XMOE_EXT missing: run kernels/build.sh (or XMOE_EXT= for the stock kernel)"; exit 1; }
[[ $EXL3_DO != 1 || -d $REPO/kernels/build/xmoe_do ]] || { echo "kernels/build/xmoe_do missing: run kernels/build.sh (or EXL3_DO=0)"; exit 1; }
[[ -f $REPO/plans/$PLAN ]] || { echo "plans/$PLAN missing"; exit 1; }
export EXL3_DO
mkdir -p "$CACHE_DIR"
# Runtime knobs (all read by patches/vllm/sitecustomize.py or the patched vllm-exl3 plugin). Each can be overridden from the
# caller's environment; recipe/README.md explains them.
export EXL3_HOST_EXPERTS=${EXL3_HOST_EXPERTS:-/opt/recipe/plans/$PLAN}
export EXL3_HOST_OVERLAP_SMS=${EXL3_HOST_OVERLAP_SMS:-32}                  # SMs for the cold (host) experts during decode
export EXL3_HOST_OVERLAP_SMS_PREFILL=${EXL3_HOST_OVERLAP_SMS_PREFILL:-64}  # ... during prefill
export EXL3_HOST_OVERLAP_GROUP=${EXL3_HOST_OVERLAP_GROUP:-16}
export EXL3_FAT_SLICE=${EXL3_FAT_SLICE:-0}
export EXL3_LEAN_PREP=${EXL3_LEAN_PREP:-1}                                 # lean routing prep on decode-sized calls
export THOR_MHC_SPLITS=${THOR_MHC_SPLITS:-80}                              # mHC prenorm GEMM split cap (stock 16)
export THOR_DENSE_GEMV=${THOR_DENSE_GEMV:-1}                               # split-K FP8 GEMV for dense layers at decode sizes
export THOR_ROUTER_SPLITK=${THOR_ROUTER_SPLITK:-1}                         # router GEMM split-K + Triton partial-sum reduce
export THOR_DENSE_ONE_COPY=${THOR_DENSE_ONE_COPY:-1}                       # big dense layers keep one FP8 copy (no Marlin repack)
export THOR_DENSE_GEMV_MAX=${THOR_DENSE_GEMV_MAX:-64} THOR_DENSE_MBLOCK=${THOR_DENSE_MBLOCK:-64}
export THOR_DENSE_PY=${THOR_DENSE_PY:-/opt/recipe/kernels/thor_dense2.py}
export THOR_DENSE_TABLE=${THOR_DENSE_TABLE:-/opt/recipe/kernels/thor_dense_table_v2.json}
export THOR_WOA_FP8=${THOR_WOA_FP8:-1}                                     # attention wo_a held FP8 (exact requantize)
export THOR_EMBED_HOST=${THOR_EMBED_HOST:-1}                               # embedding table in pinned host RAM
export THOR_INDEXER_BUF_MULT=${THOR_INDEXER_BUF_MULT:-8}                   # indexer prefill workspace = max_model_len x 8
export THOR_INDEXER_EXPAND_CAP=${THOR_INDEXER_EXPAND_CAP:-1}               # indexer decode block-table buffer trimmed
export THOR_ROPE_CAP=${THOR_ROPE_CAP:-1}                                   # rope cache sized to max_model_len
export THOR_FP8_LMHEAD=${THOR_FP8_LMHEAD:-1}                               # FP8 lm_head
export DENSE_HYBRID_M=${DENSE_HYBRID_M:-128}                               # FlashInfer MXFP8 for dense prefill rows > 128
extra=()
CAPTURE_SIZES=${CAPTURE_SIZES:-"1 2 4 6 8 12 16 24 32 48"}   # 8 seqs x (1 + 5 DSpark tokens) = 48
[[ $EAGER == 1 ]] && extra+=(--enforce-eager) || extra+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"],"cudagraph_decoder_replay":'"$CG_REPLAY"'}' --cudagraph-capture-sizes $CAPTURE_SIZES)
[[ -n ${NO_CUSTOM_AR:-} ]] && extra+=(--disable-custom-all-reduce)
[[ -n ${LINEAR_BACKEND:-} ]] && extra+=(--linear-backend "$LINEAR_BACKEND")
[[ $THINKING == off ]] && extra+=(--default-chat-template-kwargs '{"thinking":false,"reasoning_effort":"low"}')
# DRAFT_PACK=<dir with only the drafter shards 44-46> (prepare_pack.sh makes $PACK-drafter): vLLM loads the DSpark draft
# model by reading its whole checkpoint, so with the full pack it re-reads all 311 GB (~23 min from HDD). UNTESTED.
DRAFT_PACK=${DRAFT_PACK:-}
[[ $SPEC == dspark ]] && extra+=(--speculative-config '{"method":"dspark","num_speculative_tokens":'"$SPEC_K"',"draft_sample_method":"probabilistic"'"${DRAFT_PACK:+,\"model\":\"$(readlink -f "$DRAFT_PACK")\"}"'}')
# pass-through into the container: EXL3_*/VLLM_EXL3_*, THOR_*, XMOE_*, NCCL_PROTO/ALGO, DENSE_HYBRID_M
PASS_ENV=$(env | grep -E "^(VLLM_EXL3_[A-Z0-9_]+|EXL3_[A-Z0-9_]+|THOR_[A-Z0-9_]+|XMOE_[A-Z_]+|NCCL_PROTO|NCCL_ALGO|DENSE_HYBRID_M)=" | sed "s/^/-e /" | tr "\n" " ")
cleanup() { [ -n "${WPID:-}" ] && kill $WPID 2>/dev/null; }   # bwrap --die-with-parent ends the server with this script
trap 'cleanup; exit 130' INT TERM
if [[ $WARMUP == 1 ]]; then
  ( t0=$(date +%s); until curl -sf -o /dev/null -H "Authorization: Bearer $API_KEY" "http://$WARM_HOST:$PORT/v1/models"; do sleep 5; [ $(( $(date +%s)-t0 )) -gt 3600 ] && exit 0; done
    echo "[warmup] start"; w0=$(date +%s)
    req() { curl -s -o /dev/null -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' "http://$WARM_HOST:$PORT/v1/completions" \
      -d "{\"model\":\"$SERVED\",\"prompt\":\"$(python3 -c "print(('warm up the kernels with some text. ' * $1).strip())")\",\"max_tokens\":$2,\"temperature\":0}"; }
    req 400 24; req 1600 24; for i in 1 2 3 4; do req 300 48 & done; wait; req 12 64
    echo "[warmup] done in $(( $(date +%s)-w0 ))s" ) &
  WPID=$!
fi
# torch 2.13's pinned host allocator rounds every pin_memory=True allocation up to a power of two by default; the int4
# Engram tables (~22.9 + 2.9 GiB per layer and rank) would take 32 + 4 GiB each, ~41 GiB of extra pinned RAM over the
# 4 tables. Allocations above PIN_ROUND_MB are made at their exact size.
PIN_ROUND_MB=${PIN_ROUND_MB:-1024}
# NCCL_P2P_DISABLE defaults to 1: on our 2x PCIe host (GPU0-GPU1 PHB, no NVLink, IOMMU on) NCCL's P2P transport hangs in
# the first all-reduce (nccl_test.py: P2P on never finishes, P2P off completes in ~1 s). Try 0 if nccl_test.py passes.
LIMIT=(); [[ -n $CONTAINER_MEM ]] && LIMIT=(systemd-run --user --scope -q -p MemoryMax="$CONTAINER_MEM" -p MemorySwapMax=0)
"${LIMIT[@]}" "$CTR" \
  -r "$MODELS_DIR:$MODELS_DIR" -b "$CACHE_DIR:/root/.cache" -r "$REPO:/opt/recipe" \
  -r "$REPO/patches/vllm/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py" \
  ${XMOE_EXT:+-e XMOE_EXT=$XMOE_EXT} \
  -e TORCH_CUDA_ARCH_LIST=12.0a -e FLASHINFER_CUDA_ARCH_LIST="$FI_ARCH" -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1} \
  -e PYTORCH_CUDA_ALLOC_CONF="${NO_CUSTOM_AR:+expandable_segments:True,}pinned_max_round_threshold_mb:$PIN_ROUND_MB" \
  -e VLLM_PLUGINS=vllm_exl3 -e DSV41_ENGRAM_DISK=0 -e DSV41_ENGRAM_INT4=1 -e VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=0 \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 -e CUDA_CACHE_PATH=/root/.cache/nv -e XDG_CONFIG_HOME=/root/.cache/config \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN \
  -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e TRITON_CACHE_DIR=/root/.cache/triton \
  ${CUDA_VISIBLE_DEVICES:+-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES} \
  $PASS_ENV \
  -- vllm serve "$PACK" \
  --tensor-parallel-size 2 --host "${HOST:-127.0.0.1}" --port "$PORT" --api-key "$API_KEY" \
  --max-model-len "$MAX_MODEL_LEN" --kv-cache-dtype fp8 --kv-cache-memory "$KV_MEM" \
  --gpu-memory-utilization "$UTIL" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
  --kernel-config '{"enable_flashinfer_autotune":false,"enable_jit_warmup":false}' \
  --block-size 64 --quantization exl3 $([[ $VISION == 1 ]] || echo --language-model-only) \
  --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --served-model-name "$SERVED" --trust-remote-code "${extra[@]}" ${EXTRA_ARGS:-}
rc=$?; cleanup; exit $rc
