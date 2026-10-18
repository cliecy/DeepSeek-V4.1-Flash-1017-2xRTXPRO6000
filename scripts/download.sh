#!/usr/bin/env bash
# Step 3: model downloads (resumable), both at pinned revisions:
#   PixelML 1017 EXL3 3 bpw routed experts + FP8 non-routed + MXFP4 drafter     223 GB
#   official 1017 Engram shards 47-48 (FP8) + config / index / tokenizer / encoding  203 GB
# Set DRAFTER_SRC=1 to also fetch the official drafter shards 44-46 (8 GB), only needed to re-encode the drafter
# as EXL3 3-bit with the diffbot recipe's quantize/quantize_mtp_exl3.py (not used for the published results).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
extra=(); [[ ${DRAFTER_SRC:-0} == 1 ]] && extra=("model-0004[4-6]-of-00048.safetensors")
hf_cli download "$PIXELML_REPO" --revision "$PIXELML_REV" --local-dir "$PIXELML_DIR" --max-workers 8 \
  > "$LOGS/dl-pixelml.log" 2>&1 &
hf_cli download "$BASE_REPO" --revision "$BASE_REV" --local-dir "$BASE_DIR" --max-workers 8 \
  --include "model-0004[78]-of-00048.safetensors" "${extra[@]}" model.safetensors.index.json config.json "tokenizer*" \
  "encoding/*" "inference/*" LICENSE README.md > "$LOGS/dl-base.log" 2>&1 &
wait
echo "download.sh: done (logs in $LOGS)"
