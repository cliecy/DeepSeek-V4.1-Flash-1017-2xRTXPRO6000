#!/usr/bin/env bash
# Step 4 (CPU only, ~50 min, peak RSS ~121 GiB): build the serving pack for DeepSeek-V4.1-Flash-1017.
#   1. verify the official Engram shards 47-48 (sha256)
#   2. int4 Engram: the diffbot recipe's quantize/convert_engram_int4.py, unchanged (FP8 203 GB -> int4 111 GB)
#   3. assemble the pack with hardlinks: PixelML shards 1-46 + int4 Engram 47-48, index rebuilt, diffbot config.json
#      with the drafter switched to source-format (MXFP4) experts (assemble_pack_1017.py)
#   4. $PACK-drafter: hardlinks of the drafter shards 44-46 only, for serve-bwrap.sh DRAFT_PACK= (UNTESTED, see README)
# Shortcut: download the prebuilt int4 Engram delta pack (cliecy/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-2xRTXPRO6000-delta,
# see README) into $ENGRAM_DIR; then steps 1-2 are skipped (verified against its sha256sums.txt) and $BASE_DIR is not needed.
# Runs inside the rootfs (torch, safetensors) through ctr.sh. Idempotent. After it, $BASE_DIR/model-0004[78]* (203 GB)
# can be deleted. Set DRAFTER=<quantize_mtp_exl3.py output dir> to use an EXL3 3-bit drafter instead.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DRAFTER=${DRAFTER:-}
C() { nice -n 10 "$SCRIPTS/ctr.sh" -b "$MODELS_DIR:$MODELS_DIR" -r "$RECIPE:$RECIPE" -r "$SCRIPTS:$SCRIPTS" -r "$WORK:$WORK" -- "$@"; }
[[ -f $PIXELML_DIR/model-00046-of-00048.safetensors ]] || { echo "missing $PIXELML_DIR (run download.sh first)"; exit 1; }
if [[ -f $ENGRAM_DIR/sha256sums.txt && -f $ENGRAM_DIR/model-00048-of-00048.safetensors ]]; then
  echo "using the prebuilt int4 Engram delta pack in $ENGRAM_DIR"
  ( cd "$ENGRAM_DIR" && sha256sum -c sha256sums.txt ) || { echo "delta pack checksum mismatch"; exit 1; }
else
  for f in "$BASE_DIR"/model-0004{7,8}-of-00048.safetensors; do
    [[ -f $f ]] || { echo "missing $f (run download.sh first)"; exit 1; }
  done
  # sha256 of the 1017 Engram shards at revision 1bae0e10
  ( cd "$BASE_DIR" && sha256sum -c <<'EOF' ) > "$LOGS/sha256-engram.log" 2>&1 &
76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5  model-00047-of-00048.safetensors
9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd  model-00048-of-00048.safetensors
EOF
  SHA=$!
  if [[ ! -f $ENGRAM_DIR/model-00048-of-00048.safetensors ]]; then
    C python3 -u "$RECIPE/quantize/convert_engram_int4.py" "$BASE_DIR" "$ENGRAM_DIR"
  fi
  wait $SHA || true; cat "$LOGS/sha256-engram.log"
  grep -q FAILED "$LOGS/sha256-engram.log" && { echo "Engram shard checksum mismatch"; exit 1; }
fi
# tokenizer / encoding sidecars: from the official snapshot if present, else from the PixelML repo (same files)
SIDECARS=$BASE_DIR; [[ -f $SIDECARS/tokenizer.json ]] || SIDECARS=$PIXELML_DIR
C python3 "$SCRIPTS/assemble_pack_1017.py" --experts "$PIXELML_DIR" --engram "$ENGRAM_DIR" --sidecars "$SIDECARS" \
  --config "$WORK/config-diffbot.json" --out "$PACK" ${DRAFTER:+--drafter "$DRAFTER"}
# drafter-only directory: vLLM loads the DSpark draft model by reading its whole checkpoint directory
D=$PACK-drafter; mkdir -p "$D"
for f in model-0004{4,5,6}-of-00048.safetensors config.json tokenizer.json tokenizer_config.json; do
  [[ -e $D/$f ]] || ln "$PACK/$f" "$D/$f"
done
[[ -e $D/encoding ]] || cp -al "$PACK/encoding" "$D/encoding"
python3 - "$PACK" "$D" <<'EOF'
import json, sys
pack, d = sys.argv[1:]
keep = {f"model-{n:05d}-of-00048.safetensors" for n in (44, 45, 46)}
wm = {k: v for k, v in json.load(open(f"{pack}/model.safetensors.index.json"))["weight_map"].items() if v in keep}
json.dump({"metadata": {}, "weight_map": wm}, open(f"{d}/model.safetensors.index.json", "w"), indent=1)
print(f"{d}: {len(wm)} drafter tensors")
EOF
echo "prepare_pack.sh: pack ready at $PACK"
