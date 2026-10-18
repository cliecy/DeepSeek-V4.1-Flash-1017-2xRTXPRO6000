#!/usr/bin/env bash
# The full suite behind results/README.md, against a running serve-bwrap.sh.
# usage: BASE=http://127.0.0.1:8000/v1 API_KEY=... MODEL=deepseek-v4.1-flash-1017 run_all.sh [speed|quality|all]
#   METRICS=http://127.0.0.1:8000/metrics  (DSpark acceptance counters, optional)
# Samples: python3 make_samples.py samples   (run once; needs pandas, pyarrow, huggingface_hub; math-verify for MATH)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
BASE=${BASE:-http://127.0.0.1:8000/v1}; MODEL=${MODEL:-deepseek-v4.1-flash-1017}; OUT=${OUT:-out}
METRICS=${METRICS:-${BASE%/v1}/metrics}
E=(python3 eval1017.py --base "$BASE" --model "$MODEL" --key "${API_KEY:?set API_KEY}" --out "$OUT")
mkdir -p "$OUT"
m() { curl -s -H "Authorization: Bearer $API_KEY" "$METRICS" | grep -E "^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total" \
        | sed 's/{[^}]*}//' | awk -v t="$1" '{print t, $1, $2}' >> "$OUT/spec-metrics.txt"; }
if [[ ${1:-all} != quality ]]; then
  m start
  "${E[@]}" speed --ctx 2000,32000,131072 --conc 1,2,4 --think off --gen 256
  "${E[@]}" speed --ctx 2000,32000 --conc 1 --think on --gen 256
  "${E[@]}" speed --ctx 524288,1000000 --conc 1 --think off --gen 256
  m after-speed
  "${E[@]}" needle --ctx 32000,131072,524288,1040000,1160000 --depth 10,50,90
  m after-needle
fi
if [[ ${1:-all} != speed ]]; then
  "${E[@]}" tools --think off
  for t in gsm8k humaneval mmlupro ceval math500; do "${E[@]}" $t --think off --conc 8; done
  for t in gsm8k math500 humaneval ceval mmlupro; do "${E[@]}" $t --think on --conc 8; done
  m after-quality
fi
