# Results: DeepSeek-V4.1-Flash-1017, EXL3 3 bpw, 2× RTX PRO 6000 Max-Q

Measured on 2026-10-18 with the default configuration in `scripts/serve-bwrap.sh`. Chinese version: [REPORT.zh.md](REPORT.zh.md).

## Setup

| | |
|---|---|
| GPUs | 2× RTX PRO 6000 Blackwell Max-Q Workstation (96 GB, 300 W), PCIe, PHB between the cards, no NVLink, driver 595.91.07 (CUDA 13.2) |
| Host | 32 cores, 245 GiB RAM. The pack was read from an HDD at ~155 MB/s |
| Weights | PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw: routed experts EXL3 3 bpw, MUL1 codebook, **uncalibrated**; the rest FP8 as DeepSeek ships it. Engram tables from the official 1017 shards 47-48 converted to int4 (111 GB, pinned host RAM). DSpark drafter MXFP4 as stored |
| Server | vLLM nightly-5555a2d8 + diffbot's patches, TP2, `max_model_len` 1,048,576, fp8 KV cache 4.25 GiB (1.07M tokens), DSpark k=5, CUDA graphs, host-expert plan `plan-blend30-dh.json`, `NCCL_P2P_DISABLE=1`, vision encoder loaded |
| Memory | VRAM 89.9 GB/GPU idle, 95.4 GB/GPU peak at ~1M context. Host: Shmem 103 GiB (pinned Engram + cold experts), MemAvailable 59-65 GiB while serving |
| Cold start | ~85 min from HDD (see the README's known issues) |

## Speed

Measured directly against the server. Prompts are real Python source code.
- **Prefill:** one uncached request with `max_tokens=1` per stream.
- **Decode:** a streamed 256-token reply (`ignore_eos`) on the now-cached prefix, computed per stream from the token timestamps.
- Thinking off, warm server.

| Prompt (tokens) | Streams | Prefill tok/s | TTFT, uncached | Decode tok/s per stream | Decode tok/s total |
|---:|---:|---:|---:|---:|---:|
| 1.8K | 1 | 1,990 | 0.9 s | **92.8** | 92.8 |
| 1.8K | 2 | 2,043 | 1.6 s | 60.9 | 121.8 |
| 1.7K | 4 | 2,816 | 2.7 s | 33.1 | 132.5 |
| 26K | 1 | 5,179 | 5.1 s | **88.9** (97.4 in the first pass) | 88.9 |
| 26K | 2 | 5,196 | 10.2 s | 58.6 | 117.1 |
| 26K | 4 | 5,254 | 20.8 s | 40.5 | 162.1 |
| 109K | 1 | 5,085 | 21.5 s | **93.3** | 93.3 |
| 109K | 2 | 5,223 | 44.2 s | 70.4 | 140.8 |
| 101K | 4 | 5,360 | 81.0 s | 40.3 | 161.2 |
| 430K | 1 | 5,085 | 84.6 s | **99.5** | 99.5 |
| 890K | 1 | 4,866 | 182.9 s | **111.6** | 111.6 |

- **Thinking on**, single stream: 107.9 tok/s decode at 1.6K, 89.7 tok/s at 25K.
- **TTFT on a cached prefix:** 0.4-1.9 s.
- **DSpark acceptance** (vLLM `/metrics`):
  - real prompts of the quality suite: 2.97 of 5 draft tokens accepted per step, i.e. 3.97 tokens per step (59%);
  - synthetic "explain this code" generations: 3.27 tokens per step.

## Long context: needle in a haystack

A passphrase `CRANE-<6 digits>` is planted in Python source at 10/50/90% depth. **15/15 found.**

| Actual prompt length | 10% | 50% | 90% | Time per request |
|---|---|---|---|---|
| 24-29K | ✅ | ✅ | ✅ | 5-6 s |
| 105-116K | ✅ | ✅ | ✅ | 19-23 s |
| 430-463K | ✅ | ✅ | ✅ | 84-90 s |
| 886-888K | ✅ | ✅ | ✅ | 182 s |
| 988K-1,006K | ✅ | ✅ | ✅ | 205-209 s |

## Quality

- Fixed samples (seed 1234, `eval/make_samples.py`), temperature 0, scored automatically.
- `max_tokens`: 2,048 with thinking off, 16,384 with thinking on.
- HumanEval runs in a bubblewrap sandbox (no network, 15 s limit). MATH is scored with math-verify plus a units / `\text{}` normalization.
- Concurrency 8.

| Benchmark | n | Thinking off | Thinking on | Notes |
|---|---:|---:|---:|---|
| GSM8K (test) | 100 | **99%** | 98% | on: 1 item ran past 16K tokens of reasoning |
| MATH-500 | 50 | 88% | **96%** | off: 3 items cut at 2K tokens, 1 pmatrix-vs-tuple grading miss; on: the 2 misses are malformed final answers |
| HumanEval | 50 | **98%** | 90% (49/50 after re-running cut items one at a time) | on: all 5 failures were cut at 16K tokens; one at a time, 4 of them finish and pass |
| MMLU-Pro (test) | 200 | **75.0%** | 71.9% (64 items only, stopped early) | off: 14 cut at 2K; on: 9/64 cut at 16K, 46/55 = 83.6% of the rest |
| C-Eval (val) | 100 | **93%** | 92% | |
| Tool calling (OpenAI `tools`) | 10 | **10/10** | | English and Chinese, argument extraction, one "no tool" case |

**Reference points:**
- The diffbot model card (legacy V4.1, calibrated 3 bpw, same serving stack) reports GSM8K-200 98.0% and HumanEval 95.7%, thinking off, greedy. The samples differ and n is small, but we see no sign of quantization damage from the uncalibrated 1017 pack.
- Speed on the same card model, from the diffbot model card: 2K single-stream decode 74 tok/s (agent30 plan) to ~116-130 (blend30 / gen30 plans); 46K prefill ~10K tok/s.
- Here decode is similar or faster, and prefill is about half (see the README's known issues).

## Files

| File | Contents |
|---|---|
| `summary.json` | Every number above in machine-readable form |
| `data/speed.jsonl`, `data/speed-warm.jsonl` | Per-run speed records with per-stream detail (first pass, then a warm re-run of the short prompts) |
| `data/needle.jsonl` | Needle results |
| `data/quality/*.jsonl` | Per-item outputs, predictions, gold answers, token counts. `*.summary.json` holds the per-benchmark scores |
| `data/spec-metrics*.txt` | Raw DSpark counters (cumulative, sampled between runs) |
| `eval/` | `eval1017.py` (harness), `make_samples.py` (rebuilds the samples), `run_all.sh` |
