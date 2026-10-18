English | [简体中文](ARCHITECTURE.zh-CN.md)

# Architecture: requests, concurrency, VRAM and host RAM

![Serving architecture: client, vLLM scheduler, two GPUs with tensor parallelism, host RAM, disk](architecture.en.svg)

Numbers come from the runs in [results/](../results/README.md) with the default settings of `scripts/serve-bwrap.sh`.
Values marked *estimate* were not measured directly: they are computed from the model size or by subtracting
estimates from measured totals.

## Request path

1. A client sends an OpenAI-style request (`/v1/chat/completions`) to the vLLM server.
2. The vLLM scheduler runs up to 8 sequences at once (`--max-num-seqs 8`). Each running sequence takes KV cache blocks
   from one shared pool. Further requests wait in vLLM's queue until a slot and enough KV blocks are free.
3. **Prefill** runs in chunks of up to 16,384 tokens (`--max-num-batched-tokens 16384`). An 890K-token prompt takes
   ~3 min at ~5,000 tok/s.
4. **Decode** uses DSpark speculative decoding with k=5: the drafter proposes 5 tokens and the main model checks them
   in one step. On the real prompts of the quality suite, 2.97 of the 5 were accepted on average, so each step yields
   3.97 tokens (59% acceptance).
5. Every step runs on both GPUs with tensor parallelism 2: each GPU holds half of every layer.
   - Hot routed experts (70%) are in VRAM.
   - Cold routed experts (30%, plan `plan-blend30-dh.json`) stay in pinned host RAM. The GPU reads them over PCIe on a
     separate CUDA stream, next to the hot experts.
   - The int4 Engram tables and the embedding table are in pinned host RAM and are read over PCIe as well.
   - The all-reduce between the two GPUs goes through host shared memory: `NCCL_P2P_DISABLE=1`, because NCCL's P2P
     transport hangs on the test host (see the README's known issues).
6. At start the server reads the 334 GB pack from disk once: the hot experts and the non-routed layers go to the
   GPUs, the Engram tables and the cold experts to host RAM. From an HDD this takes ~85 min.

## Concurrency

Decode speed, thinking off:

| Concurrent streams | tok/s per stream | tok/s total | Source |
|---:|---:|---:|---|
| 1 | 89–112 | 89–112 | speed suite, 1.8K–890K prompts |
| 2 | 59–70 | 117–141 | speed suite, 1.8K–109K prompts |
| 4 | 33–40 | 132–162 | speed suite, 1.7K–101K prompts |
| 8 | ~14 | ~110 | quality suite, long reasoning outputs |

- Per-stream speed stays at or above 50 tok/s up to 2 concurrent streams.
- More than 8 requests are not rejected: they wait in vLLM's queue.
- **The KV pool is shared.** It holds ~1.07M tokens (fp8, 4.25 GiB in total, ~2.1 GiB per GPU). `--max-model-len 1048576`
  only caps a single request. One 1M-token request fills the whole pool; 8 running requests get ~130K tokens each on
  average. If the pool runs out, vLLM delays or preempts requests.
- **Why the total stops growing after about 4 streams** (not measured in isolation):
  - more sequences in a step route to more distinct experts, so each step reads more cold experts over PCIe;
  - the gain from speculative decoding shrinks as the batch grows, because the drafted tokens of every sequence have to
    be verified in the same step.

## GPU memory, per GPU

97.9 GB per card as reported by nvidia-smi (in thousands of MiB, so the GB figures below are rounded).
Measured: 89.9 GB used at idle, 95.4 GB peak at ~1M context. vLLM sizes its budget with `--gpu-memory-utilization
0.92`; the peak above that is not covered by vLLM's own accounting.

| Part | Size | How we know |
|---|---:|---|
| Hot routed experts (70%), EXL3 3 bpw | ~71 GB | estimate: 543.6B routed parameters × 3 bit ≈ 204 GB, × 0.7, / 2 GPUs |
| Non-routed FP8 layers (attention, shared experts, output head), DSpark drafter (MXFP4), vision encoder, CUDA graphs, activations and workspace for 16K-token prefill chunks | ~20 GB | estimate: the measured idle total minus the other two rows |
| KV cache, fp8 | ~2.1 GiB | set by `KV_MEM` (4.25 GiB for both GPUs) |
| Used at idle / peak | 89.9 / 95.4 GB | measured |

## Host RAM

The test host has 245 GiB.

| Part | Size | How we know |
|---|---:|---|
| Engram tables, int4, pinned | 103 GiB | measured (Shmem); the int4 shards are 111 GB on disk |
| Cold routed experts (30%), pinned | ~60 GB (~56 GiB) | estimate: 204 GB × 0.3 |
| Embedding table, vLLM processes, buffers, rest | ~22 GiB | estimate: what remains |
| Free (MemAvailable) while serving | 59–65 GiB | measured |
| In use while serving | ~180–186 GiB | measured (245 GiB − MemAvailable) |

The pinned allocations are made with `PYTORCH_CUDA_ALLOC_CONF=pinned_max_round_threshold_mb:1024` (`PIN_ROUND_MB`).
By default, torch 2.13 rounds each pinned allocation up to a power of two. For the Engram tables that wastes ~41 GiB:
the diffbot recipe reports ~230 GiB of host RAM, and this host would not have enough.

## Tuning knobs

None of these were measured here.

- **More concurrent requests.** Set `MAX_NUM_SEQS=16` and extend `CAPTURE_SIZES` up to 96 (16 × (1 + `SPEC_K`)). The 16
  requests then share the same ~1.07M-token pool (~67K each on average), so also lower `MAX_MODEL_LEN` to 524288 or
  262144, so that one long request cannot take the whole pool. Expect lower per-stream speed: total throughput already
  levels off at 4 streams.
- **Faster single streams.** Build a cold-expert plan from 1017 routing statistics: collect them with
  `EXL3_EXPERT_STATS`, then run the recipe's `plans/make_plan.py`, and set `PLAN`. The shipped plans come from routing
  statistics of the legacy model.
- **Faster prefill.** If `scripts/nccl_test.py` passes with P2P on (for example after booting with `iommu=pt`), set
  `NCCL_P2P_DISABLE=0`. We expect up to ~2× prefill: the diffbot card reports ~10K tok/s on the same GPUs, against
  ~5K here.
- **Larger KV pool.** `KV_MEM=5637144576` (5.25 GiB) gives ~1.32M tokens, but the measured peak (95.4 of 97.9 GB)
  leaves little room.
