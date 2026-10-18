English | [简体中文](LLAMACPP-FIXES.zh-CN.md)

# llama.cpp fixes for the GGUF route

The main route in this repository is vLLM with the EXL3 pack. We also ran DeepSeek-V4.1-Flash-1017 as a GGUF with
llama.cpp on the same two GPUs. DeepSeek V4.1 support in llama.cpp is not merged yet: it lives in
[ggml-org/llama.cpp#28696](https://github.com/ggml-org/llama.cpp/pull/28696) (`vcruz305:feat/deepseek-v41-convert`,
commit `2076412`). On top of that PR we needed 7 patches for long contexts, multi-GPU speed and `llama-imatrix`.

They are available as:
- the files in [`patches/llama.cpp/`](../patches/llama.cpp/) (`git format-patch` output, one per commit);
- the branch [cliecy/llama.cpp `dsv41-1017-longctx`](https://github.com/cliecy/llama.cpp/tree/dsv41-1017-longctx)
  (tip `1f91bd60a`);
- prebuilt binaries in `bin/` of the GGUF repo
  [cliecy/DeepSeek-V4.1-Flash-1017-UD-IQ2XS-GGUF](https://huggingface.co/cliecy/DeepSeek-V4.1-Flash-1017-UD-IQ2XS-GGUF).

These patches are not submitted upstream. llama.cpp's contribution policy requires the person who submits a PR to
write its description themselves.

## Apply and build

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git fetch origin pull/28696/head      # the PR branch; or clone the dsv41-1017-longctx branch above, which already has the patches
git checkout 2076412
git am /path/to/this/repo/patches/llama.cpp/*.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j --target llama-server llama-imatrix llama-quantize
```

`120a` is for RTX PRO 6000 (Blackwell, sm_120); use the architecture of your GPUs.

## The patches

| # | Commit | What it fixes |
|---|---|---|
| 1 | CUDA: pad quantized src1 of MUL_MAT_ID by the widest MMQ tile | The MoE matmul (MMQ) could read past its input buffer on prompt batches of a few thousand tokens and crash with "illegal memory access". Backport: ggml master already has the fix as #29953. Drop this commit when rebasing onto master. Reported with a reproducer by Mermiges in the PR thread. |
| 2 | deepseek41: reuse the previous index source's top-k on reader layers | Only the layers that compute an index built a top-k selection; the other compressed layers attended to all compressed rows. DeepSeek's reference code has them reuse the top-k of the previous index-source layer. Now they do, which matches the reference and removes a cost that grew with context. Idea and diff from nicholasshirley in the PR thread. |
| 3 | deepseek41: implement the two-level candidate mask | V4.1 first picks coarse candidate blocks (on the released checkpoint: source layer 20, 2,048 blocks of 8 positions, i.e. 16K positions), then makes the fine top-k selection inside them. The PR only had a comment about the 16K limit, so contexts longer than that did not work. Defaults to the released config; optional GGUF keys `deepseek41.attention.indexer.candidate_layer`, `…candidate_top_k_blocks`, `…candidate_block_size` override it, and `LLAMA_DSV41_NO_CANDIDATES=1` turns it off. |
| 4 | deepseek41: use the fused lightning indexer, bound the unfused fallback | V4 uses the fused lightning indexer; the V4.1 port did not. The unfused path read the index keys once per head (32×): ~9 ms per token at 60K context on one GPU. Now the fused op is used; when it is not available, the unfused path is computed head by head once its buffer would exceed 256 MiB. |
| 5 | deepseek41: gather the selected compressed rows at decode | At decode, every compressed layer concatenated the whole compressed cache and masked all but the selected rows (~3.5 ms per token at 60K). Now it gathers only the 512 selected rows. Prefill is unchanged. |
| 6 | resolve_fused_ops: a fused node placed on another device at a layer boundary is not missing support | On a two-GPU split, the fused hyper-connection "pre" op was disabled on every layer because one node at the split boundary sat on the other GPU. Author: midagedev ([vcruz305/llama.cpp#2](https://github.com/vcruz305/llama.cpp/pull/2)). |
| 7 | imatrix: skip nodes without a weight operand | `llama-imatrix` crashed (segfault) on the first chunk, because the V4.1 graph has nodes with no weight operand. |

## Tensor split: use `-ts 48,52` on 2 GPUs

This needs no code change. With the default 2-GPU layer split, the split falls at layer 21. Layer 20 is the ratio-1
compressed-KV source that layers 21–39 read, so it stays on GPU0 while its readers are on GPU1. Every token then copies
that KV from GPU0 through host memory to GPU1 about 18 times: ~340 MB per token at 16K context, growing linearly with
context. `-ts 48,52` moves the split before layer 20.

## Results

Q2_K GGUF, 2× RTX PRO 6000 Max-Q, single-stream decode in tok/s:

| Context | Original PR | PR + `-ts 48,52` only | PR + patches + `-ts 48,52` |
|---:|---:|---:|---:|
| 512 | 44.9 | 40.8 | 42.2 |
| 4K | 36.5 | 39.8 | 47.3 |
| 16K | 23.1 | 34.2 | 45.3 |
| 32K | — | 28.6 | 47.0 |
| 64K | — | 21.7 (at 60K) | 27.9 |
| 120K | — | — | 18.8 |

- Prompt processing: 233–433 tok/s, ~200 tok/s at 120K.
- Needle at 48.5K: 2/2 found. wikitext-2 perplexity 4.749 (Q2_K). Both were measured before patches 4 and 5.
- **Remaining issue:** decode still slows down from 64K on. The cause is `ggml_top_k` on CUDA 12.x, which has no
  `cub::DeviceTopK` (added in CCCL 3.2). A CUDA 13.2 build is expected to fix this; not measured.
