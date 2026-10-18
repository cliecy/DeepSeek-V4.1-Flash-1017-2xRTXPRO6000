# Notices

The code in this repository is MIT-licensed (`LICENSE`). Some files are derived from, or are diffs against, MIT-licensed
files of the diffbot recipe. Everything else this setup uses is **downloaded at build time from its upstream** and is
not redistributed here.

## Derived files in this repository

| File | Derived from | License |
|---|---|---|
| `scripts/serve-bwrap.sh` | `recipe/serve/serve.sh` of [diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) @ `6062be5f` | MIT, Copyright (c) 2026 Diffbot |
| `scripts/build-bwrap.sh` | `recipe/docker/build.sh` and `recipe/kernels/build.sh` of the same repository | MIT, Copyright (c) 2026 Diffbot |
| `scripts/assemble_pack_1017.py` | `recipe/quantize/assemble_pack.py` of the same repository | MIT, Copyright (c) 2026 Diffbot |
| `patches/*.diff` | diffs between the files above and their diffbot originals | MIT, Copyright (c) 2026 Diffbot (original lines) |
| `patches/llama.cpp/*.patch` | patches against [llama.cpp](https://github.com/ggml-org/llama.cpp) PR #28696 @ `2076412`; `0006` is by midagedev (vcruz305/llama.cpp#2) | MIT, Copyright (c) The ggml authors (context lines) |

The diffbot recipe's MIT license:

> MIT License — Copyright (c) 2026 Diffbot. Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal in the Software without restriction,
> including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
> of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
> Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

## Fetched at build or run time (not included)

| Component | Source | License |
|---|---|---|
| diffbot recipe (`recipe/`, incl. kernels, vLLM patches, `convert_engram_int4.py`) | diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 @ `6062be5f` | MIT, except `recipe/patches/vllm-exl3/` (AGPL-3.0-only); see its `THIRD_PARTY_NOTICES.md` |
| vllm-exl3 | [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) @ `d3cfd394` | AGPL-3.0-only (with MIT/Apache notices) |
| ExLlamaV3 | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) @ `5be88657` | MIT |
| sfxnz recipe helpers | [sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark) @ `460ac89c` | MIT |
| vLLM image | `vllm/vllm-openai` nightly-5555a2d8 ([vllm-project/vllm](https://github.com/vllm-project/vllm)) | Apache-2.0 (image contents carry their own licenses) |
| crane | [google/go-containerregistry](https://github.com/google/go-containerregistry) v0.22.1 | Apache-2.0 |
| Weights | [PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw](https://huggingface.co/PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw), [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | MIT |
| Evaluation datasets (`results/eval/make_samples.py`) | GSM8K, MATH-500, HumanEval, MMLU-Pro, C-Eval | their own licenses (C-Eval: CC BY-NC-SA 4.0); not redistributed. `results/data/quality/` holds model outputs and gold answers only |

Running the stack combines these components; the AGPL-3.0 terms of vllm-exl3 apply to that plugin (and the recipe's
patch to it) if you distribute or offer it over a network.
