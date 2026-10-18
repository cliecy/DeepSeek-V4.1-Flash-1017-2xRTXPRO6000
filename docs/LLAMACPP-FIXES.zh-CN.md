[English](LLAMACPP-FIXES.md) | 简体中文

# GGUF 路线：llama.cpp 的修复补丁

本仓库的主路线是 vLLM + EXL3 模型包。我们也在同样的两张 GPU 上用 llama.cpp 跑了 DeepSeek-V4.1-Flash-1017 的 GGUF。
llama.cpp 对 DeepSeek V4.1 的支持还没有合并，目前在
[ggml-org/llama.cpp#28696](https://github.com/ggml-org/llama.cpp/pull/28696)（`vcruz305:feat/deepseek-v41-convert`，
commit `2076412`）里。为了支持长上下文、提高多卡速度、让 `llama-imatrix` 能用，我们在这个 PR 之上加了 7 个补丁。

获取方式：
- [`patches/llama.cpp/`](../patches/llama.cpp/) 里的文件（`git format-patch` 的输出，每个 commit 一个文件）；
- 分支 [cliecy/llama.cpp `dsv41-1017-longctx`](https://github.com/cliecy/llama.cpp/tree/dsv41-1017-longctx)（最新 commit `1f91bd60a`）；
- 预编译的二进制文件在 GGUF 仓库
  [cliecy/DeepSeek-V4.1-Flash-1017-UD-IQ2XS-GGUF](https://huggingface.co/cliecy/DeepSeek-V4.1-Flash-1017-UD-IQ2XS-GGUF) 的 `bin/` 目录下。

这些补丁没有提交到上游。llama.cpp 的贡献规则要求提交 PR 的人自己撰写 PR 说明。

## 打补丁和编译

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git fetch origin pull/28696/head      # the PR branch; or clone the dsv41-1017-longctx branch above, which already has the patches
git checkout 2076412
git am /path/to/this/repo/patches/llama.cpp/*.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j --target llama-server llama-imatrix llama-quantize
```

（第二行取 PR 的分支；也可以直接克隆上面的 `dsv41-1017-longctx` 分支，它已经包含这些补丁。）
`120a` 对应 RTX PRO 6000（Blackwell，sm_120），其他 GPU 请换成对应的架构。

## 补丁列表

| # | Commit | 修复的问题 |
|---|---|---|
| 1 | CUDA: pad quantized src1 of MUL_MAT_ID by the widest MMQ tile | MoE 矩阵乘（MMQ）在几千 token 的 prompt batch 上可能读越过输入缓冲区的末尾，报 "illegal memory access" 崩溃。这是向后移植：ggml master 已经在 #29953 修复，rebase 到 master 时去掉这个 commit。Mermiges 在 PR 讨论里报告并给了复现方法。 |
| 2 | deepseek41: reuse the previous index source's top-k on reader layers | 原来只有计算 index 的层才做 top-k 选择，其他压缩层对所有压缩行做注意力。DeepSeek 的参考实现让这些层沿用上一个 index 源层的 top-k。改后与参考实现一致，也去掉了一项随上下文增长的开销。思路和 diff 来自 PR 讨论里的 nicholasshirley。 |
| 3 | deepseek41: implement the two-level candidate mask | V4.1 先选粗粒度的候选块（发布的模型：源层 20，2,048 个块、每块 8 个位置，共 16K 个位置），再在候选块里做细粒度的 top-k。PR 里只有一句关于 16K 限制的注释，所以超过这个长度的上下文无法正常工作。默认使用发布模型的配置；可选的 GGUF 键 `deepseek41.attention.indexer.candidate_layer`、`…candidate_top_k_blocks`、`…candidate_block_size` 可以覆盖，`LLAMA_DSV41_NO_CANDIDATES=1` 关闭这个功能。 |
| 4 | deepseek41: use the fused lightning indexer, bound the unfused fallback | V4 用了融合的 lightning indexer，V4.1 的移植漏掉了。非融合路径对每个 head 读一遍 index key（32 倍）：60K 上下文时单卡每 token 约 9 ms。现在改用融合算子；没有融合算子时，如果一次性缓冲区会超过 256 MiB，就按 head 逐个计算。 |
| 5 | deepseek41: gather the selected compressed rows at decode | decode 时每个压缩层都把整个压缩 cache 拼接起来，再把未选中的行屏蔽掉（60K 时每 token 约 3.5 ms）。现在只取出选中的 512 行。prefill 不变。 |
| 6 | resolve_fused_ops: a fused node placed on another device at a layer boundary is not missing support | 两卡切分时，切分边界上有一个节点放在另一张卡上，导致融合的 hyper-connection "pre" 算子在所有层上都被关闭。作者：midagedev（[vcruz305/llama.cpp#2](https://github.com/vcruz305/llama.cpp/pull/2)）。 |
| 7 | imatrix: skip nodes without a weight operand | `llama-imatrix` 在第一个 chunk 就崩溃（segfault），因为 V4.1 的计算图里有没有权重操作数的节点。 |

## 两卡切分：使用 `-ts 48,52`

这一条不需要改代码。默认的两卡按层切分把切分点放在第 21 层。第 20 层是 ratio-1 的压缩 KV 源，第 21–39 层都要读它，
于是它留在 GPU0 上，而读它的层在 GPU1 上。每个 token 都要把这份 KV 从 GPU0 经主机内存拷到 GPU1 约 18 次：
16K 上下文时每 token 约 340 MB，并随上下文线性增长。`-ts 48,52` 把切分点移到第 20 层之前。

## 结果

Q2_K GGUF，2× RTX PRO 6000 Max-Q，单路 decode，单位 tok/s：

| 上下文 | 原 PR | PR + 只加 `-ts 48,52` | PR + 补丁 + `-ts 48,52` |
|---:|---:|---:|---:|
| 512 | 44.9 | 40.8 | 42.2 |
| 4K | 36.5 | 39.8 | 47.3 |
| 16K | 23.1 | 34.2 | 45.3 |
| 32K | — | 28.6 | 47.0 |
| 64K | — | 21.7（60K 时） | 27.9 |
| 120K | — | — | 18.8 |

- prompt 处理：233–433 tok/s，120K 时约 200 tok/s。
- 48.5K 大海捞针 2/2 找到；wikitext-2 困惑度 4.749（Q2_K）。这两项是在补丁 4、5 之前测的。
- **遗留问题：** 64K 以上 decode 仍然变慢。原因是 CUDA 12.x 上的 `ggml_top_k`：CCCL 3.2 之前没有 `cub::DeviceTopK`。
  用 CUDA 13.2 编译预计能解决，尚未测量。
