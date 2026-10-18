[English](ARCHITECTURE.md) | 简体中文

# 部署结构：请求、并发、显存和主机内存

![部署结构：客户端、vLLM 调度器、张量并行的两张 GPU、主机内存、磁盘](architecture.zh-CN.svg)

数字来自 [results/](../results/REPORT.zh.md) 里的评测，使用 `scripts/serve-bwrap.sh` 的默认设置。
标为“估算”的值没有直接测量，是按模型大小推算，或用实测总量减去其他估算值得到的。

## 一个请求怎么走

1. 客户端把 OpenAI 格式的请求（`/v1/chat/completions`）发给 vLLM 服务。
2. vLLM 调度器最多同时运行 8 条序列（`--max-num-seqs 8`）。每条运行中的序列从同一个共享池里取 KV cache 块。
   更多的请求在 vLLM 的队列里等待，直到有空位、KV 块也够用。
3. **prefill** 按最多 16,384 token 一块分块执行（`--max-num-batched-tokens 16384`）。prefill 约 5,000 tok/s，
   890K token 的 prompt 约需 3 分钟。
4. **decode** 用 DSpark 投机解码，k=5：drafter 先提出 5 个 token，主模型在一步里验证。在质量评测的真实 prompt 上，
   平均每步接受 5 个中的 2.97 个，即每步产出 3.97 个 token（接受率 59%）。
5. 每一步都在两张卡上以张量并行 2 运行：每张卡存每一层的一半。
   - 热路由专家（70%）在显存里。
   - 冷路由专家（30%，方案 `plan-blend30-dh.json`）留在锁页的主机内存里，GPU 在单独的 CUDA stream 上经 PCIe 读取，
     和热专家同时计算。
   - int4 Engram 表和 embedding 表也在锁页的主机内存里，同样经 PCIe 读取。
   - 两张卡之间的 all-reduce 经主机共享内存完成：设了 `NCCL_P2P_DISABLE=1`，因为测试机上 NCCL 的 P2P 传输会挂死
     （见 README 的已知问题）。
6. 启动时从磁盘把 334 GB 的模型包读一遍：热专家和非路由层进显存，Engram 表和冷专家进主机内存。从 HDD 读约需 85 分钟。

## 并发

decode 速度，思考关：

| 并发路数 | 每路 tok/s | 总 tok/s | 来源 |
|---:|---:|---:|---|
| 1 | 89–112 | 89–112 | 速度测试，prompt 1.8K–890K |
| 2 | 59–70 | 117–141 | 速度测试，prompt 1.8K–109K |
| 4 | 33–40 | 132–162 | 速度测试，prompt 1.7K–101K |
| 8 | 约 14 | 约 110 | 质量评测，推理输出较长 |

- 并发不超过 2 路时，每路都能保持 50 tok/s 以上。
- 超过 8 个的请求不会被拒绝，而是在 vLLM 的队列里等待。
- **KV 池是共享的。** 共约 107 万 token（fp8，总共 4.25 GiB，每卡约 2.1 GiB）。`--max-model-len 1048576`
  只限制单个请求的长度。一个 1M token 的请求就占满整个池；8 个请求同时运行时，平均每个约 13 万 token。
  池不够时，vLLM 会推迟或抢占请求。
- **为什么超过约 4 路后总吞吐不再增长**（没有单独测量）：
  - 一步里的序列越多，路由到的不同专家越多，每步要经 PCIe 读的冷专家也越多；
  - batch 越大，投机解码的收益越小，因为每条序列草拟的 token 都要在同一步里验证。

## 显存（每张卡）

nvidia-smi 显示每张卡 97.9 GB（单位是千 MiB，所以下面的 GB 数都是约数）。
实测：空闲时用 89.9 GB，约 1M 上下文时峰值 95.4 GB。vLLM 按 `--gpu-memory-utilization 0.92` 计算自己的预算，
峰值中超出的部分不在 vLLM 自己的统计里。

| 部分 | 大小 | 依据 |
|---|---:|---|
| 热路由专家（70%），EXL3 3 bpw | 约 71 GB | 估算：543.6B 路由参数 × 3 bit ≈ 204 GB，× 0.7，÷ 2 张卡 |
| 非路由 FP8 层（attention、共享专家、输出 head）、DSpark drafter（MXFP4）、视觉编码器、CUDA graph、16K 分块 prefill 的激活和工作区 | 约 20 GB | 估算：实测空闲总量减去另外两行 |
| KV cache，fp8 | 约 2.1 GiB | 由 `KV_MEM` 设定（两张卡共 4.25 GiB） |
| 空闲时 / 峰值 | 89.9 / 95.4 GB | 实测 |

## 主机内存

测试机共 245 GiB。

| 部分 | 大小 | 依据 |
|---|---:|---|
| Engram 表，int4，锁页 | 103 GiB | 实测（Shmem）；int4 shard 在磁盘上是 111 GB |
| 冷路由专家（30%），锁页 | 约 60 GB（约 56 GiB） | 估算：204 GB × 0.3 |
| embedding 表、vLLM 进程、缓冲区等 | 约 22 GiB | 估算：剩余部分 |
| serving 时空闲（MemAvailable） | 59–65 GiB | 实测 |
| serving 时已用 | 约 180–186 GiB | 实测（245 GiB 减 MemAvailable） |

锁页内存用 `PYTORCH_CUDA_ALLOC_CONF=pinned_max_round_threshold_mb:1024`（即 `PIN_ROUND_MB`）分配。
torch 2.13 默认把每次锁页分配向上取整到 2 的幂，Engram 表因此多占约 41 GiB：diffbot 原方案报告主机内存约 230 GiB，
这台机器放不下。

## 可调参数

以下都没有在本机测量过。

- **更多并发请求。** 设 `MAX_NUM_SEQS=16`，并把 `CAPTURE_SIZES` 扩展到 96（16 × (1 + `SPEC_K`)）。这 16 个请求仍然共用
  约 107 万 token 的池（平均每个约 6.7 万），所以同时把 `MAX_MODEL_LEN` 降到 524288 或 262144，免得一个长请求占满整个池。
  每路速度会更低：总吞吐在 4 路时就已经不再增长。
- **单路更快。** 用 1017 的路由统计重新生成冷专家方案：用 `EXL3_EXPERT_STATS` 收集统计，再运行 recipe 里的
  `plans/make_plan.py`，然后设置 `PLAN`。现有的方案来自 legacy 模型的路由统计。
- **prefill 更快。** 如果开启 P2P 后 `scripts/nccl_test.py` 能通过（例如开机参数加 `iommu=pt`），设
  `NCCL_P2P_DISABLE=0`。预计 prefill 最多快约 2 倍：diffbot 的模型卡在同样的 GPU 上报告约 10K tok/s，这里约 5K。
- **更大的 KV 池。** `KV_MEM=5637144576`（5.25 GiB）约 132 万 token，但实测峰值（97.9 GB 里用了 95.4 GB）
  留下的余量很少。
