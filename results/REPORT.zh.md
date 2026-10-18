# DeepSeek-V4.1-Flash-1017 · EXL3 3bpw 在 2× RTX PRO 6000 Max-Q 上的评测

评测日期 2026-10-18。英文版和文件说明见 [README.md](README.md)。

## 配置
- **硬件**：2× RTX PRO 6000 Max-Q（96 GB，300 W），PCIe 连接，两卡之间是 PHB，没有 NVLink；主机内存 245 GiB；模型放在 HDD 上（约 155 MB/s）。
- **权重**：PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw。
  - 路由专家：EXL3 3 bpw，mul1 codebook，**未校准**；其余层保持 FP8。
  - Engram 表：取 1017 官方 shard 47/48，用 diffbot 的 `convert_engram_int4.py` 转成 int4（111 GB），放在主机内存里锁页。
  - DSpark drafter：MXFP4 原样。
  - 按 diffbot 的包格式组装：48 个 shard，共 311 GiB。
- **软件栈**：diffbot recipe，即 vLLM nightly-5555a2d8 + 6 个补丁、vllm-exl3（thor 补丁）、ExLlamaV3 5be88657、xmoe/xmoe_do 内核。不用 Docker：用 crane 导出官方镜像的 rootfs，再用 bwrap 在非特权 user namespace 里运行。
- **运行参数**：TP2；`max_model_len` 1,048,576；fp8 KV cache 4.25 GiB，约 107 万 token；DSpark k=5；冷专家方案 `plan-blend30-dh`；`NCCL_P2P_DISABLE=1`；视觉编码器加载。
- **资源占用**：
  - 显存：空闲 89.9 GB/卡，约 1M 上下文时峰值 95.4 GB/卡。
  - 主机内存：Shmem 103 GiB，serving 期间 MemAvailable 59–65 GiB。diffbot 原版约 230 GiB；torch 的锁页分配器默认把大块向上取整到 2 的幂，关掉这个取整后省下约 41 GiB。

## 速度（热态，思考关）
- prompt 用真实 Python 源码。
- prefill：不命中缓存的请求，设 `max_tokens=1`。
- decode：在已缓存的前缀上流式生成 256 token，按每路的 token 时间戳计算。

| prompt | 并发 | prefill tok/s | 首 token（无缓存） | decode 每路 | decode 合计 |
|---|---:|---:|---:|---:|---:|
| 1.8K | 1 | 1,990 | 0.9 s | **92.8** | 92.8 |
| 1.8K | 2 | 2,043 | 1.6 s | 60.9 | 121.8 |
| 1.7K | 4 | 2,816 | 2.7 s | 33.1 | 132.5 |
| 26K | 1 | 5,179 | 5.1 s | **88.9** | 88.9 |
| 26K | 2 | 5,196 | 10.2 s | 58.6 | 117.1 |
| 26K | 4 | 5,254 | 20.8 s | 40.5 | 162.1 |
| 109K | 1 | 5,085 | 21.5 s | **93.3** | 93.3 |
| 109K | 2 | 5,223 | 44.2 s | 70.4 | 140.8 |
| 101K | 4 | 5,360 | 81.0 s | 40.3 | 161.2 |
| 430K | 1 | 5,085 | 84.6 s | **99.5** | 99.5 |
| 890K | 1 | 4,866 | 182.9 s | **111.6** | 111.6 |

- 思考开启时单路 decode：1.6K 为 107.9 tok/s，25K 为 89.7 tok/s。
- 前缀命中缓存后，首 token 0.4–1.9 s。
- DSpark 每步产出 token 数：真实问答 3.97（接受率 59%），合成的“解释代码”生成 3.27。

## 大海捞针
在源码中插入口令，深度分别为 10%、50%、90%，长度从 24K 到 1,006K。**15/15 全部找到**；约 1M 的一次耗时约 205 s。

## 质量
- 固定种子 1234 抽样，temperature 0，自动判分，并发 8。
- `max_tokens`：思考关 2,048，思考开 16,384。

| 测试 | n | 思考关 | 思考开 |
|---|---:|---:|---:|
| GSM8K | 100 | **99%** | 98% |
| MATH-500 | 50 | 88% | **96%** |
| HumanEval | 50 | **98%** | 90%（被截断的题单路重跑后 49/50） |
| MMLU-Pro | 200 | **75.0%** | 71.9%（只测了 64 题） |
| C-Eval（val） | 100 | **93%** | 92% |
| 工具调用 | 10 | **10/10** | — |

- 思考开的失分大多是推理超过 16K token 被截断：HumanEval 5/50、MMLU-Pro 9/64、GSM8K 1/100。单路重跑时大多能正常结束。
- diffbot 模型卡（legacy、校准版、同一套栈）报 GSM8K-200 98.0%、HumanEval 95.7%。本次未校准的 1017 包看不出量化损失。

## 已知问题
1. **prefill 约 5K tok/s，是 diffbot 卡上的一半。** 本机开启 P2P 时，NCCL 第一次 all-reduce 就挂死，只能关掉 P2P，让 all-reduce 走主机共享内存。推测这是 prefill 慢的主因（未实测）；根因可能是 IOMMU（内核没有加 `iommu=pt`）。
2. **从 HDD 冷启动约 85 分钟**：读盘 23 分钟；内存不够导致页缓存被挤掉、反复读盘 35 分钟；DSpark drafter 又读一遍整个 checkpoint 约 23 分钟。只含 drafter 的目录（`DRAFT_PACK`）应该能省掉最后一段，但还没验证。
3. **思考模式偶尔推理过长**。官方建议 `max_tokens ≥ 256K`。
4. **冷专家方案按 legacy 模型挑选**。用 1017 的路由统计重新生成 plan，应该能进一步提高 decode 和多路并发的速度。
