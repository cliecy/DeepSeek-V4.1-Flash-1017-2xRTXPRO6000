#!/usr/bin/env python3
"""2-GPU NCCL all-reduce smoke test (torchrun-free): python3 nccl_test.py  -> prints per-rank result and timing."""
import os, time, torch, torch.distributed as dist, torch.multiprocessing as mp


def run(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=os.environ.get("PORT_NCCL", "29533"))
    torch.cuda.set_device(rank)
    t = time.time()
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    x = torch.full((1 << 20,), float(rank + 1), device="cuda")
    dist.all_reduce(x); torch.cuda.synchronize()
    print(f"rank {rank}: all_reduce ok value={x[0].item()} in {time.time() - t:.1f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, nprocs=2)
