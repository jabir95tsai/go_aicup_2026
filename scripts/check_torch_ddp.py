"""Check single-node CUDA/NCCL communication and DDP gradient averaging.

Run inside a Linux GPU allocation with:
python -m torch.distributed.run --standalone --nnodes=1 \
    --nproc-per-node=2 scripts/check_torch_ddp.py
"""

from __future__ import annotations

from datetime import timedelta
import json
import os
import platform


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("This NCCL check requires a Linux GPU allocation.")

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError("Install the Linux CUDA PyTorch build with NCCL support.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the GPU allocation and driver.")

    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    if any(key not in os.environ for key in required):
        raise RuntimeError("Launch this check with torchrun / torch.distributed.run.")
    rank, local_rank, world_size, local_world_size = (
        int(os.environ[key]) for key in required
    )
    if not (1 <= world_size <= 8 and world_size == local_world_size and rank == local_rank):
        raise RuntimeError("Only one node with 1-8 GPU processes is supported.")
    if torch.cuda.device_count() < local_world_size:
        raise RuntimeError("The process count exceeds the visible allocated GPUs.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))
    try:
        value = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size + 1) / 2
        torch.testing.assert_close(value, torch.full_like(value, expected_sum))

        # Different inputs on each rank verify that DDP averages the gradients.
        model = DistributedDataParallel(
            torch.nn.Linear(4, 2, bias=False).to(device), device_ids=[local_rank]
        )
        model(torch.full((1, 4), rank + 1.0, device=device)).sum().backward()
        gradient = model.module.weight.grad
        if gradient is None:
            raise RuntimeError("DDP did not produce a gradient.")
        torch.testing.assert_close(
            gradient, torch.full_like(gradient, (world_size + 1) / 2)
        )
        dist.barrier()
        if rank == 0:
            print(json.dumps({
                "status": "passed",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "libc": platform.libc_ver(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "world_size": world_size,
                "gpus": [torch.cuda.get_device_name(i) for i in range(local_world_size)],
                "all_reduce_sum": value.item(),
                "ddp_gradient": gradient[0, 0].item(),
            }), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
