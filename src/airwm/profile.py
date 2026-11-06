"""Measure training memory using synthetic tensors, without loading the simulator."""

import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from airwm.model import WorldModel


def profile(args):
    if min(args.batch_size, args.length, args.steps) < 1 or args.length < 2:
        raise ValueError("Positive batch/step counts and sequence length >=2 are required")
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = WorldModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    b, t = args.batch_size, args.length
    batch = {
        "image": torch.rand(b, t, 4, 64, 64, device=device),
        "imu": torch.randn(b, t, 6, device=device),
        "actions": torch.rand(b, t - 1, 20, device=device),
        "depth_valid": torch.ones(b, t, 1, 64, 64, device=device, dtype=torch.bool),
        "motion": torch.zeros(b, t, 12, device=device),
    }
    mixed = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for step in range(args.steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16) if mixed else nullcontext():
            loss, _ = model.loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
        print(f"Profile step {step + 1}/{args.steps}: {times[-1]:.3f}s", flush=True)
    report = {
        "synthetic_benchmark": True,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "batch_size": b,
        "sequence_length": t,
        "bf16": mixed,
        "step_seconds": times,
        "peak_tensor_GiB": torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else None,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved(device) / 1024**3
        if device.type == "cuda"
        else None,
        "scope": "Training only; includes forward, backward and Adam state; excludes Isaac Sim.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
