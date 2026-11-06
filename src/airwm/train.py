"""Offline training. No simulator process is imported or started here."""

import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from airwm.data import FlightSequences, split_flights
from airwm.model import ModelConfig, WorldModel


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train(args):
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("Training steps and batch size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run outside a GPU-restricted sandbox or choose --device cpu"
        )
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    split = split_flights(args.data, args.seed)
    (run / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    dataset = FlightSequences(split["train"], args.length)
    validation = FlightSequences(split["validation"], args.length, stride=args.length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    config = (
        ModelConfig(deter=64, hidden=64, embed=64, stoch=4, classes=4, cnn=4)
        if args.tiny
        else ModelConfig()
    )
    model = WorldModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
    mixed = device.type == "cuda" and torch.cuda.is_bf16_supported() and not args.no_amp

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if mixed else nullcontext()

    settings = {
        **vars(args),
        "model": config.to_dict(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "amp_bfloat16": mixed,
        "torch": torch.__version__,
    }
    settings.pop("func", None)
    (run / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    started = time.perf_counter()
    best = float("inf")
    with (run / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = to_device(batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss, metrics = model.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100, error_if_nonfinite=True)
            optimizer.step()
            record = {
                "step": step,
                **{k: float(v) for k, v in metrics.items()},
                "grad_norm": float(norm),
            }
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                values = []
                with torch.no_grad(), autocast():
                    for i, vbatch in enumerate(val_loader):
                        vloss, _ = model.loss(to_device(vbatch, device))
                        values.append(float(vloss))
                        if i >= 7:
                            break
                record["validation_loss"] = float(np.mean(values))
                checkpoint = {
                    "model": model.state_dict(),
                    "model_config": config.to_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "split": split,
                    "settings": settings,
                }
                torch.save(checkpoint, run / "last.pt")
                if record["validation_loss"] < best:
                    best = record["validation_loss"]
                    torch.save(checkpoint, run / "best.pt")
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(json.dumps(record), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {
        "steps": args.steps,
        "parameters": settings["parameters"],
        "elapsed_seconds": time.perf_counter() - started,
        "best_validation_loss": best,
        "batch_size_requested": args.batch_size,
        "max_batch_size_used": min(args.batch_size, len(dataset)),
        "sequence_length": args.length,
        "peak_tensor_GiB": torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else None,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved(device) / 1024**3
        if device.type == "cuda"
        else None,
    }
    (run / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
