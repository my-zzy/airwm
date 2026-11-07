"""Open-loop prediction on held-out flights, with a last-frame baseline."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

from airwm.baselines import integrate_relative_pose, nominal_rollout
from airwm.config import FlightConfig
from airwm.data import FlightSequences, motion_targets
from airwm.dynamics import quat_multiply, quaternion_rotvec
from airwm.model import ModelConfig, WorldModel
from airwm.train import to_device


def evaluation_indices(windows, count):
    """Round-robin flights with evenly spaced, nonoverlapping windows in each.

    Allocate the window budget across flights before choosing their offsets so
    even a small evaluation covers different flights and different flight times.
    """
    groups = {}
    for index, (episode, _) in enumerate(windows):
        groups.setdefault(episode, []).append(index)
    budgets = dict.fromkeys(groups, 0)
    remaining = min(count, len(windows))
    while remaining:
        for episode, indices in groups.items():
            if budgets[episode] < len(indices) and remaining:
                budgets[episode] += 1
                remaining -= 1
    selected = []
    for episode, indices in groups.items():
        n = budgets[episode]
        # Midpoints of equally sized bins include late flight times even at n=1.
        selected.append([indices[int((j + 0.5) * len(indices) / n)] for j in range(n)])
    return [
        row[j] for j in range(max(budgets.values(), default=0)) for row in selected if j < len(row)
    ]


def summarize_metrics(metrics):
    summaries = {}
    for horizon in sorted({m["horizon_seconds"] for m in metrics}):
        records = [m for m in metrics if m["horizon_seconds"] == horizon]
        numeric = [
            k
            for k, v in records[0].items()
            if isinstance(v, float) and k not in ("horizon_seconds", "start_time_seconds")
        ]
        summaries[str(horizon)] = {
            "windows": len(records),
            "flights": len({m["flight"] for m in records}),
            "mean": {k: float(np.mean([m[k] for m in records])) for k in numeric},
            "std": {k: float(np.std([m[k] for m in records])) for k in numeric},
        }
    return summaries


def evaluation_paths(checkpoint, split, split_file=None):
    """Allow comparisons on identical flights without admitting training data."""
    splits = json.loads(Path(split_file).read_text()) if split_file else checkpoint["split"]
    paths = splits[split]
    resolved = [str(Path(p).resolve()) for p in paths]
    training = {str(Path(p).resolve()) for p in checkpoint["split"]["train"]}
    if not paths or len(set(resolved)) != len(paths):
        raise ValueError("Evaluation flights must be nonempty and unique")
    if training.intersection(resolved):
        raise ValueError("Evaluation overlaps the checkpoint's training flights")
    return paths


def save_preview(path, truth, predicted, persistence, camera_hz):
    # RGB rows followed by metric-depth grayscale rows; columns span the rollout.
    indices = np.linspace(0, len(truth) - 1, min(8, len(truth)), dtype=int)
    width = 120 + 64 * len(indices)
    canvas = Image.new("RGB", (width, 6 * 84), "white")
    draw = ImageDraw.Draw(canvas)
    labels = [
        "RGB actual",
        "RGB predicted",
        "RGB persistence",
        "Depth actual",
        "Depth predicted",
        "Depth persistence",
    ]
    arrays = [truth, predicted, persistence] * 2
    for row, (label, frames) in enumerate(zip(labels, arrays)):
        draw.text((4, row * 84 + 28), label, fill="black")
        for col, index in enumerate(indices):
            pixels = (
                frames[index, :3].transpose(1, 2, 0)
                if row < 3
                else np.repeat(frames[index, 3:4].transpose(1, 2, 0), 3, axis=-1)
            )
            pixels = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(pixels), (120 + col * 64, row * 84 + 16))
            draw.text((120 + col * 64, row * 84), f"{(index + 1) / camera_hz:.2f}s", fill="black")
    canvas.save(path)


def evaluate(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WorldModel(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    split_file = getattr(args, "split_file", None)
    paths = evaluation_paths(checkpoint, args.split, split_file)
    dataset = FlightSequences(
        paths, args.context + args.horizon, stride=args.context + args.horizon
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    metrics = []
    selected = evaluation_indices(dataset.windows, args.windows)
    for i, dataset_index in enumerate(selected):
        batch = to_device({k: v[None] for k, v in dataset[dataset_index].items()}, device)
        c = args.context
        prediction, motion = model.predict(
            batch["image"][:, :c],
            batch["imu"][:, :c],
            batch["actions"][:, : c - 1],
            batch["actions"][:, c - 1 :],
        )
        truth = batch["image"][:, c:]
        persist = batch["image"][:, c - 1 : c].expand_as(truth)
        mask = batch["depth_valid"][:, c:].float()
        # Counterfactual motors test action sensitivity, not counterfactual correctness.
        changed = (batch["actions"][:, c - 1 :] + 0.15).clamp(0, 1)
        alternate, _ = model.predict(
            batch["image"][:, :c],
            batch["imu"][:, :c],
            batch["actions"][:, : c - 1],
            changed,
        )
        # Read privileged reference state separately. Neither prediction above
        # receives this state; it anchors evaluation and initializes the baseline.
        episode, start = dataset.windows[dataset_index]
        with h5py.File(dataset.paths[episode]) as f:
            cfg = FlightConfig(**json.loads(f.attrs["config"]))
            states = torch.from_numpy(f["state"][start : start + c + args.horizon]).float()
            past_actions = torch.from_numpy(f["actions"][: start + c - 1]).float()
            collisions = f["collision"][start + c : start + c + args.horizon]
        far, hz = cfg.depth_far, cfg.camera_hz
        future_actions = batch["actions"][0, c - 1 :].cpu().reshape(-1, cfg.controls_per_frame, 4)
        nominal = nominal_rollout(states[c - 1], future_actions, past_actions, cfg)
        nominal_motion = motion_targets(torch.cat((states[c - 1 : c], nominal)))[1:]
        reference_motion = motion_targets(states[c - 1 :])[1:]
        learned_poses = integrate_relative_pose(states[c - 1, :7], motion[0, :, 6:].cpu())

        def orientation_error(predicted_q, target_q):
            inverse = predicted_q.clone()
            inverse[:3] *= -1
            return float(quaternion_rotvec(quat_multiply(inverse, target_q)).norm())

        for seconds in (0.5, 1.0, 2.0):
            t = round(seconds * hz) - 1
            if t >= args.horizon:
                continue
            valid = mask[:, t]
            record = {
                "window": i,
                "dataset_index": dataset_index,
                "flight": dataset.paths[episode],
                "start_frame": start,
                "start_time_seconds": start / hz,
                "horizon_seconds": seconds,
                "rgb_mse": float((prediction[:, t, :3] - truth[:, t, :3]).square().mean()),
                "persistence_rgb_mse": float((persist[:, t, :3] - truth[:, t, :3]).square().mean()),
                "depth_rmse_m": float(
                    (((prediction[:, t, 3:] - truth[:, t, 3:]) * far).square() * valid)
                    .sum()
                    .div(valid.sum().clamp_min(1))
                    .sqrt()
                ),
                "persistence_depth_rmse_m": float(
                    (((persist[:, t, 3:] - truth[:, t, 3:]) * far).square() * valid)
                    .sum()
                    .div(valid.sum().clamp_min(1))
                    .sqrt()
                ),
                "velocity_rmse_m_s": float(
                    (motion[:, t, :3] - batch["motion"][:, c + t, :3]).square().mean().sqrt()
                ),
                "angular_velocity_rmse_rad_s": float(
                    (motion[:, t, 3:6] - batch["motion"][:, c + t, 3:6]).square().mean().sqrt()
                ),
                "relative_translation_rmse_m": float(
                    (motion[:, t, 6:9] - batch["motion"][:, c + t, 6:9]).square().mean().sqrt()
                ),
                "relative_rotation_rmse_rad": float(
                    (motion[:, t, 9:] - batch["motion"][:, c + t, 9:]).square().mean().sqrt()
                ),
                "action_change_rgb_mse": float(
                    (alternate[:, t, :3] - prediction[:, t, :3]).square().mean()
                ),
                "position_error_m": float((learned_poses[t, :3] - states[c + t, :3]).norm()),
                "orientation_error_rad": orientation_error(
                    learned_poses[t, 3:7], states[c + t, 3:7]
                ),
                "nominal_position_error_m": float((nominal[t, :3] - states[c + t, :3]).norm()),
                "nominal_orientation_error_rad": orientation_error(
                    nominal[t, 3:7], states[c + t, 3:7]
                ),
                "nominal_velocity_rmse_m_s": float(
                    (nominal_motion[t, :3] - reference_motion[t, :3]).square().mean().sqrt()
                ),
                "nominal_angular_velocity_rmse_rad_s": float(
                    (nominal_motion[t, 3:6] - reference_motion[t, 3:6]).square().mean().sqrt()
                ),
                "collision_before_horizon": bool(np.any(collisions[: t + 1])),
            }
            metrics.append(record)
        if i == 0:
            save_preview(
                out / "prediction.png",
                truth[0].cpu().numpy(),
                prediction[0].cpu().numpy(),
                persist[0].cpu().numpy(),
                hz,
            )
            np.savez_compressed(
                out / "prediction.npz",
                truth=truth[0].cpu().numpy(),
                prediction=prediction[0].cpu().numpy(),
                motion=motion[0].cpu().numpy(),
                predicted_pose=learned_poses.numpy(),
                nominal_state=nominal.numpy(),
                reference_state=states[c:].numpy(),
            )
    report = {
        "split": args.split,
        "split_file": str(Path(split_file).resolve()) if split_file else None,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "context_frames": args.context,
        "future_frames": args.horizon,
        "windows": len(selected),
        "flights_evaluated": len({dataset.windows[i][0] for i in selected}),
        "window_sampling": "balanced across flights, evenly spaced within each flight",
        "note": "Learned pose deltas are integrated from the last observed ground-truth pose for evaluation only. The nominal baseline receives the last ground-truth state, recorded past/future commands, and nominal physical parameters. It omits contacts and PhysX solver details. Ground-truth state is never an RSSM input.",
        "metrics": metrics,
        "summary_by_horizon": summarize_metrics(metrics),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved evaluation to {out}", flush=True)
