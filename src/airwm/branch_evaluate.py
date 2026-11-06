"""Evaluate causal action responses using matched simulator branches."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from airwm.baselines import integrate_relative_pose, nominal_rollout
from airwm.branching import perturb_actions
from airwm.config import FlightConfig
from airwm.data import motion_targets
from airwm.dynamics import quat_multiply, quaternion_rotvec
from airwm.evaluate import save_preview
from airwm.model import ModelConfig, WorldModel


def read_flight(path):
    with h5py.File(path) as f:
        if not f.attrs.get("complete", False) or f.attrs.get("schema_version") != 1:
            raise ValueError(f"Incomplete or unsupported flight: {path}")
        result = {
            k: f[k][:]
            for k in ("rgb", "depth", "depth_valid", "imu", "state", "actions", "time", "collision")
        }
        result["metadata"] = json.loads(f.attrs["metadata"])
        result["config"] = FlightConfig(**json.loads(f.attrs["config"]))
    n, cfg = len(result["time"]), result["config"]
    if result["actions"].shape != (n - 1, cfg.controls_per_frame, 4):
        raise ValueError(f"Misaligned motor commands: {path}")
    if not np.allclose(np.diff(result["time"]), 1 / cfg.camera_hz, atol=1e-7):
        raise ValueError(f"Misaligned timestamps: {path}")
    for name, value in result.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {name}: {path}")
    if np.any((result["actions"] < 0) | (result["actions"] > 1)):
        raise ValueError(f"Out-of-range commands: {path}")
    return result


def image_tensor(flight, start, end, device):
    rgb = torch.from_numpy(flight["rgb"][start:end]).permute(0, 3, 1, 2).float() / 255
    depth = torch.from_numpy(flight["depth"][start:end]).float().unsqueeze(1)
    return torch.cat((rgb, (depth / flight["config"].depth_far).clamp(0, 1)), 1).to(device)


def relative_rotation(first, second):
    inverse = first.clone()
    inverse[..., :3] *= -1
    return quaternion_rotvec(quat_multiply(inverse, second))


def effect_metrics(actual, predicted, unit, prefix):
    """A gain of zero ignores the action; one reproduces its signed effect."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    magnitude = float(np.linalg.norm(actual))
    return {
        f"actual_{prefix}_effect_{unit}": magnitude,
        f"predicted_{prefix}_effect_{unit}": float(np.linalg.norm(predicted)),
        f"{prefix}_effect_error_{unit}": float(np.linalg.norm(predicted - actual)),
        f"{prefix}_effect_gain": float(predicted @ actual / (actual @ actual))
        if magnitude > 1e-5
        else None,
    }


def validate_replay(source, control, repeat, start):
    cfg = source["config"]
    count = control["metadata"]["requested_future_frames"]
    if len(control["time"]) != count + 1 or len(repeat["time"]) != count + 1:
        raise ValueError("Control/repeat branches must retain the full reference horizon")
    reference = source["state"][start : start + count + 1]
    if not np.array_equal(control["actions"], source["actions"][start : start + count]):
        raise ValueError("Control branch does not replay the reference commands")
    if not np.array_equal(control["actions"], repeat["actions"]):
        raise ValueError("Repeat branch uses different motor commands")
    report = {}
    for name, a, b in (
        ("reference_replay", control["state"], reference),
        ("repeat", repeat["state"], control["state"]),
    ):
        pos = float(np.linalg.norm(a[:, :3] - b[:, :3], axis=-1).max())
        orientation = float(
            relative_rotation(torch.from_numpy(a[:, 3:7]), torch.from_numpy(b[:, 3:7]))
            .norm(dim=-1)
            .max()
        )
        velocity = float(np.max(np.abs(a[:, 7:] - b[:, 7:])))
        report[name] = {
            "maximum_position_error_m": pos,
            "maximum_orientation_error_rad": orientation,
            "maximum_velocity_component_error": velocity,
        }
        if pos > 0.001 or orientation > 0.001 or velocity > 0.005:
            raise ValueError(f"State restoration failed: {name} {report[name]}")
    common = control["depth_valid"] & repeat["depth_valid"]
    if not common.any():
        raise ValueError("No valid repeat depth")
    report["repeat_rgb_mse"] = float(
        np.mean(((control["rgb"].astype(float) - repeat["rgb"].astype(float)) / 255) ** 2)
    )
    report["repeat_depth_rmse_m"] = float(
        np.sqrt(np.mean((control["depth"][common] - repeat["depth"][common]) ** 2))
    )
    if report["repeat_depth_rmse_m"] > 0.01:
        raise ValueError("Repeated matched states have inconsistent depth")
    report["camera_hz"] = cfg.camera_hz
    return report


def evaluate_branches(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WorldModel(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    paths = sorted(Path(args.data).glob("*.h5"))
    if not paths:
        raise ValueError("No complete branch flights")
    if list(Path(args.data).glob("*.partial")):
        raise ValueError("Branch collection is unfinished")
    groups = defaultdict(dict)
    for path in paths:
        flight = read_flight(path)
        meta = flight["metadata"]
        key = (meta["source"], meta["source_start_frame"])
        if meta["variant"] in groups[key]:
            raise ValueError("Duplicate source/variant")
        groups[key][meta["variant"]] = (path, flight)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    records, checks, censored, trajectories = [], [], [], []
    for source_index, ((source_path, start), variants) in enumerate(sorted(groups.items())):
        if not {"control", "repeat"}.issubset(variants):
            raise ValueError("Every history needs control and repeat branches")
        source = read_flight(source_path)
        with Path(source_path).open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        control, repeat = variants["control"][1], variants["repeat"][1]
        check = validate_replay(source, control, repeat, start)
        check["source"] = source_path
        checks.append(check)
        cfg = source["config"]
        cstart = start - args.context + 1
        if cstart < 0:
            raise ValueError("Not enough shared observation history")
        history = image_tensor(source, cstart, start + 1, device)[None]
        imu = torch.from_numpy(source["imu"][cstart : start + 1]).to(device)[None]
        past = torch.from_numpy(source["actions"][cstart:start]).flatten(1).to(device)[None]
        full_past = torch.from_numpy(source["actions"][:start])
        initial = torch.from_numpy(source["state"][start])

        def predict(flight):
            future = torch.from_numpy(flight["actions"]).flatten(1).to(device)[None]
            prediction, motion = model.predict(history, imu, past, future)
            pose = integrate_relative_pose(initial[:7], motion[0, :, 6:].cpu())
            return prediction[0].cpu(), motion[0].cpu(), pose

        base_image, _, base_pose = predict(control)
        control_pose = torch.from_numpy(control["state"][1:, :7])
        control_image = image_tensor(control, 1, None, "cpu")
        for variant, (path, flight) in variants.items():
            meta = flight["metadata"]
            if meta["source_sha256"] != source_hash or flight["config"] != cfg:
                raise ValueError("Source flight or configuration changed")
            expected_actions, clipped = perturb_actions(control["actions"], variant, cfg)
            if not np.array_equal(flight["actions"], expected_actions[: len(flight["actions"])]):
                raise ValueError("Recorded actions disagree with the declared intervention")
            if clipped != meta["clipped_command_values"] or not np.array_equal(
                meta["initial_motor_speed"], control["metadata"]["initial_motor_speed"]
            ):
                raise ValueError("Inconsistent branch motor metadata")
            initial_error = float(np.max(np.abs(flight["state"][0] - source["state"][start])))
            if initial_error > 1e-5:
                raise ValueError("Branches do not share the same initial state")
            prediction, motion, pose = predict(flight)
            truth = image_tensor(flight, 1, None, "cpu")
            states = torch.from_numpy(flight["state"])
            target_motion = motion_targets(states)[1:]
            nominal = nominal_rollout(initial, torch.from_numpy(flight["actions"]), full_past, cfg)
            n = len(truth)
            mask = torch.from_numpy(flight["depth_valid"][1:]).bool()
            for seconds in (0.5, 1.0, 2.0):
                t = round(seconds * cfg.camera_hz) - 1
                if t >= n:
                    censored.append(
                        {
                            "source": source_path,
                            "variant": variant,
                            "horizon_seconds": seconds,
                            "available_future_frames": n,
                        }
                    )
                    continue
                valid = mask[t]
                common = valid & torch.from_numpy(control["depth_valid"][t + 1]).bool()
                if not valid.any() or not common.any():
                    raise ValueError("No valid depth for branch comparison")
                actual_effect = states[t + 1, :3] - control_pose[t, :3]
                predicted_effect = pose[t, :3] - base_pose[t, :3]
                actual_rotation = relative_rotation(control_pose[t, 3:], states[t + 1, 3:7])
                predicted_rotation = relative_rotation(base_pose[t, 3:], pose[t, 3:])
                row = {
                    "source": source_path,
                    "branch": str(path.resolve()),
                    "variant": variant,
                    "horizon_seconds": seconds,
                    "rgb_mse": float((prediction[t, :3] - truth[t, :3]).square().mean()),
                    "unchanged_command_rgb_mse": float(
                        (base_image[t, :3] - truth[t, :3]).square().mean()
                    ),
                    "persistence_rgb_mse": float(
                        (history[0, -1, :3].cpu() - truth[t, :3]).square().mean()
                    ),
                    "depth_rmse_m": float(
                        (prediction[t, 3] - truth[t, 3])[valid].square().mean().sqrt()
                        * cfg.depth_far
                    ),
                    "unchanged_command_depth_rmse_m": float(
                        (base_image[t, 3] - truth[t, 3])[valid].square().mean().sqrt()
                        * cfg.depth_far
                    ),
                    "position_error_m": float((pose[t, :3] - states[t + 1, :3]).norm()),
                    "unchanged_command_position_error_m": float(
                        (base_pose[t, :3] - states[t + 1, :3]).norm()
                    ),
                    "orientation_error_rad": float(
                        relative_rotation(pose[t, 3:], states[t + 1, 3:7]).norm()
                    ),
                    "nominal_position_error_m": float((nominal[t, :3] - states[t + 1, :3]).norm()),
                    "velocity_rmse_m_s": float(
                        (motion[t, :3] - target_motion[t, :3]).square().mean().sqrt()
                    ),
                    "angular_velocity_rmse_rad_s": float(
                        (motion[t, 3:6] - target_motion[t, 3:6]).square().mean().sqrt()
                    ),
                    "rgb_effect_error_mse": float(
                        (
                            (prediction[t, :3] - base_image[t, :3])
                            - (truth[t, :3] - control_image[t, :3])
                        )
                        .square()
                        .mean()
                    ),
                    "actual_rgb_effect_mse": float(
                        (truth[t, :3] - control_image[t, :3]).square().mean()
                    ),
                    "depth_effect_error_rmse_m": float(
                        (
                            (prediction[t, 3] - base_image[t, 3])
                            - (truth[t, 3] - control_image[t, 3])
                        )[common]
                        .square()
                        .mean()
                        .sqrt()
                        * cfg.depth_far
                    ),
                    "actual_depth_effect_rmse_m": float(
                        (truth[t, 3] - control_image[t, 3])[common].square().mean().sqrt()
                        * cfg.depth_far
                    ),
                    "collision_before_horizon": bool(np.any(flight["collision"][: t + 2])),
                    **effect_metrics(
                        actual_effect.numpy(), predicted_effect.numpy(), "m", "position"
                    ),
                    **effect_metrics(
                        actual_rotation.numpy(), predicted_rotation.numpy(), "rad", "rotation"
                    ),
                }
                if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
                    raise FloatingPointError("Nonfinite branch metric")
                records.append(row)
            trajectories.append(
                {
                    "source": source_path,
                    "variant": variant,
                    "actual_delta_xyz": (states[1:, :3] - control_pose[:n, :3]).tolist(),
                    "predicted_delta_xyz": (pose[:, :3] - base_pose[:n, :3]).tolist(),
                    "actual_delta_rotation": relative_rotation(
                        control_pose[:n, 3:], states[1:, 3:7]
                    ).tolist(),
                    "predicted_delta_rotation": relative_rotation(
                        base_pose[:n, 3:], pose[:, 3:]
                    ).tolist(),
                }
            )
            if source_index == 0:
                save_preview(
                    out / f"{variant}.png",
                    truth.numpy(),
                    prediction.numpy(),
                    history[0, -1:].cpu().expand_as(truth).numpy(),
                    cfg.camera_hz,
                )
                np.savez_compressed(
                    out / f"{variant}.npz",
                    truth=truth.numpy(),
                    prediction=prediction.numpy(),
                    unchanged_command_prediction=base_image[:n].numpy(),
                    actual_state=states.numpy(),
                    predicted_pose=pose.numpy(),
                )
        print(f"Evaluated {source_index + 1}/{len(groups)} common histories", flush=True)
    summaries = {}
    for variant in sorted({r["variant"] for r in records}):
        summaries[variant] = {}
        for seconds in (0.5, 1.0, 2.0):
            rows = [
                r for r in records if r["variant"] == variant and r["horizon_seconds"] == seconds
            ]
            if not rows:
                continue
            keys = {
                k
                for r in rows
                for k, v in r.items()
                if isinstance(v, float) and k != "horizon_seconds"
            }
            means = {
                k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in sorted(keys)
            }
            summaries[variant][str(seconds)] = {"histories": len(rows), "mean": means}
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint["step"],
        "context_frames": args.context,
        "common_histories": len(groups),
        "branch_files": len(paths),
        "source_overlap_with_checkpoint_splits": sorted(
            set(k[0] for k in groups)
            & {p for files in checkpoint.get("split", {}).values() for p in files}
        ),
        "protocol": "Restore body pose/velocity and reconstructed rotor lag at the same source frame; apply a 0.25-second motor-speed pulse; then replay identical recorded commands without controller feedback. All predictions use the identical source RGB-D/IMU/action history. State is excluded from learned-model inputs and is used only for evaluation anchoring and the privileged nominal-physics baseline.",
        "interpretation": "Compare matched-command predictions against the same model given unchanged commands. Paired effect gain 0 means ignoring the intervention; gain 1 means correct signed magnitude. RGB repeat differences measure rendering variation; depth and state replay checks establish physical matching. Shared histories are the unit of comparison, not independent branch samples.",
        "replay_checks": checks,
        "censored_horizons": censored,
        "summary_by_variant_and_horizon": summaries,
        "metrics": records,
        "trajectories": trajectories,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved branch evaluation to {out}", flush=True)
