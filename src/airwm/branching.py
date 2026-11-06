"""Matched action interventions; preparation does not import the simulator."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from airwm.config import FlightConfig
from airwm.dynamics import Motors

# Offsets are normalized desired rotor speed, applied for 0.25 s. Following
# commands are the identical recorded open-loop sequence, without controller
# adaptation. Motor order: front-left, rear-left, rear-right, front-right.
INTERVENTIONS = {
    "control": (0.0, [1, 1, 1, 1]),
    "repeat": (0.0, [1, 1, 1, 1]),
    "collective_up_small": (0.005, [1, 1, 1, 1]),
    "collective_down_small": (-0.005, [1, 1, 1, 1]),
    "collective_up": (0.02, [1, 1, 1, 1]),
    "collective_down": (-0.02, [1, 1, 1, 1]),
    "yaw_positive": (0.005, [1, -1, 1, -1]),
    "yaw_negative": (-0.005, [1, -1, 1, -1]),
    "pitch_positive": (0.0005, [-1, 1, 1, -1]),
    "pitch_negative": (-0.0005, [-1, 1, 1, -1]),
    "roll_positive": (0.0005, [1, 1, -1, -1]),
    "roll_negative": (-0.0005, [1, 1, -1, -1]),
}


def perturb_actions(actions, variant, cfg=FlightConfig(), pulse_seconds=0.25):
    result = np.asarray(actions, dtype=np.float32).copy()
    amplitude, pattern = INTERVENTIONS[variant]
    count = round(pulse_seconds * cfg.control_hz)
    flat = result.reshape(-1, 4)
    flat[:count] += amplitude * np.asarray(pattern, dtype=np.float32)
    clipped = int(np.count_nonzero((flat < 0) | (flat > 1)))
    return np.clip(result, 0, 1), clipped


def reconstruct_motors(actions, cfg=FlightConfig()):
    """Restore lag state by replaying the exact physics-rate motor updates."""
    motors = Motors(cfg)
    motors.reset()
    for command in torch.as_tensor(actions).reshape(-1, 4):
        for _ in range(cfg.physics_per_control):
            motors.step(command[None], 1 / cfg.physics_hz)
    return motors.speed[0].numpy().copy()


def prepare_branches(args):
    sources = sorted(Path(args.source).glob("*.h5"))[: args.sources]
    if len(sources) != args.sources:
        raise ValueError(f"Expected {args.sources} complete source flights, found {len(sources)}")
    plans = []
    for source in sources:
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with h5py.File(source) as f:
            cfg = FlightConfig(**json.loads(f.attrs["config"]))
            if cfg != FlightConfig() or not f.attrs["complete"]:
                raise ValueError(f"Unsupported/incomplete source flight: {source}")
            metadata = json.loads(f.attrs["metadata"])
            if metadata.get("room") != "procedural_6x6x3_v1":
                raise ValueError("Branching requires the same procedural room")
            start, end = args.start_frame, args.start_frame + args.horizon
            if start < 19 or end >= len(f["time"]) or np.any(f["collision"][: end + 1]):
                raise ValueError(
                    "Need at least 20 context frames and a complete collision-free reference"
                )
            state, imu = f["state"][start], f["imu"][start]
            reference_states = f["state"][start : end + 1]
            baseline = f["actions"][start:end]
            speed = reconstruct_motors(f["actions"][:start], cfg)
        for variant in args.branches:
            actions, clipped = perturb_actions(baseline, variant, cfg)
            plans.append(
                {
                    "initial_state": state,
                    "initial_imu": imu,
                    "motor_speed": speed,
                    "actions": actions,
                    "reference_states": reference_states,
                    "filename": f"{source.stem}__{variant}.h5",
                    "metadata": {
                        "source": str(source.resolve()),
                        "source_sha256": digest,
                        "source_start_frame": start,
                        "source_start_seconds": start / cfg.camera_hz,
                        "source_phase": metadata["phase"],
                        "variant": variant,
                        "pulse_seconds": 0.25,
                        "amplitude": INTERVENTIONS[variant][0],
                        "motor_pattern": INTERVENTIONS[variant][1],
                        "initial_motor_speed": speed.tolist(),
                        "motor_state_restoration": "replay all source commands from hover at physics rate",
                        "clipped_command_values": clipped,
                        "requested_future_frames": args.horizon,
                        "evaluation_only": True,
                    },
                }
            )
    return plans
