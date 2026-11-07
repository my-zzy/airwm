"""Audit complete simulator flights before training; no GPU or simulator needed."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

from airwm.config import FlightConfig
from airwm.dynamics import Motors, controller
from airwm.maneuvers import PATTERNS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def audit_pulses(metadata, cfg, states, actions):
    """Recompute pulse onset from recorded state and verify the held command."""
    if metadata.get("maneuver_protocol") != "waypoints_axis_pulses_v1":
        return 0
    motors = Motors(FlightConfig(**cfg))
    flat = actions.reshape(-1, 4)
    checked = 0
    for event in metadata["pulse_events"]:
        count = event["applied_controls"]
        if not count:
            continue
        start, end = event["start_control"], event["end_control"]
        assert 0 < count <= end - start and start + count <= len(flat)
        if not event["cancelled"]:
            assert count == end - start
        per_frame = cfg["control_hz"] // cfg["camera_hz"]
        assert start % per_frame == 0, "Pulse onset must have a recorded camera state"
        state = torch.from_numpy(states[start // per_frame])[None]
        index = start // round(metadata["waypoint_period_seconds"] * cfg["control_hz"])
        target = torch.tensor([metadata["waypoints_xyz_m"][index]], dtype=torch.float32)
        yaw = torch.tensor([metadata["waypoint_yaw_rad"][index]], dtype=torch.float32)
        base = controller(
            motors, state[:, :3], state[:, 3:7], state[:, 7:10], state[:, 10:13], target, yaw
        )
        offset = event["amplitude"] * torch.tensor(PATTERNS[event["axis"]])
        expected = (base[0] + offset).clamp(0, 1).numpy()
        np.testing.assert_allclose(flat[start], expected, atol=1e-5, rtol=0)
        np.testing.assert_array_equal(
            flat[start : start + count], np.broadcast_to(flat[start], (count, 4))
        )
        checked += 1
    return checked


def audit(folder, output, expected):
    files = sorted(folder.glob("*.h5"))
    if expected is not None and len(files) != expected:
        raise ValueError(f"Expected {expected} complete flights, found {len(files)}")
    if expected is not None and list(folder.glob("*.partial")):
        raise ValueError("Collection still contains unfinished flights")
    if not files:
        raise ValueError("No complete flights")
    output.mkdir(parents=True, exist_ok=False)
    records, positions, velocities, headings, commands, trajectories = [], [], [], [], [], []
    angular_velocities = []
    preview = Image.new("RGB", (8 * 128, 5 * 152), "white")
    draw = ImageDraw.Draw(preview)
    preview_flights = np.linspace(0, len(files) - 1, min(5, len(files)), dtype=int).tolist()
    for i, path in enumerate(files):
        with h5py.File(path, "r") as f:
            cfg = json.loads(f.attrs["config"])
            metadata = json.loads(f.attrs["metadata"])
            t, state, actions = f["time"][:], f["state"][:], f["actions"][:]
            n = len(t)
            assert f.attrs["complete"] and f.attrs["schema_version"] == 1, path
            assert n >= 64 and actions.shape == (n - 1, cfg["control_hz"] // cfg["camera_hz"], 4), (
                path
            )
            assert np.allclose(np.diff(t), 1 / cfg["camera_hz"], atol=1e-7), path
            assert np.allclose(np.linalg.norm(state[:, 3:7], axis=-1), 1, atol=1e-4), path
            assert actions.min() >= 0 and actions.max() <= 1, path
            pulses_checked = audit_pulses(metadata, cfg, state, actions)
            valid_pixels, total_pixels, rgb_stds, depth_min, depth_max = 0, 0, [], float("inf"), 0
            for start in range(0, n, 128):
                stop = min(n, start + 128)
                for name in ("rgb", "depth", "depth_valid", "imu", "state", "time", "collision"):
                    assert len(f[name]) == n and np.isfinite(f[name][start:stop]).all(), (
                        path,
                        name,
                    )
                depth = f["depth"][start:stop]
                valid = f["depth_valid"][start:stop].astype(bool)
                assert (
                    valid.any() and ((depth[valid] > 0) & (depth[valid] < cfg["depth_far"])).all()
                ), path
                valid_pixels += int(valid.sum())
                total_pixels += valid.size
                depth_min = min(depth_min, float(depth[valid].min()))
                depth_max = max(depth_max, float(depth[valid].max()))
                rgb_stds.extend(f["rgb"][start:stop].reshape(stop - start, -1).std(-1).tolist())
            assert np.isfinite(actions).all(), path
            collisions = int(f["collision"][:].sum())
            positions.append(state[:, :3])
            velocities.append(state[:, 7:10])
            angular_velocities.append(state[:, 10:13])
            trajectories.append(state[:, :3])
            x, y, z, w = state[:, 3:7].T
            headings.append(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            commands.append(actions.reshape(-1, 4))
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            records.append(
                {
                    "path": str(path.resolve()),
                    "observations": n,
                    "transitions": n - 1,
                    "duration_seconds": float(t[-1] - t[0]),
                    "collision_frames": collisions,
                    "termination": str(f.attrs.get("termination", "unknown")),
                    "depth_valid_fraction": valid_pixels / total_pixels,
                    "depth_min_m": depth_min,
                    "depth_max_m": depth_max,
                    "minimum_rgb_std_uint8": min(rgb_stds),
                    "phase": metadata.get("phase"),
                    "sha256": digest,
                    "trajectory": metadata.get("trajectory"),
                    "pulse_events": metadata.get("pulse_events", []),
                    "recovery_controls": metadata.get("recovery_controls", 0),
                    "clipped_command_values": metadata.get("clipped_command_values", 0),
                    "recorded_pulses_verified": pulses_checked,
                }
            )
            if i in preview_flights:
                row = preview_flights.index(i)
                for col, index in enumerate(np.linspace(0, n - 1, 8, dtype=int)):
                    frame = Image.fromarray(f["rgb"][index]).resize((128, 128))
                    preview.paste(frame, (col * 128, row * 152 + 24))
                    draw.text(
                        (col * 128 + 3, row * 152 + 4), f"Flight {i}: {t[index]:.1f}s", fill="black"
                    )
    position, velocity, heading, action = map(
        np.concatenate, (positions, velocities, headings, commands)
    )
    cells_xy = np.unique(np.floor((position[:, :2] + 3) / 0.25).astype(int), axis=0)
    cells_xyz = np.unique(np.floor((position + [3, 3, 0]) / 0.25).astype(int), axis=0)
    yaw_histogram, _ = np.histogram(heading, np.linspace(-np.pi, np.pi, 25))
    # Orthonormal axis patterns expose differential excitation that per-motor
    # marginal ranges alone cannot establish. Offsets remain speed units, not N.
    axis_commands = action @ np.array(list(PATTERNS.values()), dtype=np.float64).T / 4
    axis_covariance = np.cov(axis_commands, rowvar=False)
    events = [e for r in records for e in r["pulse_events"]]
    report = {
        "complete_flights": len(files),
        "partial_files": [str(p) for p in folder.glob("*.partial")],
        "observations": sum(r["observations"] for r in records),
        "transitions": sum(r["transitions"] for r in records),
        "simulated_seconds": sum(r["duration_seconds"] for r in records),
        "collision_frames": sum(r["collision_frames"] for r in records),
        "size_GiB": sum(p.stat().st_size for p in files) / 1024**3,
        "position_min_xyz_m": position.min(0).tolist(),
        "position_max_xyz_m": position.max(0).tolist(),
        "speed_percentiles_m_s": dict(
            zip(
                ("p0", "p50", "p95", "p100"),
                np.percentile(np.linalg.norm(velocity, axis=-1), [0, 50, 95, 100]).tolist(),
            )
        ),
        "angular_speed_percentiles_rad_s": dict(
            zip(
                ("p0", "p50", "p95", "p100"),
                np.percentile(
                    np.linalg.norm(np.concatenate(angular_velocities), axis=-1), [0, 50, 95, 100]
                ).tolist(),
            )
        ),
        "motor_command_min": action.min(0).tolist(),
        "motor_command_max": action.max(0).tolist(),
        "motor_command_std": action.std(0).tolist(),
        "motor_command_at_bound_fraction": ((action == 0) | (action == 1)).mean(0).tolist(),
        "axis_command_std": dict(zip(PATTERNS, axis_commands.std(0).tolist())),
        "axis_command_covariance": axis_covariance.tolist(),
        "pulse_counts_by_axis": {axis: sum(e["axis"] == axis for e in events) for axis in PATTERNS},
        "pulse_controls_applied": sum(e["applied_controls"] for e in events),
        "recorded_pulses_verified": sum(r["recorded_pulses_verified"] for r in records),
        "cancelled_pulses": sum(e["cancelled"] for e in events),
        "recovery_controls": sum(r["recovery_controls"] for r in records),
        "clipped_command_values": sum(r["clipped_command_values"] for r in records),
        "clipping_note": "The maneuver policy's clipped_command_values counts clipping after adding a pulse offset. Feedback mixing can also reach actuator bounds; motor_command_at_bound_fraction measures the actual recorded commands at 0 or 1.",
        "yaw_15degree_bin_counts": yaw_histogram.tolist(),
        "visited_xy_0_25m_cells": len(cells_xy),
        "room_xy_0_25m_cells": 24 * 24,
        "visited_xyz_0_25m_cells": len(cells_xyz),
        "coverage_note": "Pose occupancy measures where the drone flew, not what surfaces were seen. These paths do not establish complete room reconstruction. Axis-command statistics describe collected excitation, not learned action-response accuracy.",
        "flights": records,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    preview.save(output / "flight-preview.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for trajectory in trajectories:
        axes[0].plot(trajectory[:, 0], trajectory[:, 1], alpha=0.2, lw=0.5)
    axes[0].set(
        xlim=(-3, 3),
        ylim=(-3, 3),
        xlabel="x (m)",
        ylabel="y (m)",
        title="Drone paths in 6 x 6 m room",
        aspect="equal",
    )
    axes[1].hist(position[:, 2], bins=30)
    axes[1].set(xlabel="Altitude (m)", ylabel="Observations", title="Altitude distribution")
    axes[2].hist(np.degrees(heading), bins=np.linspace(-180, 180, 25))
    axes[2].set(xlabel="Yaw (degrees)", ylabel="Observations", title="Heading distribution")
    fig.tight_layout()
    fig.savefig(output / "coverage.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "flights"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-flights", type=int)
    args = parser.parse_args()
    audit(args.data, args.output, args.expected_flights)
