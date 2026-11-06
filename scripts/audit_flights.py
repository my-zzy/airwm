"""Audit complete simulator flights before training; no GPU or simulator needed."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
        "motor_command_min": action.min(0).tolist(),
        "motor_command_max": action.max(0).tolist(),
        "motor_command_std": action.std(0).tolist(),
        "yaw_15degree_bin_counts": yaw_histogram.tolist(),
        "visited_xy_0_25m_cells": len(cells_xy),
        "room_xy_0_25m_cells": 24 * 24,
        "visited_xyz_0_25m_cells": len(cells_xyz),
        "coverage_note": "Pose occupancy measures where the drone flew, not what surfaces were seen. Central scripted sweeps do not establish complete room reconstruction or broad dynamics coverage.",
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
