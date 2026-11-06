"""Flight files and sequence sampling; simulator state is supervision only.

For N observations there are N-1 action blocks. actions[t, k] is the motor
command applied during control interval k between observations t and t+1.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from airwm.config import FlightConfig
from airwm.dynamics import quat_multiply, quaternion_rotvec, rotation_matrix


class EpisodeWriter:
    def __init__(self, path, cfg: FlightConfig, metadata=None):
        self.path = Path(path)
        self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.partial, "x")
        self.file.attrs.update(
            schema_version=1,
            config=json.dumps(cfg.to_dict()),
            metadata=json.dumps(metadata or {}),
            quaternion_order="xyzw",
            action_alignment="actions[t] drives observations[t] -> observations[t+1]",
            imu_convention="body specific acceleration (m/s^2), body angular velocity (rad/s)",
            state_layout="world position xyz, body-to-world quaternion xyzw, world velocity xyz, world angular velocity xyz",
        )
        self.cfg = cfg
        self.count = 0
        self.last_time = None

    def _append(self, name, value):
        value = np.asarray(value)
        if name not in self.file:
            self.file.create_dataset(
                name,
                shape=(0, *value.shape),
                maxshape=(None, *value.shape),
                dtype=value.dtype,
                chunks=(1, *value.shape),
                compression="lzf",
            )
        ds = self.file[name]
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = value

    def append(self, observation, actions=None):
        expected = {
            "rgb": (self.cfg.image_size, self.cfg.image_size, 3),
            "depth": (self.cfg.image_size, self.cfg.image_size),
            "depth_valid": (self.cfg.image_size, self.cfg.image_size),
            "imu": (6,),
            "state": (13,),
            "time": (),
            "collision": (),
        }
        for name, shape in expected.items():
            value = np.asarray(observation[name])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name}: expected finite {shape}, got {value.shape}")
        now = float(observation["time"])
        if self.count:
            if actions is None or np.shape(actions) != (self.cfg.controls_per_frame, 4):
                raise ValueError("Every transition must retain all motor commands")
            if not np.isfinite(actions).all() or np.min(actions) < 0 or np.max(actions) > 1:
                raise ValueError("Motor commands must be finite and normalized to [0,1]")
            if not np.isclose(now - self.last_time, 1 / self.cfg.camera_hz, atol=1e-7):
                raise ValueError("Observation timestamps are not aligned to the camera cadence")
            self._append("actions", np.asarray(actions, dtype=np.float32))
        elif actions is not None:
            raise ValueError("First observation has no incoming transition")
        for name in expected:
            self._append(name, observation[name])
        self.last_time = now
        self.count += 1

    def close(self, complete=True):
        if not self.file.id.valid:
            return
        self.file.attrs["complete"] = bool(complete and self.count >= 2)
        self.file.flush()
        self.file.close()
        if complete and self.count >= 2:
            self.partial.rename(self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.close(complete=exc_type is None)


def split_flights(folder, seed=0):
    """Split whole flights before sampling windows; validation/test are disjoint."""
    files = sorted(str(p.resolve()) for p in Path(folder).glob("*.h5"))
    if len(files) < 3:
        raise ValueError(
            "Collect at least three complete flights for train/validation/test splitting"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    held_out = max(1, int(len(files) * 0.1))
    return {
        "train": files[2 * held_out :],
        "validation": files[:held_out],
        "test": files[held_out : 2 * held_out],
    }


def motion_targets(states):
    """Body velocity, body gyro, previous-body-frame pose delta (xyz + rotvec).

    Ground truth is used only to construct this target, never encoder inputs.
    """
    s = torch.as_tensor(states, dtype=torch.float32)
    r = rotation_matrix(s[:, 3:7])
    body_vel = (r.transpose(-1, -2) @ s[:, 7:10, None]).squeeze(-1)
    body_omega = (r.transpose(-1, -2) @ s[:, 10:13, None]).squeeze(-1)
    delta_pos = torch.zeros_like(body_vel)
    delta_rot = torch.zeros_like(body_vel)
    delta_pos[1:] = (r[:-1].transpose(-1, -2) @ (s[1:, :3] - s[:-1, :3]).unsqueeze(-1)).squeeze(-1)
    inverse = s[:-1, 3:7].clone()
    inverse[:, :3] *= -1
    delta_rot[1:] = quaternion_rotvec(quat_multiply(inverse, s[1:, 3:7]))
    return torch.cat((body_vel, body_omega, delta_pos, delta_rot), -1)


class FlightSequences(Dataset):
    def __init__(self, paths, length=64, stride=16):
        if length < 2 or stride < 1:
            raise ValueError("Invalid sequence length or stride")
        self.paths = list(paths)
        self.length = length
        self.windows = []
        for i, path in enumerate(self.paths):
            with h5py.File(path, "r") as f:
                if not f.attrs.get("complete", False) or f.attrs.get("schema_version") != 1:
                    raise ValueError(f"Incomplete or unsupported episode: {path}")
                n = len(f["time"])
                if len(f["actions"]) != n - 1:
                    raise ValueError(f"Misaligned episode: {path}")
                self.windows.extend((i, j) for j in range(0, n - length + 1, stride))
        if not self.windows:
            raise ValueError("No flights long enough for the requested sequence length")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start = self.windows[index]
        end = start + self.length
        with h5py.File(self.paths[episode], "r") as f:
            cfg = json.loads(f.attrs["config"])
            rgb = torch.from_numpy(f["rgb"][start:end].copy()).permute(0, 3, 1, 2).float() / 255
            depth = (
                torch.from_numpy(f["depth"][start:end].copy()).unsqueeze(1).float()
                / cfg["depth_far"]
            )
            mask = torch.from_numpy(f["depth_valid"][start:end].copy()).unsqueeze(1).bool()
            actions = torch.from_numpy(f["actions"][start : end - 1].copy()).flatten(1)
            imu = torch.from_numpy(f["imu"][start:end].copy()).float()
            targets = motion_targets(f["state"][max(0, start - 1) : end])[-self.length :]
        return {
            "image": torch.cat((rgb, depth.clamp(0, 1)), 1),
            "depth_valid": mask,
            "imu": imu,
            "actions": actions,
            "motion": targets,
        }
