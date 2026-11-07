"""Seeded waypoint flights with independent motor excitation for data collection.

This privileged-state behavior policy is not an input to the world model. During
a pulse it holds the controller's pre-pulse command plus an axis offset; normal
feedback resumes afterwards. Every actual command is recorded by the collector.
"""

import math

import numpy as np
import torch

from airwm.config import FlightConfig
from airwm.dynamics import controller, rotation_matrix

PATTERNS = {
    "collective": [1, 1, 1, 1],
    "roll": [1, 1, -1, -1],
    "pitch": [-1, 1, 1, -1],
    "yaw": [1, -1, 1, -1],
}
AMPLITUDES = {
    "collective": [0.005, 0.01, 0.02],
    "roll": [0.0005, 0.001, 0.0015],
    "pitch": [0.0005, 0.001, 0.0015],
    "yaw": [0.005, 0.01, 0.015],
}


class ManeuverPolicy:
    def __init__(self, seed, episode, seconds, device="cpu", cfg=FlightConfig()):
        rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 719]))
        self.cfg = cfg
        self.device = device
        self.waypoint_steps = 4 * cfg.control_hz
        count = math.ceil(seconds / 4) + 1
        positions = rng.uniform([-1.1, -1.1, 1.1], [1.1, 1.1, 1.9], (count, 3))
        # Return to the start periodically; this is collection diversity, not a
        # dedicated memory-retention evaluation.
        positions[3::4] = positions[0]
        headings = np.empty(count)
        headings[0] = rng.uniform(-math.pi, math.pi)
        headings[1:] = headings[0] + np.cumsum(rng.uniform(-1.2, 1.2, count - 1))
        self.positions = torch.tensor(positions, dtype=torch.float32, device=device)
        self.headings = torch.tensor(headings, dtype=torch.float32, device=device)
        self.events = []
        axes = list(PATTERNS)
        order = []
        # Pulses start one second into each two-second slot. Axis order, sign,
        # size and duration vary independently of position and image history.
        for i, start in enumerate(
            range(cfg.control_hz, round(seconds * cfg.control_hz), 2 * cfg.control_hz)
        ):
            if i % 4 == 0:
                order = rng.permutation(axes).tolist()
            axis = order[i % 4]
            amplitude = float(rng.choice(AMPLITUDES[axis]) * rng.choice([-1, 1]))
            duration = int(rng.choice([15, 20, 25]))  # 0.15, 0.20, 0.25 s at 100 Hz
            duration = round(duration * cfg.control_hz / 100)
            end = min(start + duration, round(seconds * cfg.control_hz))
            self.events.append(
                {
                    "start_control": start,
                    "end_control": end,
                    "axis": axis,
                    "amplitude": amplitude,
                    "applied_controls": 0,
                    "cancelled": False,
                }
            )
        self.offsets = [
            torch.tensor(PATTERNS[e["axis"]], device=device)[None] * e["amplitude"]
            for e in self.events
        ]
        self.event_index = 0
        self.held_command = None
        self.recovery_controls = 0
        self.clipped_values = 0

    def target(self, control_step):
        index = min(control_step // self.waypoint_steps, len(self.positions) - 1)
        return self.positions[index : index + 1], self.headings[index : index + 1]

    def metadata(self):
        return {
            "maneuver_protocol": "waypoints_axis_pulses_v1",
            "waypoint_period_seconds": self.waypoint_steps / self.cfg.control_hz,
            "waypoints_xyz_m": self.positions.cpu().tolist(),
            "waypoint_yaw_rad": self.headings.cpu().tolist(),
            "pulse_mode": "hold pre-pulse feedback command plus offset, then resume feedback",
            "pulse_units": "normalized desired rotor speed",
            "motor_patterns": PATTERNS,
            "pulse_events": [dict(e) for e in self.events],
            "recovery_controls": self.recovery_controls,
            "clipped_command_values": self.clipped_values,
            "recovery_envelope": {"abs_xy_m": 1.65, "z_m": [0.9, 2.3], "tilt_degrees": 30},
        }

    def command(self, control_step, motors, state):
        target, yaw = self.target(control_step)
        pos, quat = state[:, :3], state[:, 3:7]
        # Cancel a pulse and return toward the room centre if the drone leaves
        # the collection envelope. Never omit the resulting data from the file.
        recover = bool(
            (
                (pos[:, :2].abs() > 1.65).any()
                | (pos[:, 2] < 0.9).any()
                | (pos[:, 2] > 2.3).any()
                | (rotation_matrix(quat)[:, 2, 2] < math.cos(math.pi / 6)).any()
            )
        )
        if recover:
            target = torch.tensor([[0.0, 0.0, 1.5]], device=self.device)
            self.recovery_controls += 1
        feedback = controller(motors, pos, quat, state[:, 7:10], state[:, 10:13], target, yaw)
        while (
            self.event_index < len(self.events)
            and control_step >= self.events[self.event_index]["end_control"]
        ):
            self.event_index += 1
            self.held_command = None
        if self.event_index == len(self.events):
            return feedback
        event = self.events[self.event_index]
        if control_step < event["start_control"]:
            return feedback
        if recover:
            event["cancelled"] = True
        if event["cancelled"]:
            return feedback
        if self.held_command is None:
            self.held_command = feedback.clone()
        command = self.held_command + self.offsets[self.event_index]
        self.clipped_values += int(((command < 0) | (command > 1)).sum())
        event["applied_controls"] += 1
        return command.clamp(0, 1)
