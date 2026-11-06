"""Motor dynamics and a privileged-state collection controller, independent of Isaac."""

import math

import torch
import torch.nn.functional as F

from airwm.config import FlightConfig


def rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Body to world rotation for normalized xyzw quaternions."""
    x, y, z, w = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def body_vector(q, world_vector):
    return (rotation_matrix(q).transpose(-1, -2) @ world_vector.unsqueeze(-1)).squeeze(-1)


def quat_multiply(a, b):
    """Hamilton product for xyzw quaternions, composing body-frame rotations."""
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    return torch.cat(
        (aw * bv + bw * av + torch.linalg.cross(av, bv), aw * bw - (av * bv).sum(-1, keepdim=True)),
        -1,
    )


def rotvec_quaternion(vector):
    angle = vector.norm(dim=-1, keepdim=True)
    scale = 0.5 * torch.sinc(angle / (2 * math.pi))
    return torch.cat((vector * scale, torch.cos(angle / 2)), -1)


def quaternion_rotvec(quaternion):
    q = F.normalize(quaternion, dim=-1)
    q = torch.where(q[..., 3:] < 0, -q, q)
    sine = q[..., :3].norm(dim=-1, keepdim=True)
    scale = torch.where(
        sine > 1e-7,
        2 * torch.atan2(sine, q[..., 3:]) / sine.clamp_min(1e-7),
        torch.full_like(sine, 2.0),
    )
    return q[..., :3] * scale


class Motors:
    """Four independent first-order motors, per-rotor thrust and reaction torque.

    Motor order: front-left, rear-left, rear-right, front-right; x forward/y left/z up.
    Commands are normalized desired rotor speed. Rotor forces are analytically
    summed at their application points into an equivalent body-centre wrench.
    """

    def __init__(self, cfg=FlightConfig(), device="cpu", num_envs=1):
        self.cfg = cfg
        a = cfg.arm_length / math.sqrt(2)
        self.positions = torch.tensor(
            [[a, a, 0], [-a, a, 0], [-a, -a, 0], [a, -a, 0]], device=device
        )
        self.spin = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device)
        self.speed = torch.zeros(num_envs, 4, device=device)
        self.allocation = torch.stack(
            (
                torch.ones(4, device=device),
                self.positions[:, 1],
                -self.positions[:, 0],
                self.spin * cfg.yaw_moment_ratio,
            )
        )
        self.inverse = torch.linalg.inv(self.allocation)

    def reset(self, hover=True):
        self.speed.fill_(
            math.sqrt(self.cfg.mass * self.cfg.gravity / (4 * self.cfg.max_rotor_thrust))
            if hover
            else 0
        )

    def mix(self, collective, torque):
        thrusts = torch.cat((collective[..., None], torque), -1) @ self.inverse.T
        return (thrusts.clamp(0, self.cfg.max_rotor_thrust) / self.cfg.max_rotor_thrust).sqrt()

    def step(self, commands, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.speed.lerp_(commands.clamp(0, 1), 1 - math.exp(-dt / self.cfg.motor_time_constant))
        thrusts = self.cfg.max_rotor_thrust * self.speed.square()
        wrench = thrusts @ self.allocation.T
        force = torch.zeros_like(wrench[..., 1:])
        force[..., 2] = wrench[..., 0]
        return force, wrench[..., 1:]


def controller(motors, pos, quat, vel, omega_world, target, target_yaw):
    """Geometric position/attitude PD for data collection, with simulator state input."""
    r = rotation_matrix(quat)
    acceleration = 2.5 * (target - pos) - 2.6 * vel
    acceleration[..., :2].clamp_(-2, 2)
    acceleration[..., 2].clamp_(-3, 3)
    acceleration[..., 2] += motors.cfg.gravity
    desired_z = F.normalize(acceleration, dim=-1)
    heading = torch.stack((target_yaw.cos(), target_yaw.sin(), torch.zeros_like(target_yaw)), -1)
    desired_y = F.normalize(torch.linalg.cross(desired_z, heading), dim=-1)
    desired_x = torch.linalg.cross(desired_y, desired_z)
    desired_r = torch.stack((desired_x, desired_y, desired_z), -1)
    error_matrix = (desired_r.transpose(-1, -2) @ r - r.transpose(-1, -2) @ desired_r) / 2
    error = torch.stack(
        (error_matrix[..., 2, 1], error_matrix[..., 0, 2], error_matrix[..., 1, 0]), -1
    )
    omega_body = (r.transpose(-1, -2) @ omega_world.unsqueeze(-1)).squeeze(-1)
    torque = -0.12 * error - 0.025 * omega_body
    collective = motors.cfg.mass * (acceleration * r[..., :, 2]).sum(-1)
    return motors.mix(collective.clamp_min(0), torque)


def imu_sample(quat, velocity, previous_velocity, omega_world, dt, gravity=9.81):
    """Ideal body-frame accelerometer specific force and gyroscope, without noise."""
    acceleration = (velocity - previous_velocity) / dt
    acceleration = acceleration.clone()
    acceleration[..., 2] += gravity
    return torch.cat((body_vector(quat, acceleration), body_vector(quat, omega_world)), -1)
