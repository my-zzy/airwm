"""Privileged-state nominal dynamics for evaluation, never a world-model input."""

import torch
import torch.nn.functional as F

from airwm.config import FlightConfig
from airwm.dynamics import Motors, body_vector, quat_multiply, rotation_matrix, rotvec_quaternion


def integrate_relative_pose(initial_pose, relative_deltas):
    """Anchor predicted previous-body-frame pose increments at a known initial pose."""
    position = initial_pose[:3].clone()
    quaternion = initial_pose[3:7].clone()
    poses = []
    for delta in relative_deltas:
        position = position + rotation_matrix(quaternion) @ delta[:3]
        quaternion = F.normalize(quat_multiply(quaternion, rotvec_quaternion(delta[3:])), dim=-1)
        poses.append(torch.cat((position, quaternion)))
    return torch.stack(poses)


def nominal_rollout(initial_state, future_actions, past_actions, cfg: FlightConfig):
    """Semi-implicit rigid-body rollout at the collector's physics/control rates.

    This baseline receives the last ground-truth state and nominal parameters.
    Rotor speed is reconstructed from recorded commands since episode start.
    Contacts and PhysX solver details are not reproduced here.
    """
    device = initial_state.device
    motors = Motors(cfg, device)
    motors.reset()
    for command in past_actions.reshape(-1, 4):
        motors.step(command[None], 1 / cfg.control_hz)
    pos = initial_state[:3].clone()
    q = initial_state[3:7].clone()
    vel = initial_state[7:10].clone()
    omega = body_vector(q, initial_state[10:13])
    inertia = torch.tensor([0.003, 0.003, 0.005], device=device)
    gravity = torch.tensor([0.0, 0.0, -cfg.gravity], device=device)
    dt = 1 / cfg.physics_hz
    states = []
    for block in future_actions:
        for command in block:
            for _ in range(cfg.physics_per_control):
                force, torque = motors.step(command[None], dt)
                acceleration = (
                    rotation_matrix(q) @ force[0] - cfg.linear_drag * vel
                ) / cfg.mass + gravity
                torque = torque[0] - cfg.angular_drag * omega
                omega = omega + (torque - torch.linalg.cross(omega, inertia * omega)) / inertia * dt
                q = F.normalize(quat_multiply(q, rotvec_quaternion(omega * dt)), dim=-1)
                vel = vel + acceleration * dt
                pos = pos + vel * dt
        states.append(torch.cat((pos, q, vel, rotation_matrix(q) @ omega)))
    return torch.stack(states)
