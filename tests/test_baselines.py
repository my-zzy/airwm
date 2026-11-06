import math

import torch

from airwm.baselines import integrate_relative_pose, nominal_rollout
from airwm.config import FlightConfig
from airwm.data import motion_targets
from airwm.dynamics import rotvec_quaternion


def test_nominal_hover_and_linear_drag():
    cfg = FlightConfig()
    initial = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hover = math.sqrt(cfg.mass * cfg.gravity / (4 * cfg.max_rotor_thrust))
    actions = torch.full((20, 5, 4), hover)
    prediction = nominal_rollout(initial, actions, torch.empty(0, 5, 4), cfg)
    assert abs(prediction[-1, 2] - 1) < 1e-5
    assert abs(prediction[-1, 7] - math.exp(-cfg.linear_drag / cfg.mass)) < 0.001
    torch.testing.assert_close(prediction[-1, 3:7], initial[3:7])


def test_pose_composition_uses_previous_body_frame():
    pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    deltas = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    result = integrate_relative_pose(pose, deltas)
    torch.testing.assert_close(result[-1, :3], torch.tensor([1.0, 1.0, 0.0]), atol=1e-6, rtol=0)


def test_rotation_target_at_half_turn():
    states = torch.zeros(2, 13)
    states[0, 6] = 1
    states[1, 3:7] = rotvec_quaternion(torch.tensor([0.0, 0.0, math.pi]))
    target = motion_targets(states)
    assert abs(target[1, 9:].norm() - math.pi) < 1e-5
