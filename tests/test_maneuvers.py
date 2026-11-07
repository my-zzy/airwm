import math

import numpy as np
import torch

from airwm.dynamics import Motors, controller
from airwm.maneuvers import PATTERNS, ManeuverPolicy


def hover_state(policy):
    position, yaw = policy.target(0)
    state = torch.zeros(1, 13)
    state[:, :3] = position
    state[:, 5] = torch.sin(yaw / 2)
    state[:, 6] = torch.cos(yaw / 2)
    return state


def test_maneuver_seeds_bounds_revisits_and_axis_coverage():
    a = ManeuverPolicy(12, 0, 45)
    b = ManeuverPolicy(12, 0, 45)
    c = ManeuverPolicy(12, 1, 45)
    assert a.metadata() == b.metadata()
    assert a.metadata() != c.metadata()
    assert (a.positions[:, :2].abs() <= 1.1).all()
    assert ((a.positions[:, 2] >= 1.1) & (a.positions[:, 2] <= 1.9)).all()
    assert torch.equal(a.positions[3], a.positions[0])
    for start in range(0, 20, 4):
        assert {e["axis"] for e in a.events[start : start + 4]} == set(PATTERNS)
    for e in a.events:
        assert 15 <= e["end_control"] - e["start_control"] <= 25
        assert e["amplitude"] != 0
    patterns = np.array(list(PATTERNS.values()))
    np.testing.assert_array_equal(patterns @ patterns.T, 4 * np.eye(4))


def test_pulse_holds_base_command_then_resumes_feedback():
    policy = ManeuverPolicy(3, 0, 4)
    motors = Motors()
    motors.reset()
    state = hover_state(policy)
    event = policy.events[0]
    start, end = event["start_control"], event["end_control"]
    original = policy.command(start - 1, motors, state)
    first = policy.command(start, motors, state)
    expected = original + event["amplitude"] * torch.tensor(PATTERNS[event["axis"]])
    torch.testing.assert_close(first, expected)
    state[:, 7] = 0.1
    for t in range(start + 1, end):
        torch.testing.assert_close(policy.command(t, motors, state), first)
    target, yaw = policy.target(end)
    resumed = controller(
        motors, state[:, :3], state[:, 3:7], state[:, 7:10], state[:, 10:13], target, yaw
    )
    torch.testing.assert_close(policy.command(end, motors, state), resumed)
    assert not torch.equal(first, resumed)
    assert event["applied_controls"] == end - start
    assert policy.clipped_values == 0


def test_recovery_cancels_pulse_and_retains_accounting():
    policy = ManeuverPolicy(3, 0, 4)
    state = hover_state(policy)
    motors = Motors()
    motors.reset()
    event = policy.events[0]
    policy.command(event["start_control"], motors, state)
    state[:, 3:7] = torch.tensor([math.sin(math.pi / 8), 0, 0, math.cos(math.pi / 8)])
    command = policy.command(event["start_control"] + 1, motors, state)
    assert event["cancelled"] and event["applied_controls"] == 1
    assert policy.recovery_controls == 1
    assert torch.isfinite(command).all() and ((command >= 0) & (command <= 1)).all()
    policy.command(event["start_control"] + 2, motors, hover_state(policy))
    assert event["applied_controls"] == 1
    assert policy.metadata()["pulse_events"][0]["cancelled"]
