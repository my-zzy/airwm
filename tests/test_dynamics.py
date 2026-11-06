import math

import torch

from airwm.config import FlightConfig
from airwm.dynamics import Motors, controller, imu_sample, rotation_matrix


def test_hover_wrench_and_controller():
    motors = Motors()
    motors.reset()
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    zero = torch.zeros(1, 3)
    command = controller(motors, zero, quat, zero, zero, zero, torch.zeros(1))
    force, torque = motors.step(command, 0.005)
    torch.testing.assert_close(force, torch.tensor([[0.0, 0.0, 0.5 * 9.81]]))
    torch.testing.assert_close(torque, zero, atol=1e-6, rtol=0)


def test_motor_mixing_signs_and_lag():
    motors = Motors()
    requested = torch.tensor([[0.01, -0.02, 0.005]])
    command = motors.mix(torch.tensor([4.0]), requested)
    motors.speed.copy_(command)
    force, torque = motors.step(command, 0.005)
    torch.testing.assert_close(force[:, 2], torch.tensor([4.0]))
    torch.testing.assert_close(torque, requested, atol=1e-6, rtol=1e-5)
    motors.reset(hover=False)
    motors.step(torch.ones(1, 4), motors.cfg.motor_time_constant)
    torch.testing.assert_close(motors.speed, torch.full((1, 4), 1 - math.exp(-1)))


def test_specific_force_in_hover_and_freefall():
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    z = torch.zeros(1, 3)
    torch.testing.assert_close(
        imu_sample(q, z, z, z, 0.005), torch.tensor([[0.0, 0.0, 9.81, 0.0, 0.0, 0.0]])
    )
    fallen = torch.tensor([[0.0, 0.0, -9.81 * 0.005]])
    torch.testing.assert_close(imu_sample(q, fallen, z, z, 0.005), torch.zeros(1, 6))
    yaw90 = torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]])
    torch.testing.assert_close(
        rotation_matrix(yaw90)[0, :, 0],
        torch.tensor([0.0, 1.0, 0.0]),
        atol=1e-6,
        rtol=0,
    )


def test_rate_alignment():
    cfg = FlightConfig()
    assert cfg.controls_per_frame == 5
    assert cfg.physics_per_control == 2
