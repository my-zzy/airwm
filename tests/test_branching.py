import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from airwm.branch_evaluate import effect_metrics, evaluate_branches, read_flight, validate_replay
from airwm.branching import perturb_actions, prepare_branches, reconstruct_motors
from airwm.config import FlightConfig
from airwm.data import EpisodeWriter
from airwm.dynamics import Motors
from airwm.model import ModelConfig, WorldModel


def test_interventions_change_only_the_declared_pulse():
    original = np.full((40, 5, 4), 0.65, dtype=np.float32)
    changed, clipped = perturb_actions(original, "collective_up")
    np.testing.assert_allclose(changed[:5], 0.67)
    np.testing.assert_array_equal(changed[5:], original[5:])
    np.testing.assert_array_equal(original, np.full_like(original, 0.65))
    assert clipped == 0
    for variant in ("control", "repeat"):
        np.testing.assert_array_equal(perturb_actions(original, variant)[0], original)


def test_intervention_motor_patterns_produce_declared_axes():
    original = np.full((40, 5, 4), 0.65, dtype=np.float32)
    for axis, variant in enumerate(("roll_positive", "pitch_positive", "yaw_positive")):
        actions, _ = perturb_actions(original, variant)
        motors = Motors()
        motors.speed.copy_(torch.from_numpy(actions[0, 0])[None])
        _, torque = motors.step(motors.speed.clone(), 0.005)
        assert torque[0, axis] > 0
        assert torque[0, [i for i in range(3) if i != axis]].abs().max() < 1e-6


def test_restored_rotor_lag_matches_uninterrupted_motor_updates():
    cfg = FlightConfig()
    rng = np.random.default_rng(8)
    actions = rng.uniform(0.5, 0.8, size=(40, 5, 4)).astype(np.float32)
    motors = Motors(cfg)
    motors.reset()
    for command in actions.reshape(-1, 4):
        for _ in range(cfg.physics_per_control):
            motors.step(torch.from_numpy(command)[None], 1 / cfg.physics_hz)
    np.testing.assert_array_equal(reconstruct_motors(actions, cfg), motors.speed[0].numpy())
    restored = Motors(cfg)
    restored.speed.copy_(torch.from_numpy(reconstruct_motors(actions, cfg))[None])
    next_command = torch.full((1, 4), 0.7)
    a = motors.step(next_command, 1 / cfg.physics_hz)
    b = restored.step(next_command, 1 / cfg.physics_hz)
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
    torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)


def test_effect_score_distinguishes_correct_ignored_and_wrong_direction():
    actual = np.array([0.0, 0.0, 0.2])
    correct = effect_metrics(actual, actual, "m", "position")
    ignored = effect_metrics(actual, np.zeros(3), "m", "position")
    wrong = effect_metrics(actual, -actual, "m", "position")
    assert correct["position_effect_gain"] == pytest.approx(1)
    assert correct["position_effect_error_m"] == 0
    assert ignored["position_effect_gain"] == 0
    assert ignored["position_effect_error_m"] == pytest.approx(0.2)
    assert wrong["position_effect_gain"] == pytest.approx(-1)
    assert wrong["position_effect_error_m"] == pytest.approx(0.4)


def test_branch_protocol_evaluation_and_bad_restore_rejected(tmp_path):
    """Synthetic fixtures check protocol/plumbing only, never prediction quality."""
    torch.set_num_threads(2)
    source = tmp_path / "source" / "flight.h5"
    cfg = FlightConfig()
    observation = {
        "rgb": np.full((64, 64, 3), 100, dtype=np.uint8),
        "depth": np.ones((64, 64), dtype=np.float32),
        "depth_valid": np.ones((64, 64), dtype=bool),
        "imu": np.array([0, 0, 9.81, 0, 0, 0], dtype=np.float32),
        "state": np.array([0, 0, 1.3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "collision": np.bool_(False),
    }
    commands = np.full((5, 4), 0.65, dtype=np.float32)
    with EpisodeWriter(
        source, cfg, {"phase": 0.0, "room": "procedural_6x6x3_v1", "synthetic_fixture": True}
    ) as writer:
        for frame in range(61):
            writer.append(
                {**observation, "time": np.float64(frame / 20)}, commands if frame else None
            )
    args = Namespace(
        source=source.parent, sources=1, start_frame=20, horizon=40, branches=["control", "repeat"]
    )
    plans = prepare_branches(args)
    folder = tmp_path / "branches"
    for plan in plans:
        with EpisodeWriter(folder / plan["filename"], cfg, plan["metadata"]) as writer:
            for frame in range(41):
                writer.append(
                    {**observation, "time": np.float64(frame / 20)},
                    plan["actions"][frame - 1] if frame else None,
                )
    config = ModelConfig(deter=64, hidden=64, embed=64, stoch=4, classes=4, cnn=4)
    model = WorldModel(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"model_config": config.to_dict(), "model": model.state_dict(), "step": 0}, checkpoint
    )
    output = tmp_path / "evaluation"
    evaluate_branches(
        Namespace(data=folder, checkpoint=checkpoint, output=output, context=20, device="cpu")
    )
    report = json.loads((output / "metrics.json").read_text())
    assert report["common_histories"] == 1 and report["branch_files"] == 2
    assert len(report["metrics"]) == 6
    assert not report["censored_horizons"]
    for row in report["metrics"]:
        assert row["predicted_position_effect_m"] == 0
        assert row["rgb_mse"] == row["unchanged_command_rgb_mse"]
    original = read_flight(source)
    control = read_flight(folder / plans[0]["filename"])
    repeat = read_flight(folder / plans[1]["filename"])
    repeat["state"][0, 0] += 0.1
    with pytest.raises(ValueError, match="State restoration failed"):
        validate_replay(original, control, repeat, 20)
