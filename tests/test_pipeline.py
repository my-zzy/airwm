import json

import numpy as np
import torch

from airwm.cli import main
from airwm.config import FlightConfig
from airwm.data import EpisodeWriter


def test_offline_train_checkpoint_and_open_loop_evaluation(tmp_path):
    """Synthetic fixtures exercise plumbing only; they are not flight training data."""
    torch.set_num_threads(2)
    data = tmp_path / "flights"
    for episode in range(3):
        with EpisodeWriter(
            data / f"{episode}.h5", FlightConfig(), {"synthetic_fixture": True}
        ) as writer:
            for t in range(12):
                obs = dict(
                    rgb=np.full((64, 64, 3), 100 + episode * 20 + t, dtype=np.uint8),
                    depth=np.ones((64, 64), dtype=np.float32),
                    depth_valid=np.ones((64, 64), dtype=bool),
                    imu=np.array([0, 0, 9.81, 0, 0, 0], dtype=np.float32),
                    state=np.array(
                        [t * 0.01, 0, 1, 0, 0, 0, 1, 0.2, 0, 0, 0, 0, 0],
                        dtype=np.float32,
                    ),
                    time=np.float64(t / 20),
                    collision=np.bool_(False),
                )
                writer.append(obs, np.full((5, 4), 0.65, dtype=np.float32) if t else None)
    run = tmp_path / "run"
    assert (
        main(
            [
                "train",
                "--data",
                str(data),
                "--run",
                str(run),
                "--steps",
                "2",
                "--batch-size",
                "1",
                "--length",
                "4",
                "--tiny",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (run / "best.pt").is_file()
    output = tmp_path / "evaluation"
    assert (
        main(
            [
                "evaluate",
                "--checkpoint",
                str(run / "best.pt"),
                "--output",
                str(output),
                "--context",
                "2",
                "--horizon",
                "10",
                "--windows",
                "1",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    report = json.loads((output / "metrics.json").read_text())
    assert report["split"] == "test"
    assert len(report["metrics"]) == 1
    assert np.isfinite(report["metrics"][0]["rgb_mse"])
    assert (output / "prediction.png").is_file()
