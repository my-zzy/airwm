import h5py
import numpy as np
import pytest

from airwm.config import FlightConfig
from airwm.data import EpisodeWriter, FlightSequences, split_flights


def observation(t):
    return dict(
        rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        depth=np.ones((64, 64), dtype=np.float32),
        depth_valid=np.ones((64, 64), dtype=bool),
        imu=np.array([0, 0, 9.81, 0, 0, 0], dtype=np.float32),
        state=np.array([t, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32),
        time=np.float64(t),
        collision=np.bool_(False),
    )


def test_transition_alignment_and_no_state_inputs(tmp_path):
    path = tmp_path / "episode.h5"
    with EpisodeWriter(path, FlightConfig()) as writer:
        writer.append(observation(0))
        for i in range(1, 5):
            writer.append(observation(i / 20), np.full((5, 4), i / 10, dtype=np.float32))
    dataset = FlightSequences([path], length=3, stride=1)
    assert len(dataset) == 3
    batch = dataset[1]
    assert "state" not in batch
    assert batch["image"].shape == (3, 4, 64, 64)
    assert batch["actions"].shape == (2, 20)
    assert batch["actions"][0, 0].item() == pytest.approx(0.2)
    assert batch["motion"][0, 6].item() == pytest.approx(0.05)
    with h5py.File(path) as f:
        assert len(f["actions"]) == len(f["time"]) - 1


def test_failed_episode_is_not_trainable(tmp_path):
    path = tmp_path / "failed.h5"
    with pytest.raises(ValueError), EpisodeWriter(path, FlightConfig()) as writer:
        writer.append(observation(0))
        writer.append(observation(0.08), np.zeros((5, 4)))
    assert not path.exists()
    assert path.with_suffix(".h5.partial").exists()


def test_whole_flight_split(tmp_path):
    for i in range(10):
        (tmp_path / f"{i}.h5").touch()
    split = split_flights(tmp_path)
    assert [len(split[k]) for k in ("train", "validation", "test")] == [8, 1, 1]
    assert not (set(split["train"]) & set(split["validation"]))
    assert split == split_flights(tmp_path)
