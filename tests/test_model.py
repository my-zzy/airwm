import torch

from airwm.model import ModelConfig, WorldModel


def small_model():
    return WorldModel(ModelConfig(deter=64, hidden=64, embed=64, stoch=4, classes=4, cnn=4))


def test_training_gradients_and_action_conditioning():
    torch.manual_seed(2)
    torch.set_num_threads(2)
    model = small_model()
    batch = dict(
        image=torch.rand(2, 4, 4, 64, 64),
        imu=torch.rand(2, 4, 6),
        actions=torch.rand(2, 3, 20),
        depth_valid=torch.ones(2, 4, 1, 64, 64, dtype=torch.bool),
        motion=torch.randn(2, 4, 12),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, _ = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.transition_input[0].weight.grad.abs().sum() > 0
    assert model.posterior[0].weight.grad.abs().sum() > 0
    assert model.image_encoder[0].weight.grad.abs().sum() > 0
    optimizer.step()
    a, motion_a = model.predict(
        batch["image"][:, :2],
        batch["imu"][:, :2],
        batch["actions"][:, :1],
        torch.zeros(2, 2, 20),
    )
    b, motion_b = model.predict(
        batch["image"][:, :2],
        batch["imu"][:, :2],
        batch["actions"][:, :1],
        torch.ones(2, 2, 20),
    )
    assert a.shape == (2, 2, 4, 64, 64)
    assert motion_a.shape == (2, 2, 12)
    assert not torch.allclose(a, b)
    assert not torch.allclose(motion_a, motion_b)


def test_invalid_depth_does_not_change_loss():
    model = small_model()
    batch = dict(
        image=torch.rand(1, 2, 4, 64, 64),
        imu=torch.rand(1, 2, 6),
        actions=torch.rand(1, 1, 20),
        depth_valid=torch.zeros(1, 2, 1, 64, 64, dtype=torch.bool),
        motion=torch.zeros(1, 2, 12),
    )
    _, metrics = model.loss(batch)
    assert metrics["depth_mse"] == 0
