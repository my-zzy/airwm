"""Compact categorical RSSM inspired by DreamerV3, without actor/critic training.

This is a project baseline, not a reproduction of every DreamerV3 detail.
It retains stochastic straight-through latents, unimix, balanced stop-gradient
KL losses and free nats. RGB/depth and motion losses are project adaptations.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    deter: int = 1024
    hidden: int = 512
    embed: int = 512
    stoch: int = 32
    classes: int = 32
    cnn: int = 32
    action_dim: int = 20
    unimix: float = 0.01
    free_nats: float = 1.0

    def to_dict(self):
        return asdict(self)


def mlp(in_size, hidden, out_size):
    return nn.Sequential(
        nn.Linear(in_size, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_size),
    )


class WorldModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        channels = [4, c.cnn, 2 * c.cnn, 4 * c.cnn, 8 * c.cnn]
        encoder = []
        for a, b in zip(channels[:-1], channels[1:]):
            encoder.extend((nn.Conv2d(a, b, 4, stride=2, padding=1), nn.GroupNorm(1, b), nn.SiLU()))
        self.image_encoder = nn.Sequential(
            *encoder,
            nn.Flatten(),
            nn.Linear(8 * c.cnn * 4 * 4, c.embed),
            nn.LayerNorm(c.embed),
            nn.SiLU(),
        )
        self.imu_encoder = mlp(6, 128, 128)
        self.fusion = mlp(c.embed + 128, c.hidden, c.embed)
        latent = c.stoch * c.classes
        self.transition_input = mlp(latent + c.action_dim, c.hidden, c.hidden)
        self.recurrent = nn.GRUCell(c.hidden, c.deter)
        self.prior = mlp(c.deter, c.hidden, latent)
        self.posterior = mlp(c.deter + c.embed, c.hidden, latent)
        self.decoder_input = nn.Linear(c.deter + latent, 8 * c.cnn * 4 * 4)
        decoder = []
        for a, b in [
            (8 * c.cnn, 4 * c.cnn),
            (4 * c.cnn, 2 * c.cnn),
            (2 * c.cnn, c.cnn),
        ]:
            decoder.extend(
                (
                    nn.ConvTranspose2d(a, b, 4, stride=2, padding=1),
                    nn.GroupNorm(1, b),
                    nn.SiLU(),
                )
            )
        decoder.append(nn.ConvTranspose2d(c.cnn, 4, 4, stride=2, padding=1))
        self.decoder = nn.Sequential(*decoder)
        self.motion = mlp(c.deter + latent, c.hidden, 12)
        # Velocity, angular velocity, per-frame translation and rotation scales.
        self.register_buffer(
            "motion_scale", torch.tensor([2.0] * 3 + [3.0] * 3 + [0.1] * 3 + [0.15] * 3)
        )

    def initial(self, batch, device):
        return (
            torch.zeros(batch, self.cfg.deter, device=device),
            torch.zeros(batch, self.cfg.stoch * self.cfg.classes, device=device),
        )

    def probs(self, logits):
        p = logits.float().reshape(*logits.shape[:-1], self.cfg.stoch, self.cfg.classes).softmax(-1)
        return (1 - self.cfg.unimix) * p + self.cfg.unimix / self.cfg.classes

    def sample(self, probs, deterministic=False):
        if deterministic:
            indices = probs.argmax(-1)
        else:
            indices = torch.multinomial(probs.reshape(-1, self.cfg.classes), 1).reshape(
                probs.shape[:-1]
            )
        onehot = F.one_hot(indices, self.cfg.classes).to(probs.dtype)
        return (onehot + probs - probs.detach()).flatten(-2)

    def encode(self, images, imu):
        shape = images.shape[:2]
        visual = self.image_encoder(images.flatten(0, 1))
        # Accelerometer includes gravity; symlog bounds sensor dynamic range.
        sensor = imu.flatten(0, 1)
        sensor = torch.sign(sensor) * torch.log1p(sensor.abs())
        return self.fusion(torch.cat((visual, self.imu_encoder(sensor)), -1)).reshape(*shape, -1)

    def step(self, state, action, embedding=None, deterministic=False):
        h, z = state
        x = self.transition_input(torch.cat((z, action), -1))
        h = self.recurrent(x, h)
        prior = self.probs(self.prior(h))
        posterior = (
            self.probs(self.posterior(torch.cat((h, embedding), -1)))
            if embedding is not None
            else prior
        )
        z = self.sample(posterior, deterministic)
        return (h, z), prior, posterior

    def decode(self, features):
        prefix = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        image = self.decoder(self.decoder_input(flat).reshape(-1, 8 * self.cfg.cnn, 4, 4)).sigmoid()
        motion = self.motion(flat).float() * self.motion_scale
        return image.reshape(*prefix, 4, 64, 64), motion.reshape(*prefix, 12)

    def observe(self, images, imu, actions, deterministic=False):
        b, t = images.shape[:2]
        if actions.shape != (b, t - 1, self.cfg.action_dim):
            raise ValueError("Expected one outgoing action block per observation transition")
        embeddings = self.encode(images, imu)
        state = self.initial(b, images.device)
        features = []
        priors = []
        posteriors = []
        for i in range(t):
            action = (
                torch.zeros(b, self.cfg.action_dim, device=images.device)
                if i == 0
                else actions[:, i - 1]
            )
            state, p, q = self.step(state, action, embeddings[:, i], deterministic)
            features.append(torch.cat(state, -1))
            priors.append(p)
            posteriors.append(q)
        return (
            state,
            torch.stack(features, 1),
            torch.stack(priors, 1),
            torch.stack(posteriors, 1),
        )

    def loss(self, batch):
        _, features, p, q = self.observe(batch["image"], batch["imu"], batch["actions"])
        prediction, motion = self.decode(features)
        rgb = F.mse_loss(prediction[:, :, :3].float(), batch["image"][:, :, :3].float())
        mask = batch["depth_valid"].float()
        depth = (
            (prediction[:, :, 3:].float() - batch["image"][:, :, 3:].float()).square() * mask
        ).sum() / mask.sum().clamp_min(1)
        target = batch["motion"].float() / self.motion_scale
        motion_loss = F.smooth_l1_loss(motion / self.motion_scale, target)

        def kl(a, b):
            return (a * (a.log() - b.log())).sum((-1, -2)).clamp_min(self.cfg.free_nats).mean()

        dynamics = kl(q.detach(), p)
        representation = kl(q, p.detach())
        # Fixed Gaussian observation losses. Tune depth/motion weighting on held-out flights.
        # Sum Gaussian reconstruction NLL over pixels, then average frames. Using
        # only per-pixel mean here would shrink reconstruction relative to KL.
        total = (
            0.5 * (rgb * 3 * 64 * 64 + depth * 64 * 64)
            + 0.1 * motion_loss
            + 0.5 * dynamics
            + 0.1 * representation
        )
        return total, {
            k: v.detach()
            for k, v in dict(
                loss=total,
                rgb_mse=rgb,
                depth_mse=depth,
                motion_loss=motion_loss,
                kl_dynamics=dynamics,
                kl_representation=representation,
            ).items()
        }

    @torch.no_grad()
    def predict(self, images, imu, context_actions, future_actions, deterministic=True):
        state, _, _, _ = self.observe(images, imu, context_actions, deterministic)
        features = []
        for action in future_actions.unbind(1):
            state, _, _ = self.step(state, action, deterministic=deterministic)
            features.append(torch.cat(state, -1))
        if not features:
            raise ValueError("At least one future action block is required")
        return self.decode(torch.stack(features, 1))
