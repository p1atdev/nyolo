from math import log, pi

import torch
from torch import Tensor, nn


class AffineCoupling(nn.Module):
    def __init__(self, reverse: bool, hidden: int) -> None:
        super().__init__()
        self.reverse = reverse
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        output = self.net[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.reverse:
            x = x.flip(-1)
        fixed, moving = x[..., :1], x[..., 1:]
        scale, shift = self.net(fixed).chunk(2, -1)
        log_scale = 2 * scale.tanh()
        z = torch.cat((fixed, (moving - shift) * (-log_scale).exp()), -1)
        return (z.flip(-1) if self.reverse else z), -log_scale.squeeze(-1)


class ResidualFlow(nn.Module):
    """A 2D RealNVP density used only during training."""

    def __init__(self, hidden: int = 64, layers: int = 6) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            AffineCoupling(bool(i % 2), hidden) for i in range(layers)
        )

    def log_prob(self, residual: Tensor) -> Tensor:
        z = residual
        log_det = torch.zeros_like(residual[..., 0])
        for layer in self.layers:
            z, delta = layer(z)
            log_det = log_det + delta
        return -0.5 * (z.square() + log(2 * pi)).sum(-1) + log_det


def rle_loss(residual: Tensor, sigma_logits: Tensor, flow: ResidualFlow) -> Tensor:
    """Per-joint RLE (paper eq. 11), residual expressed in person-scale units."""
    sigma = sigma_logits.sigmoid().clamp_min(1e-4)
    normalized = residual / sigma
    base_nll = (torch.log(2 * sigma) + normalized.abs()).sum(-1)
    return sigma.log().sum(-1) - flow.log_prob(normalized) + base_nll
