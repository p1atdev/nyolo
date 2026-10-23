"""Blocks independently composed from the YOLO26 paper's supplementary diagrams."""

import torch
from torch import Tensor, nn


class Conv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 1,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                kernel // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
            nn.SiLU() if activation else nn.Identity(),
        )


class Bottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            Conv(channels, channels, 3), Conv(channels, channels, 3)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)


class C3k(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.left = nn.Sequential(
            Conv(channels, hidden), Bottleneck(hidden), Bottleneck(hidden)
        )
        self.right = Conv(channels, hidden)
        self.merge = Conv(2 * hidden, channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.merge(torch.cat((self.left(x), self.right(x)), 1))


class Attention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.heads = max(channels // 64, 1)
        self.value_dim = channels // self.heads
        self.key_dim = self.value_dim // 2
        self.qkv = Conv(
            channels, channels + 2 * self.heads * self.key_dim, activation=False
        )
        self.position = Conv(channels, channels, 3, groups=channels, activation=False)
        self.project = Conv(channels, channels, activation=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x).reshape(
            b, self.heads, 2 * self.key_dim + self.value_dim, h * w
        )
        q, k, v = qkv.split((self.key_dim, self.key_dim, self.value_dim), 2)
        attention = (q.transpose(-1, -2) @ k * self.key_dim**-0.5).softmax(-1)
        mixed = (v @ attention.transpose(-1, -2)).reshape(b, c, h, w)
        return self.project(mixed + self.position(v.reshape(b, c, h, w)))


class PSABlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = Attention(channels)
        self.ffn = nn.Sequential(
            Conv(channels, 2 * channels), Conv(2 * channels, channels, activation=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(x)
        return x + self.ffn(x)


class C3k2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        c3k: bool = False,
        attention: bool = False,
        repeats: int = 1,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.split = Conv(in_channels, hidden * 2)
        block = C3k(hidden) if c3k else Bottleneck(hidden)
        self.block = nn.Sequential(block, PSABlock(hidden)) if attention else block
        self.extra_blocks = nn.ModuleList(
            C3k(hidden) if c3k else Bottleneck(hidden) for _ in range(repeats - 1)
        )
        self.merge = Conv(hidden * (2 + repeats), out_channels)

    def forward(self, x: Tensor) -> Tensor:
        left, right = self.split(x).chunk(2, 1)
        pieces = [left, right, self.block(right)]
        for block in self.extra_blocks:
            pieces.append(block(pieces[-1]))
        return self.merge(torch.cat(pieces, 1))


class SPPF(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.reduce = Conv(channels, channels // 2)
        self.pool = nn.MaxPool2d(5, 1, 2)
        self.merge = Conv(2 * channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        a = self.reduce(x)
        b = self.pool(a)
        c = self.pool(b)
        return self.merge(torch.cat((a, b, c, self.pool(c)), 1))


class C2PSA(nn.Module):
    def __init__(self, channels: int, repeats: int = 1) -> None:
        super().__init__()
        self.split = Conv(channels, channels)
        self.block = PSABlock(channels // 2)
        self.extra_blocks = nn.ModuleList(
            PSABlock(channels // 2) for _ in range(repeats - 1)
        )
        self.merge = Conv(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        left, right = self.split(x).chunk(2, 1)
        right = self.block(right)
        for block in self.extra_blocks:
            right = block(right)
        return self.merge(torch.cat((left, right), 1))
