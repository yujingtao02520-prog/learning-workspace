import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    """为 GroupNorm 选择一个能整除通道数的组数。"""

    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """把标量时间 t 编码成 sinusoidal embedding，供卷积网络使用。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        # 训练脚本传入归一化时间 [0, 1]，放大后能让正弦时间特征更有区分度。
        t = t.float().reshape(-1) * 1000.0

        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=t.device) / max(half_dim - 1, 1)
        freqs = torch.exp(exponent)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConvBlock(nn.Module):
    """卷积块：卷积特征 + 时间 embedding 加法注入 + 残差连接。"""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)

        # 将时间信息投影到通道维，并广播到 HxW 后加到特征图上。
        time_bias = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_bias
        h = F.silu(h)

        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """低分辨率瓶颈处的轻量 self-attention，用于增强全局结构建模。"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        hidden = self.norm(x).flatten(2).transpose(1, 2)
        hidden, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        hidden = hidden.transpose(1, 2).reshape(b, c, h, w)
        return x + hidden


class TinyUNet(nn.Module):
    """
    可放大的 U-Net。

    - num_downs=1：接近最初的小 demo；
    - num_downs=2/3：更适合 64x64 和 5090 这类 GPU。
    同一个网络可用于 DDPM 的 epsilon_theta，也可用于 Flow Matching 的 v_theta。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 64,
        num_downs: int = 3,
        use_attention: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_downs not in {1, 2, 3}:
            raise ValueError("num_downs 目前支持 1、2 或 3。")

        self.num_downs = num_downs
        self.use_attention = use_attention
        time_dim = model_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        c1 = model_channels
        c2 = model_channels * 2
        c3 = model_channels * 4
        c4 = model_channels * 8

        self.enc1 = ConvBlock(in_channels, c1, time_dim, dropout=dropout)
        self.down = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.enc2 = ConvBlock(c2, c2, time_dim, dropout=dropout)

        if num_downs >= 2:
            self.down2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)
            self.enc3 = ConvBlock(c3, c3, time_dim, dropout=dropout)

        if num_downs >= 3:
            self.down3 = nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1)
            self.enc4 = ConvBlock(c4, c4, time_dim, dropout=dropout)

        bottleneck_channels = c2 if num_downs == 1 else c3 if num_downs == 2 else c4
        self.mid1 = ConvBlock(bottleneck_channels, bottleneck_channels, time_dim, dropout=dropout)
        self.attn = AttentionBlock(bottleneck_channels) if use_attention else nn.Identity()
        self.mid2 = ConvBlock(bottleneck_channels, bottleneck_channels, time_dim, dropout=dropout)

        if num_downs >= 3:
            self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1)
            self.dec3 = ConvBlock(c3 + c3, c3, time_dim, dropout=dropout)

        if num_downs >= 2:
            self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
            self.dec2 = ConvBlock(c2 + c2, c2, time_dim, dropout=dropout)

        self.up = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(c1 + c1, c1, time_dim, dropout=dropout)
        self.out = nn.Conv2d(c1, out_channels, kernel_size=1)

    @staticmethod
    def _match_spatial(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """当输入尺寸不是 2 的整数幂时，对齐上采样特征和 skip 的空间尺寸。"""

        if x.shape[-2:] == skip.shape[-2:]:
            return x
        return F.interpolate(x, size=skip.shape[-2:], mode="nearest")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t)

        # 编码阶段保留多尺度 skip connection，帮助恢复边缘和局部结构。
        skip1 = self.enc1(x, time_emb)
        h = F.silu(self.down(skip1))
        skip2 = self.enc2(h, time_emb)

        if self.num_downs >= 2:
            h = F.silu(self.down2(skip2))
            skip3 = self.enc3(h, time_emb)
        else:
            skip3 = None

        if self.num_downs >= 3:
            h = F.silu(self.down3(skip3))
            skip4 = self.enc4(h, time_emb)
        else:
            skip4 = None

        if self.num_downs == 1:
            h = skip2
        elif self.num_downs == 2:
            h = skip3
        else:
            h = skip4

        # 瓶颈层在低分辨率上聚合全局上下文；attention 可增强空间长程关系。
        h = self.mid1(h, time_emb)
        h = self.attn(h)
        h = self.mid2(h, time_emb)

        if self.num_downs >= 3:
            h = self.up3(h)
            h = self._match_spatial(h, skip3)
            h = torch.cat([h, skip3], dim=1)
            h = self.dec3(h, time_emb)

        if self.num_downs >= 2:
            h = self.up2(h)
            h = self._match_spatial(h, skip2)
            h = torch.cat([h, skip2], dim=1)
            h = self.dec2(h, time_emb)

        h = self.up(h)
        h = self._match_spatial(h, skip1)
        h = torch.cat([h, skip1], dim=1)
        h = self.dec1(h, time_emb)
        return self.out(h)


if __name__ == "__main__":
    model = TinyUNet(model_channels=64, num_downs=3, use_attention=True)
    x = torch.randn(2, 1, 64, 64)
    t = torch.rand(2)
    y = model(x, t)
    print(y.shape)
