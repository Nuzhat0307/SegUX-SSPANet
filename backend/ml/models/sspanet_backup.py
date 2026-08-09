"""
SSPANet — Strip-Style Pooling Attention Network.

This is the core attention module from the research paper:
"Enhancing Brain Tumor Classification with a Novel Attention-Based
Explainable Deep Learning Framework"

SSPANet combines:
1. Channel attention (squeeze-and-excitation style)
2. Strip pooling (horizontal and vertical pooling for long-range dependencies)
3. Style pooling (variance-based pooling for richer feature aggregation)

Adapted from the paper's architecture with standard PyTorch primitives.

Reference: The original SSPANet design uses strip-style pooling to capture
long-range spatial dependencies that standard global pooling misses. This
implementation faithfully reproduces the three pooling streams and their
fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class StripPooling(nn.Module):
    """
    Strip pooling module — captures long-range horizontal and vertical
    dependencies through separate 1D pooling operations.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1d_h = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv1d_v = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv2d = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Horizontal strip pooling: (B, C, H, W) -> (B, C, W) -> (B, C, 1, W)
        h_pool = F.adaptive_avg_pool2d(x, (1, w)).squeeze(2)
        h_pool = self.conv1d_h(h_pool).unsqueeze(2)  # (B, C, 1, W)

        # Vertical strip pooling: (B, C, H, W) -> (B, C, H) -> (B, C, H, 1)
        v_pool = F.adaptive_avg_pool2d(x, (h, 1)).squeeze(3)
        v_pool = self.conv1d_v(v_pool).unsqueeze(3)  # (B, C, H, 1)

        # Expand and combine
        h_pool = h_pool.expand(b, -1, h, w)
        v_pool = v_pool.expand(b, -1, h, w)

        # Fusion
        combined = self.bn(self.conv2d(x) + h_pool + v_pool)
        return self.sigmoid(combined) * x


class StylePooling(nn.Module):
    """
    Style pooling — captures variance-based feature statistics
    for richer texture and style representation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean (content) and std (style) pooling
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        style = torch.cat([mean, std], dim=1)
        style = self.conv(style)
        return self.sigmoid(self.bn(style)) * x


class SSPANetBlock(nn.Module):
    """
    SSPANet attention block with optional segmentation guidance.

    The segmentation guidance map is converted into a spatial
    attention gate and applied before the three SSPANet streams.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.channel_attn = ChannelAttention(channels)
        self.strip_pool = StripPooling(channels, channels)
        self.style_pool = StylePooling(channels, channels)

        # Segmentation-guided spatial attention
        self.guidance_conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, guidance=None) -> torch.Tensor:

        # --------------------------------------------------
        # Segmentation-guided attention
        # --------------------------------------------------
        if guidance is not None:

            guidance = F.interpolate(
                guidance,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            spatial_gate = self.guidance_conv(guidance)

            # Residual guidance:
            # preserves original features while emphasizing
            # regions predicted as tumor.
            x = x * (1.0 + spatial_gate)

        # --------------------------------------------------
        # SSPANet attention streams
        # --------------------------------------------------
        ca = self.channel_attn(x)
        sp = self.strip_pool(x)
        stp = self.style_pool(x)

        fused = self.fuse_conv(
            torch.cat([ca, sp, stp], dim=1)
        )

        return x + fused


class SSPANet(nn.Module):
    """
    SSPANet attention module with optional segmentation guidance.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = SSPANetBlock(channels)

    def forward(self, x: torch.Tensor, guidance=None):
        return self.block(x, guidance)